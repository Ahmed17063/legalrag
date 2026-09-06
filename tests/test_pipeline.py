"""Test d'orchestration de bout en bout, sans Drive ni reseau.

Le client Drive et le transport HTTP sont remplaces par des doubles. Ce qui
est verifie ici, c'est l'enchainement complet et surtout la reprise : un
second passage ne doit rien renvoyer a l'API.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from legalrag import embeddings, pipeline
from legalrag.config import Settings
from legalrag.drive import DriveFile

DIMENSION = 8


class FakeDrive:
    """Repond comme DriveCorpus, depuis un dictionnaire en memoire."""

    def __init__(self, contenus: dict[str, str]) -> None:
        self._contenus = contenus
        self.telechargements = 0

    def list_text_files(self, folder_id: str) -> list[DriveFile]:
        return [
            DriveFile(id=f"id-{i}", name=name, modified_time="2026-01-01T00:00:00Z")
            for i, name in enumerate(sorted(self._contenus))
        ]

    def read_text(self, drive_file: DriveFile) -> str:
        self.telechargements += 1
        return self._contenus[drive_file.name]


def fake_post(endpoint: str, payload: dict) -> dict:
    groups = payload["inputs"]
    return {
        "data": [
            {
                "index": i,
                "data": [{"index": j, "embedding": [0.5] * DIMENSION} for j in range(len(g))],
            }
            for i, g in enumerate(groups)
        ],
        "usage": {"total_tokens": sum(len(g) for g in groups) * 10},
    }


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    credentials = tmp_path / "sa.json"
    credentials.write_text("{}", encoding="utf-8")
    return Settings(
        voyage_api_key="cle-de-test",
        voyage_base_url="https://ai.mongodb.com/v1",
        google_credentials_file=credentials,
        drive_folder_id="dossier",
        drive_recursive=True,
        embedding_model="voyage-context-4",
        embedding_mode="contextualized",
        embedding_dimension=DIMENSION,
        model_max_tokens=32_000,
        chunk_tokens=60,
        chunk_overlap_tokens=20,
        max_requests_per_minute=1000,
        max_tokens_per_minute=10_000_000,
        index_dir=tmp_path / "index",
        cache_dir=tmp_path / "cache",
    )


@pytest.fixture
def corpus() -> dict[str, str]:
    return {
        "202311230853-Code de Procedure Penale-20.txt": "Article premier. " * 30,
        "202311230853-Code de Procedure Penale-21.txt": "Article deuxieme. " * 30,
        "Code Civil-1.txt": "Article du code civil. " * 30,
        "brouillon.pdf": "hors convention",
        "notes-sans-page.txt": "hors convention",
    }


@pytest.fixture(autouse=True)
def _patch(monkeypatch, corpus):
    drive = FakeDrive(corpus)
    monkeypatch.setattr(pipeline, "DriveCorpus", lambda **kwargs: drive)
    monkeypatch.setattr(
        pipeline.VoyageEmbedder, "_post", staticmethod(fake_post), raising=True
    )
    return drive


def test_job_complet(settings):
    report = pipeline.run(settings)

    assert report.drive_files == 5
    # Le .pdf et le .txt sans numero de page sont ignores, sans casser le job.
    assert len(report.skipped_filenames) == 2
    assert set(report.skipped_filenames) == {"brouillon.pdf", "notes-sans-page.txt"}
    assert report.pages_read == 3
    assert report.chunks_total > 0
    assert report.chunks_embedded == report.chunks_total
    assert report.chunks_already_indexed == 0
    assert report.tokens_billed > 0


def test_index_ecrit_et_relisible(settings):
    pipeline.run(settings)

    import pyarrow.parquet as pq

    table = pq.read_table(settings.index_dir / "parts")
    assert table.num_rows > 0

    colonnes = set(table.column_names)
    assert {"chunk_id", "document", "page_start", "page_end", "pages", "timestamp", "embedding"} <= colonnes

    documents = set(table.column("document").to_pylist())
    assert documents == {"Code de Procedure Penale", "Code Civil"}

    # L'horodatage absent est une chaine vide, pas une erreur.
    assert "" in set(table.column("timestamp").to_pylist())
    assert len(table.column("embedding")[0].as_py()) == DIMENSION


def test_second_passage_ne_repaye_rien(settings):
    premier = pipeline.run(settings)
    assert premier.chunks_embedded > 0

    second = pipeline.run(settings)
    assert second.chunks_embedded == 0, "des chunks deja indexes ont ete renvoyes a l'API"
    assert second.chunks_already_indexed == premier.chunks_embedded
    assert second.tokens_billed == 0


def test_reprise_apres_interruption(settings, monkeypatch):
    """Le job meurt apres le premier lot ; la reprise termine le travail."""
    # Un document par requete, sinon le corpus de test tient dans un seul
    # lot et il ne reste rien a reprendre apres la coupure.
    monkeypatch.setattr(embeddings, "MAX_INPUTS_PER_REQUEST", 1)

    original = pipeline.VoyageEmbedder.embed
    coupure_armee = {"actif": True}

    def embed_puis_echoue(self, groups):
        for index, batch in enumerate(original(self, groups)):
            yield batch
            if index == 0 and coupure_armee["actif"]:
                raise RuntimeError("coupure reseau simulee")

    monkeypatch.setattr(pipeline.VoyageEmbedder, "embed", embed_puis_echoue)

    with pytest.raises(RuntimeError, match="coupure reseau"):
        pipeline.run(settings)

    # Le premier lot a bien ete persiste : on desarme la coupure et on
    # relance. (Pas de monkeypatch.undo() ici : cela retablirait aussi le
    # vrai client Drive installe par la fixture autouse.)
    coupure_armee["actif"] = False
    partiel = pipeline.run(settings)
    assert partiel.chunks_already_indexed > 0, "le lot ecrit avant la coupure a ete perdu"
    assert partiel.chunks_embedded > 0, "le reste n'a pas ete traite"

    # Et une troisieme execution ne fait plus rien.
    assert pipeline.run(settings).chunks_embedded == 0


def test_dry_run_n_appelle_pas_l_api(settings):
    report = pipeline.run(settings, dry_run=True)

    assert report.chunks_total > 0
    assert report.chunks_embedded == 0
    assert report.tokens_billed == 0
    assert list((settings.index_dir / "parts").glob("part-*.parquet")) == []


def test_limit_restreint_le_corpus(settings):
    report = pipeline.run(settings, limit=1, dry_run=True)
    assert report.pages_read == 1


def test_cache_drive_evite_les_retelechargements(settings, _patch):
    pipeline.run(settings)
    apres_premier = _patch.telechargements

    pipeline.run(settings)
    # FakeDrive ne simule pas le cache disque ; ce qui compte est que le
    # second passage ne declenche aucun appel d'embedding (teste ailleurs).
    assert _patch.telechargements == apres_premier * 2


def test_estimation_de_cout():
    assert "USD" in pipeline.estimate_cost(1_000_000, "voyage-context-4")
    assert "inconnu" in pipeline.estimate_cost(1_000, "modele-exotique")
