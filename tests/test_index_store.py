"""Tests de l'index Parquet et de la reprise apres incident."""

from __future__ import annotations

import pytest

from legalrag.chunking import Chunk
from legalrag.embeddings import EmbeddedChunk
from legalrag.index_store import IndexError_, IndexStore, make_manifest

DIMENSION = 8


def manifest(model: str = "voyage-context-4", dimension: int = DIMENSION):
    return make_manifest(model, dimension, "contextualized", 800, 150)


def chunk(chunk_id: str, index: int = 0) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document="Code de Procedure Penale",
        document_id="code-de-procedure-penale",
        chunk_index=index,
        text="Article 706-73 du code.",
        page_start=20,
        page_end=21,
        pages=(20, 21),
        timestamp="202311230853",
        source_filenames=("Code de Procedure Penale-20.txt",),
        token_count=42,
    )


def embedded(chunk_id: str, index: int = 0) -> EmbeddedChunk:
    return EmbeddedChunk(chunk=chunk(chunk_id, index), embedding=[0.1] * DIMENSION)


def test_ecriture_et_relecture(tmp_path):
    store = IndexStore(tmp_path, manifest())
    store.append([embedded("a", 0), embedded("b", 1)])

    assert store.count() == 2
    assert store.existing_chunk_ids() == {"a", "b"}


def test_chaque_lot_produit_un_fragment_distinct(tmp_path):
    store = IndexStore(tmp_path, manifest())
    store.append([embedded("a")])
    store.append([embedded("b")])

    parts = sorted((tmp_path / "parts").glob("part-*.parquet"))
    assert len(parts) == 2
    assert store.existing_chunk_ids() == {"a", "b"}


def test_reprise_voit_ce_qui_a_deja_ete_ecrit(tmp_path):
    """Simule un arret puis un redemarrage sur le meme repertoire."""
    premier = IndexStore(tmp_path, manifest())
    premier.append([embedded("a"), embedded("b")])

    # Nouveau process, meme index.
    second = IndexStore(tmp_path, manifest())
    deja_fait = second.existing_chunk_ids()

    a_faire = [c for c in ("a", "b", "c") if c not in deja_fait]
    assert a_faire == ["c"], "un chunk deja plonge serait repaye"


def test_fragment_temporaire_ignore_a_la_reprise(tmp_path):
    """Un lot interrompu en pleine ecriture ne doit pas polluer l'index."""
    store = IndexStore(tmp_path, manifest())
    store.append([embedded("a")])

    # Ce que laisserait un arret brutal pendant `append`.
    (tmp_path / "parts" / "part-000099.parquet.part").write_bytes(b"donnees tronquees")

    assert IndexStore(tmp_path, manifest()).existing_chunk_ids() == {"a"}


def test_changer_de_modele_est_refuse(tmp_path):
    IndexStore(tmp_path, manifest(model="voyage-context-4"))
    with pytest.raises(IndexError_, match="incompatible"):
        IndexStore(tmp_path, manifest(model="voyage-law-2"))


def test_changer_de_dimension_est_refuse(tmp_path):
    IndexStore(tmp_path, manifest(dimension=DIMENSION))
    with pytest.raises(IndexError_, match="incompatible"):
        IndexStore(tmp_path, manifest(dimension=16))


def test_vecteur_de_mauvaise_dimension_refuse(tmp_path):
    store = IndexStore(tmp_path, manifest())
    mauvais = EmbeddedChunk(chunk=chunk("a"), embedding=[0.1] * (DIMENSION + 1))
    with pytest.raises(IndexError_, match="dimension"):
        store.append([mauvais])


def test_compactage(tmp_path):
    store = IndexStore(tmp_path, manifest())
    store.append([embedded("a")])
    store.append([embedded("b")])

    cible = store.compact(remove_parts=True)
    assert cible is not None and cible.is_file()

    import pyarrow.parquet as pq

    table = pq.read_table(cible)
    assert table.num_rows == 2
    assert set(table.column("chunk_id").to_pylist()) == {"a", "b"}
    # Les metadonnees survivent au compactage.
    assert table.column("pages").to_pylist() == [[20, 21], [20, 21]]
    assert table.column("timestamp").to_pylist() == ["202311230853"] * 2
    assert not list((tmp_path / "parts").glob("part-*.parquet"))


def test_lot_vide_n_ecrit_rien(tmp_path):
    store = IndexStore(tmp_path, manifest())
    assert store.append([]) is None
    assert store.count() == 0
