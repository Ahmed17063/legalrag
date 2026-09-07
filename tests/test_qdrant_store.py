"""Tests du chargement dans Qdrant.

Pas de doublure ici : `QdrantClient(":memory:")` execute le vrai moteur en
memoire. Ce qui est verifie est donc le comportement reel, y compris la
recherche.
"""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from legalrag.chunking import Chunk
from legalrag.embeddings import EmbeddedChunk
from legalrag.index_store import IndexStore, make_manifest
from legalrag.qdrant_store import (
    QdrantIndex,
    QdrantStoreError,
    QdrantTarget,
    point_id,
    read_dimension,
)

DIMENSION = 8


def vecteur(graine: int) -> list[float]:
    """Vecteur norme, distinct pour chaque graine."""
    base = [0.0] * DIMENSION
    base[graine % DIMENSION] = 1.0
    return base


def chunk(chunk_id: str, index: int = 0, document: str = "Code Penal") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document=document,
        document_id=document.lower().replace(" ", "-"),
        chunk_index=index,
        text=f"Article {index} du {document}.",
        page_start=20 + index,
        page_end=21 + index,
        pages=(20 + index, 21 + index),
        timestamp="202311230853",
        source_filenames=(f"{document}-{20 + index}.txt",),
        token_count=100 + index,
    )


@pytest.fixture
def index_parquet(tmp_path):
    """Un petit index Parquet, construit comme le ferait le pipeline."""
    store = IndexStore(
        tmp_path / "index",
        make_manifest("voyage-context-4", DIMENSION, "contextualized", 800, 150),
    )
    store.append(
        [
            EmbeddedChunk(chunk=chunk("aaa", 0), embedding=vecteur(0)),
            EmbeddedChunk(chunk=chunk("bbb", 1), embedding=vecteur(1)),
            EmbeddedChunk(chunk=chunk("ccc", 2), embedding=vecteur(2)),
        ]
    )
    return store.compact()


@pytest.fixture
def client():
    c = QdrantClient(":memory:")
    yield c
    c.close()


@pytest.fixture
def index(client):
    return QdrantIndex(client, "test", DIMENSION)


# -- identifiants ----------------------------------------------------------


def test_point_id_deterministe():
    """Condition de l'idempotence : recharger ne doit pas dupliquer."""
    assert point_id("aaa") == point_id("aaa")
    assert point_id("aaa") != point_id("bbb")


def test_point_id_est_un_uuid_valide():
    import uuid

    uuid.UUID(point_id("aaa"))  # ne doit pas lever


# -- dimension -------------------------------------------------------------


def test_dimension_lue_dans_l_index(index_parquet):
    """La dimension vient de la source, jamais de la configuration."""
    assert read_dimension(index_parquet) == DIMENSION


# -- collection ------------------------------------------------------------


def test_creation_de_collection(index):
    assert index.ensure_collection() is True
    assert index.ensure_collection() is False  # deja la


def test_recreate_vide_la_collection(index, index_parquet):
    index.ensure_collection()
    index.upsert(index_parquet)
    assert index.count() == 3

    index.ensure_collection(recreate=True)
    assert index.count() == 0


def test_dimension_incompatible_refusee(client, index_parquet):
    QdrantIndex(client, "test", DIMENSION).ensure_collection()

    autre = QdrantIndex(client, "test", DIMENSION * 2)
    with pytest.raises(QdrantStoreError, match="dimensions"):
        autre.ensure_collection()


# -- chargement ------------------------------------------------------------


def test_chargement_complet(index, index_parquet):
    index.ensure_collection()
    charges = index.upsert(index_parquet)

    assert charges == 3
    assert index.count() == 3
    assert index.existing_chunk_ids() == {"aaa", "bbb", "ccc"}


def test_metadonnees_presentes_en_payload(index, index_parquet, client):
    index.ensure_collection()
    index.upsert(index_parquet)

    points, _ = client.scroll("test", limit=10, with_payload=True)
    payload = next(p.payload for p in points if p.payload["chunk_id"] == "aaa")

    assert payload["document"] == "Code Penal"
    assert payload["document_id"] == "code-penal"
    assert payload["page_start"] == 20
    assert payload["page_end"] == 21
    assert payload["pages"] == [20, 21]
    assert payload["timestamp"] == "202311230853"
    assert payload["text"].startswith("Article 0")
    # Le vecteur n'a rien a faire dans le payload : il est deja le vecteur.
    assert "embedding" not in payload


def test_rechargement_idempotent(index, index_parquet):
    """Recharger deux fois met a jour, cela ne duplique pas."""
    index.ensure_collection()
    index.upsert(index_parquet)
    index.upsert(index_parquet)

    assert index.count() == 3


def test_chargement_par_lots(index, index_parquet):
    """Un lot plus petit que l'index ne doit rien perdre."""
    index.ensure_collection()
    assert index.upsert(index_parquet, batch_size=2) == 3
    assert index.count() == 3


# -- entretien -------------------------------------------------------------


def test_prune_supprime_les_points_obsoletes(index, index_parquet):
    index.ensure_collection()
    index.upsert(index_parquet)

    supprimes = index.prune({"aaa", "ccc"})

    assert supprimes == 1
    assert index.existing_chunk_ids() == {"aaa", "ccc"}
    assert index.count() == 2


def test_prune_sans_rien_a_supprimer(index, index_parquet):
    index.ensure_collection()
    index.upsert(index_parquet)
    assert index.prune({"aaa", "bbb", "ccc"}) == 0
    assert index.count() == 3


# -- la recherche fonctionne vraiment --------------------------------------


def test_recherche_retrouve_le_bon_chunk(index, index_parquet, client):
    """Verification de bout en bout : le vecteur du chunk 'bbb' doit
    ramener 'bbb' en tete."""
    index.ensure_collection()
    index.upsert(index_parquet)

    resultats = client.query_points("test", query=vecteur(1), limit=3).points

    assert resultats[0].payload["chunk_id"] == "bbb"
    assert resultats[0].score > resultats[1].score


def test_filtrage_par_document(index, index_parquet, client):
    """Les index de payload permettent de restreindre a un document."""
    from qdrant_client import models

    index.ensure_collection()
    index.upsert(index_parquet)

    resultats = client.query_points(
        "test",
        query=vecteur(0),
        query_filter=models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchValue(value="code-penal"))]
        ),
        limit=10,
    ).points
    assert len(resultats) == 3

    aucun = client.query_points(
        "test",
        query=vecteur(0),
        query_filter=models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchValue(value="inexistant"))]
        ),
        limit=10,
    ).points
    assert aucun == []


# -- description de la cible -----------------------------------------------


def test_description_cible_embarquee(tmp_path):
    cible = QdrantTarget(collection="legalrag", path=tmp_path)
    assert cible.is_embedded
    assert "embarque" in cible.describe()


def test_description_cible_serveur():
    cible = QdrantTarget(collection="legalrag", url="http://localhost:6333")
    assert not cible.is_embedded
    assert "localhost:6333" in cible.describe()


def test_cle_api_signalee_sans_etre_affichee():
    cible = QdrantTarget(collection="c", url="https://x.cloud.qdrant.io", api_key="secret-a-ne-pas-afficher")
    description = cible.describe()
    assert "cle d'API" in description
    assert "secret-a-ne-pas-afficher" not in description


# -- resilience d'un chargement long ---------------------------------------


def test_incident_reseau_rejoue(index, index_parquet, monkeypatch):
    """Un cluster gratuit ferme parfois la connexion sans repondre."""
    from qdrant_client.http.exceptions import ResponseHandlingException

    index.ensure_collection()
    appels = {"n": 0}
    vrai = index._client.upsert

    def upsert(collection, points, wait=True):
        appels["n"] += 1
        if appels["n"] == 1:
            raise ResponseHandlingException("Server disconnected without sending a response")
        return vrai(collection, points=points, wait=wait)

    monkeypatch.setattr(index._client, "upsert", upsert)

    assert index.upsert(index_parquet) == 3
    assert appels["n"] >= 2  # le lot a bien ete rejoue


def test_reprise_saute_ce_qui_est_deja_charge(index, index_parquet, client):
    """Relancer apres un echec ne doit pas tout recharger."""
    index.ensure_collection()
    index.upsert(index_parquet)

    # Deuxieme passage : tout est deja la, rien a envoyer.
    assert index.upsert(index_parquet) == 0
    assert index.count() == 3


def test_sans_reprise_tout_est_recharge(index, index_parquet):
    index.ensure_collection()
    index.upsert(index_parquet)
    assert index.upsert(index_parquet, reprendre=False) == 3


def test_echecs_consecutifs_interrompent(index, index_parquet, monkeypatch):
    """Si rien n'aboutit jamais, inutile de continuer."""
    from qdrant_client.http.exceptions import ResponseHandlingException

    index.ensure_collection()

    def toujours_casse(collection, points, wait=True):
        raise ResponseHandlingException("panne")

    monkeypatch.setattr(index._client, "upsert", toujours_casse)
    monkeypatch.setattr("legalrag.qdrant_store.MAX_ECHECS_CONSECUTIFS", 1)

    with pytest.raises(QdrantStoreError, match="consecutifs"):
        index.upsert(index_parquet, batch_size=1)


def test_erreur_non_transitoire_ne_se_rejoue_pas():
    from legalrag.qdrant_store import _est_transitoire

    class FauxTimeout(Exception):
        pass

    FauxTimeout.__name__ = "Timeout"
    assert _est_transitoire(FauxTimeout())
    assert not _est_transitoire(ValueError("collection introuvable"))
