"""Tests du decoupage en lots et de l'appariement chunk <-> vecteur.

Aucun appel reseau : le transport HTTP est remplace par un double qui
repond comme l'API, et qui enregistre ce qu'on lui a envoye.
"""

from __future__ import annotations

import pytest

from legalrag.chunking import Chunk
from legalrag.embeddings import (
    EmbeddingError,
    PayloadTooLargeError,
    RateLimiter,
    VoyageEmbedder,
)

DIMENSION = 4


def chunk(index: int, tokens: int = 100, document: str = "code") -> Chunk:
    return Chunk(
        chunk_id=f"{document}-{index}",
        document=document,
        document_id=document,
        chunk_index=index,
        text=f"texte {document} {index}",
        page_start=index,
        page_end=index,
        pages=(index,),
        timestamp="",
        source_filenames=(f"{document}-{index}.txt",),
        token_count=tokens,
    )


def make_embedder(contextualized: bool = True, **kwargs) -> VoyageEmbedder:
    return VoyageEmbedder(
        api_key="cle-de-test",
        model="voyage-context-4" if contextualized else "voyage-4",
        dimension=DIMENSION,
        contextualized=contextualized,
        exact_tokenizer=True,
        **kwargs,
    )


class FakeAPI:
    """Repond comme l'API et memorise les charges utiles recues."""

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def __call__(self, endpoint: str, payload: dict) -> dict:
        self.payloads.append(payload)

        if endpoint == "contextualizedembeddings":
            data = [
                {
                    "index": i,
                    "data": [
                        {"index": j, "embedding": [float(j)] * DIMENSION}
                        for j in range(len(group))
                    ],
                }
                for i, group in enumerate(payload["inputs"])
            ]
            total = sum(len(g) for g in payload["inputs"])
        else:
            data = [
                {"index": j, "embedding": [float(j)] * DIMENSION}
                for j in range(len(payload["input"]))
            ]
            total = len(payload["input"])

        return {"data": data, "usage": {"total_tokens": total * 10}}


# -- mode contextualise ----------------------------------------------------


def test_chaque_chunk_recoit_son_vecteur(monkeypatch):
    embedder = make_embedder()
    monkeypatch.setattr(embedder, "_post", FakeAPI())

    groups = [[chunk(i, document="code-a") for i in range(3)], [chunk(0, document="code-b")]]
    resultats = [item for batch in embedder.embed(groups) for item in batch]

    assert len(resultats) == 4
    assert [r.chunk.chunk_id for r in resultats] == ["code-a-0", "code-a-1", "code-a-2", "code-b-0"]
    for item in resultats:
        assert len(item.embedding) == DIMENSION


def test_les_chunks_d_un_document_partent_groupes(monkeypatch):
    """C'est la condition pour que le modele voie le contexte du document."""
    embedder = make_embedder()
    fake = FakeAPI()
    monkeypatch.setattr(embedder, "_post", fake)

    groups = [[chunk(i, document="code-a") for i in range(3)]]
    list(embedder.embed(groups))

    assert fake.payloads[0]["inputs"] == [["texte code-a 0", "texte code-a 1", "texte code-a 2"]]


def test_document_trop_long_est_scinde(monkeypatch):
    """Un document depassant le contexte du modele est coupe en sous-groupes."""
    embedder = make_embedder()
    fake = FakeAPI()
    monkeypatch.setattr(embedder, "_post", fake)

    # 10 chunks de 5000 tokens = 50 000 > 32 000 tokens de contexte.
    groups = [[chunk(i, tokens=5000) for i in range(10)]]
    resultats = [item for batch in embedder.embed(groups) for item in batch]

    assert len(resultats) == 10
    envoyes = [inner for payload in fake.payloads for inner in payload["inputs"]]
    assert len(envoyes) > 1, "le document aurait du etre scinde"


def test_budget_de_tokens_respecte_par_requete(monkeypatch):
    embedder = make_embedder()
    fake = FakeAPI()
    monkeypatch.setattr(embedder, "_post", fake)

    # 40 documents de 5000 tokens : le budget de 120 000 tokens par requete
    # impose plusieurs allers-retours.
    groups = [[chunk(0, tokens=5000, document=f"doc{i}")] for i in range(40)]
    list(embedder.embed(groups))

    assert len(fake.payloads) > 1
    for payload in fake.payloads:
        envoye = sum(len(inner) for inner in payload["inputs"])
        assert envoye <= 1000


def test_dimension_transmise_a_l_api(monkeypatch):
    embedder = make_embedder()
    fake = FakeAPI()
    monkeypatch.setattr(embedder, "_post", fake)
    list(embedder.embed([[chunk(0)]]))

    assert fake.payloads[0]["output_dimension"] == DIMENSION
    assert fake.payloads[0]["input_type"] == "document"
    assert fake.payloads[0]["model"] == "voyage-context-4"


def test_lot_trop_gros_est_scinde_et_rejoue(monkeypatch):
    """Si le serveur refuse pour la taille, on coupe en deux au lieu d'echouer."""
    fake = FakeAPI()
    appels = {"n": 0}

    def post(endpoint, payload):
        appels["n"] += 1
        if len(payload["inputs"]) > 1:
            raise PayloadTooLargeError("total tokens exceed limit")
        return fake(endpoint, payload)

    embedder = make_embedder()
    monkeypatch.setattr(embedder, "_post", post)

    groups = [[chunk(0, document=f"doc{i}")] for i in range(4)]
    resultats = [item for batch in embedder.embed(groups) for item in batch]

    assert len(resultats) == 4
    assert appels["n"] > 1


def test_reponse_incoherente_est_signalee(monkeypatch):
    """Un vecteur manquant doit lever, pas produire un index mal apparie."""
    embedder = make_embedder()
    monkeypatch.setattr(
        embedder,
        "_post",
        lambda endpoint, payload: {"data": [], "usage": {"total_tokens": 0}},
    )
    with pytest.raises(EmbeddingError, match="incoherente"):
        list(embedder.embed([[chunk(0)]]))


def test_tokens_factures_cumules(monkeypatch):
    embedder = make_embedder()
    monkeypatch.setattr(embedder, "_post", FakeAPI())
    list(embedder.embed([[chunk(i) for i in range(3)]]))
    assert embedder.total_tokens_billed == 30


# -- mode standard ---------------------------------------------------------


def test_mode_standard_aplatit_les_documents(monkeypatch):
    embedder = make_embedder(contextualized=False)
    fake = FakeAPI()
    monkeypatch.setattr(embedder, "_post", fake)

    groups = [[chunk(0, document="a"), chunk(1, document="a")], [chunk(0, document="b")]]
    resultats = [item for batch in embedder.embed(groups) for item in batch]

    assert len(resultats) == 3
    assert "input" in fake.payloads[0]
    assert fake.payloads[0]["input"] == ["texte a 0", "texte a 1", "texte b 0"]


# -- limiteur de debit -----------------------------------------------------


def test_limiteur_laisse_passer_sous_le_plafond():
    limiter = RateLimiter(max_requests_per_minute=10, max_tokens_per_minute=10_000)
    for _ in range(5):
        limiter.acquire(100)  # ne doit pas bloquer


def test_limiteur_accepte_un_lot_plus_gros_que_le_plafond():
    """Un lot unique depassant le budget ne doit pas bloquer indefiniment."""
    limiter = RateLimiter(max_requests_per_minute=10, max_tokens_per_minute=100)
    limiter.acquire(5_000)
