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


def test_lot_borne_par_le_debit_autorise():
    """Un compte sans moyen de paiement est plafonne a 10 000 tokens/min.

    Un lot plus gros que ce plafond serait rejete a chaque tentative :
    aucun reessai ne peut le sauver. La taille de lot doit donc etre
    bornee par le debit, pas seulement par la limite d'un appel.
    """
    embedder = make_embedder(max_tokens_per_minute=10_000)
    assert embedder._token_budget <= 10_000

    embedder_large = make_embedder(max_tokens_per_minute=800_000)
    assert embedder_large._token_budget <= 120_000


def test_lots_respectent_le_plafond_de_debit():
    """Le decoupage en lots est teste directement.

    Passer par `embed` ferait dormir le limiteur une minute entre chaque
    lot : c'est le comportement voulu en production, pas dans un test.
    """
    embedder = make_embedder(max_tokens_per_minute=10_000)
    groups = [[chunk(0, tokens=3000, document=f"doc{i}")] for i in range(10)]

    lots = list(embedder._pack_contextual(groups))

    assert len(lots) > 1, "tout serait parti en un seul lot"
    for lot in lots:
        cout = sum(c.token_count for groupe in lot for c in groupe)
        assert cout <= 10_000, f"lot de {cout} tokens au-dessus du plafond par minute"


def test_document_unique_scinde_par_le_budget_de_requete():
    """Un seul document plus gros que le budget doit etre scinde.

    Sans cette borne, un document de 18 000 tokens partait entier vers une
    API plafonnee a 8 000 tokens/minute : refus a chaque tentative, et
    aucun reessai ne pouvait aboutir.
    """
    embedder = make_embedder(max_tokens_per_minute=8_000)
    document = [chunk(i, tokens=1000, document="code") for i in range(18)]

    lots = list(embedder._pack_contextual([document]))

    assert len(lots) > 1, "le document est parti d'un seul tenant"
    for lot in lots:
        cout = sum(c.token_count for groupe in lot for c in groupe)
        assert cout <= 8_000, f"lot de {cout} tokens au-dessus du plafond"

    # Aucun chunk perdu, aucun duplique.
    envoyes = [c.chunk_id for lot in lots for groupe in lot for c in groupe]
    assert sorted(envoyes) == sorted(c.chunk_id for c in document)


# -- resilience d'un job long ----------------------------------------------


class LimiteurMuet:
    """Limiteur qui n'attend jamais : les tests ne doivent pas dormir."""

    budget_tokens = 10_000

    def acquire(self, tokens):
        pass

    def penalize(self, *args, **kwargs):
        return self.budget_tokens


def embedder_sans_attente(monkeypatch, **kwargs):
    """Un embedder a petits lots, dont le limiteur ne temporise pas."""
    embedder = make_embedder(max_tokens_per_minute=10_000, **kwargs)
    muet = LimiteurMuet()
    monkeypatch.setattr(embedder, "_limiter", muet)
    monkeypatch.setattr(embedder._transport, "limiter", muet)
    return embedder


def test_un_lot_en_echec_n_arrete_pas_le_job(monkeypatch):
    """Un job de plusieurs heures traverse forcement un incident.

    Constate en conditions reelles : la machine s'est mise en veille, le
    lot en cours a echoue au reveil, et tout le job s'est arrete apres
    22 chunks sur 10 942.
    """
    from legalrag.voyage_http import RetryableAPIError

    embedder = embedder_sans_attente(monkeypatch)
    fake = FakeAPI()
    appels = {"n": 0}

    def post(endpoint, payload):
        appels["n"] += 1
        if appels["n"] == 2:
            raise RetryableAPIError("veille de la machine")
        return fake(endpoint, payload)

    monkeypatch.setattr(embedder, "_post", post)

    groups = [[chunk(0, tokens=4000, document=f"doc{i}")] for i in range(10)]
    resultats = [item for batch in embedder.embed(groups) for item in batch]

    assert embedder.failed_batches == 1
    # L'essentiel a ete traite malgre l'incident : seul le lot fautif est
    # perdu, et la reprise le rattrapera.
    assert len(resultats) == 10 - embedder.failed_chunks
    assert len(resultats) >= 8


def test_echecs_consecutifs_interrompent_le_job(monkeypatch):
    """Inutile de defiler des heures d'erreurs si rien n'aboutit jamais."""
    from legalrag.voyage_http import RetryableAPIError

    embedder = embedder_sans_attente(monkeypatch)

    def toujours_refuse(endpoint, payload):
        raise RetryableAPIError("quota refuse")

    monkeypatch.setattr(embedder, "_post", toujours_refuse)

    groups = [[chunk(0, tokens=4000, document=f"doc{i}")] for i in range(20)]
    with pytest.raises(EmbeddingError, match="consecutifs"):
        list(embedder.embed(groups))

    assert embedder.failed_batches == 5


def test_le_compteur_d_echecs_se_remet_a_zero(monkeypatch):
    """Un echec isole ne doit pas rapprocher le job de l'interruption."""
    from legalrag.voyage_http import RetryableAPIError

    embedder = embedder_sans_attente(monkeypatch)
    fake = FakeAPI()
    appels = {"n": 0}

    def post(endpoint, payload):
        appels["n"] += 1
        if appels["n"] % 3 == 0:
            raise RetryableAPIError("incident passager")
        return fake(endpoint, payload)

    monkeypatch.setattr(embedder, "_post", post)

    groups = [[chunk(0, tokens=4000, document=f"doc{i}")] for i in range(20)]
    resultats = [item for batch in embedder.embed(groups) for item in batch]

    # Un echec sur trois, jamais cinq d'affilee : le job va au bout sans
    # declencher l'interruption, et l'essentiel est plonge.
    assert embedder.failed_batches >= 2
    assert len(resultats) >= 10


def test_erreur_systematique_arrete_le_job(monkeypatch):
    """Une reponse incoherente signale un bug, pas un incident passager."""
    embedder = embedder_sans_attente(monkeypatch)
    monkeypatch.setattr(
        embedder, "_post", lambda e, p: {"data": [], "usage": {"total_tokens": 0}}
    )
    with pytest.raises(EmbeddingError, match="incoherente"):
        list(embedder.embed([[chunk(0)]]))


def test_penalite_de_debit_apres_un_refus():
    """Un 429 doit ralentir durablement, pas seulement declencher un reessai."""
    from legalrag.voyage_http import RateLimiter

    limiter = RateLimiter(max_requests_per_minute=200, max_tokens_per_minute=10_000)
    assert limiter.budget_tokens == 10_000
    assert limiter.penalize() == 7_500
    assert limiter.penalize() == 5_625


def test_penalite_bornee_par_un_plancher():
    from legalrag.voyage_http import RateLimiter

    limiter = RateLimiter(200, 10_000)
    for _ in range(30):
        limiter.penalize()
    assert limiter.budget_tokens == 2_000


def test_taille_des_lots_suit_le_debit_reduit():
    """Apres penalite, les lots suivants doivent retrecir."""
    embedder = make_embedder(max_tokens_per_minute=10_000)
    avant = embedder._token_budget
    embedder._transport.limiter.penalize()
    assert embedder._token_budget < avant
