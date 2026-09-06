"""Tests de l'interface web, sans reseau.

Les ressources partagees (`etat`) sont remplacees par des doubles : ce qui
est verifie ici est le contrat HTTP, pas la chaine RAG elle-meme, deja
couverte par ailleurs.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from legalrag.generation import Answer, GenerationError
from legalrag.retrieval import Passage, SearchResult
from legalrag.web import server as module_web


def passage(article="326") -> Passage:
    return Passage(
        text=f"Article {article}- Aucune personne acquittee legalement ne peut etre reprise.",
        document="Code de Procedure Penale",
        document_id="code-de-procedure-penale",
        page_start=127,
        page_end=128,
        pages=(127, 128),
        timestamp="202311230853",
        chunk_id="c1",
        article=article,
        vector_score=0.38,
        rerank_score=0.93,
    )


class FauxRetriever:
    def __init__(self, passages=None, erreur=None):
        self._passages = passages if passages is not None else [passage()]
        self._erreur = erreur
        self.appels = []
        self.ferme = False

    def search(self, question, **kwargs):
        if self._erreur:
            raise self._erreur
        self.appels.append((question, kwargs))
        return SearchResult(
            query=question,
            passages=list(self._passages),
            chunks_examined=8,
            passages_examined=49,
            reranked=kwargs.get("rerank", True),
        )

    def documents(self):
        return [
            {"document_id": "code-de-procedure-penale", "document": "Code de Procedure Penale", "chunks": 157},
            {"document_id": "code-du-statut-personnel", "document": "Code du Statut Personnel", "chunks": 53},
        ]

    def close(self):
        self.ferme = True


class FauxGenerator:
    def __init__(self, reponse=None, erreur=None):
        self._reponse = reponse
        self._erreur = erreur

    def answer(self, question, passages):
        if self._erreur:
            raise self._erreur
        return self._reponse or Answer(
            question=question,
            text="Non, l'article 326 l'interdit [1].",
            passages=passages,
            cited=(1,),
            model="openai/gpt-oss-120b",
            prompt_tokens=500,
            completion_tokens=40,
        )


@pytest.fixture
def client(settings, monkeypatch):
    """Un client HTTP avec un etat deja initialise, sans demarrage reel."""
    etat = module_web.etat
    monkeypatch.setattr(etat, "settings", settings)
    monkeypatch.setattr(etat, "retriever", FauxRetriever())
    monkeypatch.setattr(etat, "generator", FauxGenerator())
    monkeypatch.setattr(etat, "erreur", None)
    monkeypatch.setattr(etat, "documents", etat.retriever.documents())
    # Le cycle de vie recreerait les vraies ressources : on le neutralise.
    monkeypatch.setattr(etat, "demarrer", lambda: None)
    monkeypatch.setattr(etat, "arreter", lambda: None)

    with TestClient(module_web.app) as c:
        yield c


# ---------------------------------------------------------------------------
# Page et sante
# ---------------------------------------------------------------------------


def test_page_servie(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Assistant juridique" in r.text


def test_couleurs_du_drapeau_presentes(client):
    """Charte graphique mauritanienne : vert, or, rouge."""
    page = client.get("/").text
    assert "#00A95C" in page  # vert
    assert "#FFD700" in page  # or
    assert "#D01C1F" in page  # rouge


def test_sante(client):
    donnees = client.get("/api/health").json()
    assert donnees["pret"] is True
    assert donnees["generation"] is True
    assert donnees["erreur"] is None
    assert len(donnees["documents"]) == 2
    assert donnees["modele_embedding"] == "voyage-context-4"


# ---------------------------------------------------------------------------
# Interrogation
# ---------------------------------------------------------------------------


def test_question_repondue(client):
    r = client.post("/api/ask", json={"question": "Peut-on rejuger un acquitte ?"})
    assert r.status_code == 200

    d = r.json()
    assert d["answer"] == "Non, l'article 326 l'interdit [1]."
    assert d["grounded"] is True
    assert d["sources"] == ["[1] Code de Procedure Penale, art. 326 (p. 127-128)"]
    assert d["chunks_examined"] == 8
    assert d["passages_examined"] == 49
    assert "duree" in d


def test_filtre_par_document_transmis(client):
    client.post(
        "/api/ask",
        json={"question": "Une question", "document_id": "code-du-statut-personnel"},
    )
    _, kwargs = module_web.etat.retriever.appels[-1]
    assert kwargs["document_id"] == "code-du-statut-personnel"


def test_filtre_vide_devient_absence_de_filtre(client):
    """La liste deroulante envoie une chaine vide pour « tous les documents »."""
    client.post("/api/ask", json={"question": "Une question", "document_id": ""})
    _, kwargs = module_web.etat.retriever.appels[-1]
    assert kwargs["document_id"] is None


def test_question_trop_courte_refusee(client):
    assert client.post("/api/ask", json={"question": "ab"}).status_code == 422


def test_question_absente_refusee(client):
    assert client.post("/api/ask", json={}).status_code == 422


def test_top_n_hors_bornes_refuse(client):
    r = client.post("/api/ask", json={"question": "Une question", "top_n": 99})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Degradations
# ---------------------------------------------------------------------------


def test_sans_generation_les_extraits_sont_rendus(client, monkeypatch):
    """Sans cle Groq, la recherche reste utile : on renvoie les passages."""
    monkeypatch.setattr(module_web.etat, "generator", None)

    d = client.post("/api/ask", json={"question": "Une question"}).json()
    assert d["answer"] is None
    assert "GROQ_API_KEY" in d["raison"]
    assert len(d["passages"]) == 1


def test_service_non_initialise(client, monkeypatch):
    monkeypatch.setattr(module_web.etat, "retriever", None)
    monkeypatch.setattr(module_web.etat, "erreur", "Qdrant injoignable")

    r = client.post("/api/ask", json={"question": "Une question"})
    assert r.status_code == 503
    assert "Qdrant" in r.json()["detail"]


def test_echec_de_recherche_remonte(client, monkeypatch):
    monkeypatch.setattr(
        module_web.etat, "retriever", FauxRetriever(erreur=RuntimeError("Qdrant coupe"))
    )
    r = client.post("/api/ask", json={"question": "Une question"})
    assert r.status_code == 502
    assert "Qdrant coupe" in r.json()["detail"]


def test_echec_de_redaction_remonte(client, monkeypatch):
    monkeypatch.setattr(
        module_web.etat, "generator", FauxGenerator(erreur=GenerationError("quota Groq"))
    )
    r = client.post("/api/ask", json={"question": "Une question"})
    assert r.status_code == 502
    assert "quota Groq" in r.json()["detail"]


def test_reponse_non_etayee_signalee(client, monkeypatch):
    """Le navigateur doit pouvoir afficher l'avertissement."""
    monkeypatch.setattr(
        module_web.etat,
        "generator",
        FauxGenerator(
            reponse=Answer(
                question="q",
                text="Oui, certainement.",
                passages=[passage()],
                cited=(),
            )
        ),
    )
    d = client.post("/api/ask", json={"question": "Une question"}).json()
    assert d["grounded"] is False
    assert d["cited"] == []


def test_citations_inexistantes_signalees(client, monkeypatch):
    monkeypatch.setattr(
        module_web.etat,
        "generator",
        FauxGenerator(
            reponse=Answer(
                question="q",
                text="Selon [9], oui.",
                passages=[passage()],
                cited=(),
                invalid_citations=(9,),
            )
        ),
    )
    d = client.post("/api/ask", json={"question": "Une question"}).json()
    assert d["invalid_citations"] == [9]
    assert d["grounded"] is False
