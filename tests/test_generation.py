"""Tests de la generation : ancrage, citations, garde-fous."""

from __future__ import annotations

from dataclasses import replace

import pytest

from legalrag.generation import (
    AUCUNE_REPONSE,
    SYSTEM_PROMPT,
    Answer,
    GenerationError,
    Generator,
    build_context,
    extract_citations,
)
from legalrag.retrieval import Passage


def passage(article="326", texte=None, document="Code de Procedure Penale") -> Passage:
    return Passage(
        text=texte or f"Article {article}- Contenu de l'article {article}.",
        document=document,
        document_id=document.lower().replace(" ", "-"),
        page_start=127,
        page_end=128,
        pages=(127, 128),
        timestamp="202311230853",
        chunk_id=f"chunk-{article}",
        article=article,
        vector_score=0.4,
        rerank_score=0.9,
    )


# ---------------------------------------------------------------------------
# Mise en forme du contexte
# ---------------------------------------------------------------------------


def test_contexte_numerote():
    contexte = build_context([passage("326"), passage("327")])

    assert contexte.startswith("[1] Code de Procedure Penale, art. 326 (p. 127-128)")
    assert "[2] Code de Procedure Penale, art. 327" in contexte


def test_contexte_normalise_les_blancs():
    """L'OCR laisse des retours a la ligne parasites au milieu des phrases."""
    contexte = build_context([passage(texte="Article 1-\n\n  Un   texte\n mal   espace.")])
    assert "Article 1- Un texte mal espace." in contexte


def test_consigne_impose_les_citations_et_l_aveu_d_ignorance():
    assert "[1]" in SYSTEM_PROMPT
    assert AUCUNE_REPONSE in SYSTEM_PROMPT
    assert "UNIQUEMENT" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Extraction des citations
# ---------------------------------------------------------------------------


def test_citations_valides():
    valides, invalides = extract_citations("Selon [1] et [3], la reponse est non.", 3)
    assert valides == (1, 3)
    assert invalides == ()


def test_citations_hors_plage_signalees():
    """Citer [7] quand on a fourni 3 extraits est le signe d'une derive."""
    valides, invalides = extract_citations("D'apres [1] et [7].", 3)
    assert valides == (1,)
    assert invalides == (7,)


def test_citations_dedupliquees_et_triees():
    valides, _ = extract_citations("[2] puis [1] puis encore [2].", 3)
    assert valides == (1, 2)


def test_aucune_citation():
    assert extract_citations("Une reponse sans la moindre source.", 3) == ((), ())


def test_zero_est_hors_plage():
    _, invalides = extract_citations("Voir [0].", 3)
    assert invalides == (0,)


# ---------------------------------------------------------------------------
# Qualification de la reponse
# ---------------------------------------------------------------------------


def reponse(**kwargs) -> Answer:
    base = dict(
        question="q",
        text="Non [1].",
        passages=[passage("326"), passage("327")],
        cited=(1,),
    )
    return Answer(**{**base, **kwargs})


def test_reponse_etayee():
    assert reponse().grounded


def test_reponse_sans_citation_non_etayee():
    assert not reponse(text="Non.", cited=()).grounded


def test_citation_invalide_rend_la_reponse_non_etayee():
    assert not reponse(invalid_citations=(9,)).grounded


def test_aveu_d_ignorance_detecte():
    assert reponse(text=AUCUNE_REPONSE, cited=()).declined


def test_sources_listees():
    sources = reponse(cited=(1, 2)).sources()
    assert sources == [
        "[1] Code de Procedure Penale, art. 326 (p. 127-128)",
        "[2] Code de Procedure Penale, art. 327 (p. 127-128)",
    ]


def test_sources_ignorent_les_numeros_hors_plage():
    assert reponse(cited=(1,), invalid_citations=(9,)).sources() == [
        "[1] Code de Procedure Penale, art. 326 (p. 127-128)"
    ]


# ---------------------------------------------------------------------------
# Appel au modele, sans reseau
# ---------------------------------------------------------------------------


class FauxGroq:
    """Repond comme l'API Groq, et memorise la charge utile recue."""

    def __init__(self, contenu: str):
        self.contenu = contenu
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(payload)
        return {
            "choices": [{"message": {"content": self.contenu}}],
            "model": payload["model"],
            "usage": {"prompt_tokens": 900, "completion_tokens": 60},
        }


def generateur(settings, contenu, monkeypatch):
    gen = Generator(settings)
    faux = FauxGroq(contenu)
    monkeypatch.setattr(gen, "_post", faux)
    return gen, faux


def test_reponse_complete(settings, monkeypatch):
    gen, _ = generateur(settings, "Non, l'article 326 l'interdit [1].", monkeypatch)
    r = gen.answer("Peut-on rejuger ?", [passage("326"), passage("327")])

    assert r.cited == (1,)
    assert r.grounded
    assert r.prompt_tokens == 900
    assert r.completion_tokens == 60
    assert r.model == "openai/gpt-oss-120b"


def test_temperature_nulle(settings, monkeypatch):
    """Aucune raison de vouloir de la variete quand on restitue du droit."""
    gen, faux = generateur(settings, "Reponse [1].", monkeypatch)
    gen.answer("q", [passage()])
    assert faux.payloads[0]["temperature"] == 0


def test_les_extraits_sont_transmis(settings, monkeypatch):
    gen, faux = generateur(settings, "Reponse [1].", monkeypatch)
    gen.answer("Ma question ?", [passage("326")])

    messages = faux.payloads[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "Article 326-" in messages[1]["content"]
    assert "Ma question ?" in messages[1]["content"]


def test_sans_extrait_le_modele_n_est_pas_appele(settings, monkeypatch):
    """Rien a citer : inutile de payer un appel pour une reponse inventee."""
    gen, faux = generateur(settings, "peu importe", monkeypatch)
    r = gen.answer("q", [])

    assert r.text == AUCUNE_REPONSE
    assert r.declined
    assert faux.payloads == []


def test_citation_inexistante_signalee(settings, monkeypatch, caplog):
    import logging

    gen, _ = generateur(settings, "Selon [1] et [9], oui.", monkeypatch)
    with caplog.at_level(logging.WARNING):
        r = gen.answer("q", [passage()])

    assert r.invalid_citations == (9,)
    assert not r.grounded
    assert "inexistants" in caplog.text


def test_reponse_sans_citation_signalee(settings, monkeypatch, caplog):
    import logging

    gen, _ = generateur(settings, "Oui, certainement.", monkeypatch)
    with caplog.at_level(logging.WARNING):
        r = gen.answer("q", [passage()])

    assert not r.grounded
    assert "sans aucune citation" in caplog.text


def test_aveu_d_ignorance_non_signale(settings, monkeypatch, caplog):
    """Reconnaitre qu'on ne sait pas n'est pas une anomalie."""
    import logging

    gen, _ = generateur(settings, AUCUNE_REPONSE, monkeypatch)
    with caplog.at_level(logging.WARNING):
        r = gen.answer("q", [passage()])

    assert r.declined
    assert "sans aucune citation" not in caplog.text


def test_cle_absente_refusee(settings):
    sans_cle = replace(settings, groq_api_key=None)
    with pytest.raises(GenerationError, match="GROQ_API_KEY"):
        Generator(sans_cle)


def test_sortie_json_serialisable(settings, monkeypatch):
    import json

    gen, _ = generateur(settings, "Non [1].", monkeypatch)
    r = gen.answer("q", [passage()])

    rejoue = json.loads(json.dumps(r.as_dict(), ensure_ascii=False))
    assert rejoue["grounded"] is True
    assert rejoue["sources"] == ["[1] Code de Procedure Penale, art. 326 (p. 127-128)"]
