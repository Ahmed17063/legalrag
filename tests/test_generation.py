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
    modeles_disponibles,
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


def test_temperature_de_redaction_transmise(settings, monkeypatch):
    """Bridee, mais pas nulle : 0,1 par defaut pour la redaction."""
    gen, faux = generateur(settings, "Reponse [1].", monkeypatch)
    gen.answer("q", [passage()])
    assert faux.payloads[0]["temperature"] == 0.1


def test_temperature_configurable(settings, monkeypatch):
    gen, faux = generateur(
        replace(settings, generation_temperature=0.7), "Reponse [1].", monkeypatch
    )
    gen.answer("q", [passage()])
    assert faux.payloads[0]["temperature"] == 0.7


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


# ---------------------------------------------------------------------------
# Ordre juridique : un texte europeen n'est pas le droit mauritanien
# ---------------------------------------------------------------------------


def consigne_a_plat() -> str:
    """La consigne, sans les retours a la ligne de sa mise en forme."""
    return " ".join(SYSTEM_PROMPT.split()).lower()


def test_consigne_impose_de_situer_l_ordre_juridique():
    """Constate en conditions reelles : interroge sur les obligations de
    declaration des captures, le modele a repondu uniquement en droit
    europeen, sans signaler que le reglement 1224/2009 ne s'applique pas a
    un navire sous pavillon mauritanien."""
    consigne = consigne_a_plat()
    assert "mauritanien" in consigne
    assert "europ" in consigne
    assert "ne s'applique pas de plein droit" in consigne


def test_consigne_demande_de_signaler_l_absence_de_droit_national():
    assert "n'a pas ete retrouve" in consigne_a_plat()


def test_plafond_de_sortie_suffisant(settings):
    """Une reponse juridique detaillee depasse 1024 tokens : la precedente
    valeur tronquait la reponse au milieu d'une phrase."""
    assert settings.generation_max_tokens >= 2048


# ---------------------------------------------------------------------------
# Choix du fournisseur
# ---------------------------------------------------------------------------


def avec_claude(settings, **kwargs):
    return replace(settings, anthropic_api_key="cle-anthropic-de-test", **kwargs)


def test_seul_groq_configure(settings):
    """Sans cle Anthropic, rien ne doit proposer Claude."""
    assert list(modeles_disponibles(settings)) == ["groq"]


def test_les_deux_configures(settings):
    modeles = modeles_disponibles(avec_claude(settings))
    assert set(modeles) == {"groq", "claude"}
    assert modeles["claude"].modele_api == "claude-sonnet-5"
    assert modeles["claude"].fournisseur == "anthropic"


def test_aucune_cle_ne_propose_rien(settings):
    assert modeles_disponibles(replace(settings, groq_api_key=None)) == {}


def test_un_modele_sans_cle_n_est_pas_propose(settings):
    """Le proposer ne menerait qu'a une erreur au premier appel."""
    claude_seul = replace(
        settings, groq_api_key=None, anthropic_api_key="cle-anthropic-de-test"
    )
    gen = Generator(claude_seul)
    assert [m.cle for m in gen.modeles()] == ["claude"]
    assert gen.defaut == "claude"


def test_defaut_configure_sans_cle_bascule(settings):
    """GENERATION_DEFAULT=claude sans ANTHROPIC_API_KEY ne doit pas echouer."""
    gen = Generator(replace(settings, generation_default="claude"))
    assert gen.defaut == "groq"


def test_le_defaut_est_en_tete(settings):
    gen = Generator(avec_claude(settings, generation_default="claude"))
    assert [m.cle for m in gen.modeles()][0] == "claude"


def test_modele_inconnu_refuse(settings):
    gen = Generator(settings)
    with pytest.raises(GenerationError, match="claude"):
        gen.answer("q", [passage()], modele="claude")


# ---------------------------------------------------------------------------
# Appel a Claude, sans reseau
# ---------------------------------------------------------------------------


class BlocTexte:
    type = "text"

    def __init__(self, text):
        self.text = text


class BlocRaisonnement:
    """Avec le raisonnement adaptatif, `content` contient aussi ces blocs."""

    type = "thinking"

    def __init__(self, thinking=""):
        self.thinking = thinking


class Usage:
    def __init__(self, entree, sortie):
        self.input_tokens = entree
        self.output_tokens = sortie


class MessageClaude:
    def __init__(self, blocs, stop_reason="end_turn"):
        self.content = blocs
        self.model = "claude-sonnet-5"
        self.usage = Usage(1200, 80)
        self.stop_reason = stop_reason


class FauxAnthropic:
    """Repond comme le SDK Anthropic, et memorise les arguments recus."""

    def __init__(self, blocs=None, stop_reason="end_turn"):
        self._blocs = blocs if blocs is not None else [BlocTexte("Non [1].")]
        self._stop_reason = stop_reason
        self.appels = []
        self.messages = self

    def create(self, **kwargs):
        self.appels.append(kwargs)
        return MessageClaude(self._blocs, self._stop_reason)


def generateur_claude(settings, monkeypatch, temperature=None, **kwargs):
    reglages = avec_claude(settings)
    if temperature is not None:
        reglages = replace(reglages, generation_temperature=temperature)
    gen = Generator(reglages)
    faux = FauxAnthropic(**kwargs)
    monkeypatch.setattr(gen, "_client_anthropic", faux)
    return gen, faux


def test_claude_repond(settings, monkeypatch):
    gen, _ = generateur_claude(settings, monkeypatch)
    r = gen.answer("Peut-on rejuger ?", [passage("326")], modele="claude")

    assert r.text == "Non [1]."
    assert r.cited == (1,)
    assert r.grounded
    assert r.model == "claude-sonnet-5"
    assert r.fournisseur == "anthropic"
    assert r.prompt_tokens == 1200
    assert r.completion_tokens == 80


def test_claude_ne_recoit_aucune_temperature(settings, monkeypatch):
    """Sonnet 5 rejette en 400 tout parametre d'echantillonnage dont la
    valeur n'est pas celle par defaut : 0,3 echouerait autant que 0."""
    gen, faux = generateur_claude(settings, monkeypatch, temperature=0.3)
    gen.answer("q", [passage()], modele="claude")

    envoye = faux.appels[0]
    assert "temperature" not in envoye
    assert "top_p" not in envoye
    assert "top_k" not in envoye


def test_claude_laisse_le_raisonnement_adaptatif(settings, monkeypatch):
    """Omettre `thinking` vaut adaptatif sur Sonnet 5, ce qui est voulu."""
    gen, faux = generateur_claude(settings, monkeypatch)
    gen.answer("q", [passage()], modele="claude")

    envoye = faux.appels[0]
    assert "thinking" not in envoye
    assert envoye["output_config"] == {"effort": "low"}


def test_claude_recoit_la_consigne_et_les_extraits(settings, monkeypatch):
    gen, faux = generateur_claude(settings, monkeypatch)
    gen.answer("Ma question ?", [passage("326")], modele="claude")

    envoye = faux.appels[0]
    # La consigne systeme passe par le parametre dedie, pas par `messages`.
    assert envoye["system"] == SYSTEM_PROMPT
    assert [m["role"] for m in envoye["messages"]] == ["user"]
    assert "Article 326-" in envoye["messages"][0]["content"]
    assert "Ma question ?" in envoye["messages"][0]["content"]


def test_claude_ignore_les_blocs_de_raisonnement(settings, monkeypatch):
    """Seul le texte compte : un bloc `thinking` n'est pas une reponse."""
    gen, _ = generateur_claude(
        settings, monkeypatch, blocs=[BlocRaisonnement("je reflechis"), BlocTexte("Non [1].")]
    )
    r = gen.answer("q", [passage()], modele="claude")
    assert r.text == "Non [1]."


def test_claude_plafond_atteint_signale(settings, monkeypatch, caplog):
    """Une reponse coupee au milieu d'un article doit se voir."""
    import logging

    gen, _ = generateur_claude(settings, monkeypatch, stop_reason="max_tokens")
    with caplog.at_level(logging.WARNING):
        r = gen.answer("q", [passage()], modele="claude")

    assert r.tronquee
    assert r.as_dict()["tronquee"] is True
    assert "coupee" in caplog.text


def test_claude_refus_remonte(settings, monkeypatch):
    gen, _ = generateur_claude(settings, monkeypatch, stop_reason="refusal")
    with pytest.raises(GenerationError, match="refus"):
        gen.answer("q", [passage()], modele="claude")


def test_groq_reste_joignable_quand_claude_est_configure(settings, monkeypatch):
    """Configurer Claude ne doit rien changer au chemin Groq."""
    gen = Generator(avec_claude(settings))
    faux = FauxGroq("Oui [1].")
    monkeypatch.setattr(gen, "_post", faux)

    r = gen.answer("q", [passage()], modele="groq")
    assert r.fournisseur == "groq"
    assert faux.payloads[0]["temperature"] == 0.1


def test_le_plafond_de_claude_couvre_le_raisonnement(settings):
    """`max_tokens` borne raisonnement + reponse : il doit depasser le
    plafond prevu pour une reponse seule."""
    assert settings.anthropic_max_tokens > settings.generation_max_tokens
