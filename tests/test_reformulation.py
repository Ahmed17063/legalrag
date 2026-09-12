"""Tests de la reformulation de requete, sans reseau.

Ce maillon est facultatif par construction : il ameliore le rappel quand
il fonctionne, et ne doit rien casser quand il echoue. C'est surtout cette
seconde propriete qui est verifiee ici — une reformulation ratee qui
ferait echouer la recherche serait pire que pas de reformulation du tout.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import pytest

from legalrag.generation import GenerationError
from legalrag.reformulation import CONSIGNE, Reformulateur, _nettoyer, construire

QUESTION = "Est-ce qu'on peut licencier une femme enceinte ?"
REECRITE = (
    "licenciement d'une femme enceinte, rupture du contrat de travail pendant "
    "la grossesse et le conge de maternite, protection de la salariee"
)


class FauxGenerator:
    """Ne retient que ce que la reformulation lui demande."""

    def __init__(self, reponse=REECRITE, erreur=None):
        self._reponse = reponse
        self._erreur = erreur
        self.appels: list[dict] = []

    def completer(self, systeme, invite, *, temperature, max_tokens, modele=None):
        self.appels.append(
            {
                "systeme": systeme,
                "invite": invite,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "modele": modele,
            }
        )
        if self._erreur:
            raise self._erreur
        return self._reponse


def reformulateur(settings, **kwargs):
    faux = FauxGenerator(**kwargs)
    return Reformulateur(settings, faux), faux


# ---------------------------------------------------------------------------
# Nettoyage de ce que renvoie le modele
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("brut", "attendu"),
    [
        ("licenciement femme enceinte", "licenciement femme enceinte"),
        ("  licenciement  ", "licenciement"),
        ('"licenciement femme enceinte"', "licenciement femme enceinte"),
        ("«licenciement»", "licenciement"),
        ("Requete reecrite : licenciement", "licenciement"),
        ("Reformulation: licenciement", "licenciement"),
        ("Réécriture : licenciement", "licenciement"),
        ("licenciement\n\nExplication : parce que...", "licenciement"),
        ("", ""),
        ("\n\n", ""),
    ],
)
def test_nettoyage(brut, attendu):
    assert _nettoyer(brut) == attendu


# ---------------------------------------------------------------------------
# Le cas nominal
# ---------------------------------------------------------------------------


def test_la_question_est_reecrite(settings):
    reform, _ = reformulateur(settings)
    assert reform(QUESTION) == REECRITE


def test_la_temperature_de_reformulation_est_plus_haute(settings):
    """On cherche ici de la variete lexicale, pas de la fidelite au texte."""
    reform, faux = reformulateur(settings)
    reform(QUESTION)

    assert faux.appels[0]["temperature"] == settings.reformulation_temperature
    assert faux.appels[0]["temperature"] > settings.generation_temperature


def test_la_consigne_interdit_de_repondre(settings):
    """Le risque du procede : un modele qui repond au lieu de reecrire."""
    reform, faux = reformulateur(settings)
    reform(QUESTION)

    assert faux.appels[0]["systeme"] == CONSIGNE
    a_plat = " ".join(CONSIGNE.split()).lower()
    assert "ne reponds pas a la question" in a_plat
    assert "n'invente aucun numero d'article" in a_plat
    assert QUESTION in faux.appels[0]["invite"]


def test_la_reformulation_est_journalisee(settings, caplog):
    """Une recherche doit rester explicable."""
    reform, _ = reformulateur(settings)
    with caplog.at_level(logging.INFO):
        reform(QUESTION)

    assert REECRITE in caplog.text


# ---------------------------------------------------------------------------
# Tout echec retombe sur la question d'origine
# ---------------------------------------------------------------------------


def test_modele_indisponible(settings, caplog):
    reform, _ = reformulateur(settings, erreur=GenerationError("quota atteint"))
    with caplog.at_level(logging.WARNING):
        assert reform(QUESTION) == QUESTION

    assert "question d'origine" in caplog.text.lower()


def test_reponse_vide(settings):
    reform, _ = reformulateur(settings, reponse="   ")
    assert reform(QUESTION) == QUESTION


def test_le_modele_a_commence_a_repondre(settings, caplog):
    """Cas le plus dangereux : la reponse partirait chercher a la place de
    la question, et orienterait la recherche vers ce que le modele croit
    savoir plutot que vers ce qui est demande."""
    bavard = (
        "Non, le code du travail mauritanien interdit le licenciement d'une "
        "femme enceinte pendant toute la duree de la grossesse ainsi que "
        "pendant les six semaines qui suivent l'accouchement, et l'employeur "
        "qui passerait outre s'exposerait aux sanctions prevues par l'article "
        "qui traite de cette question, ce qui signifie en pratique que la "
        "salariee peut demander sa reintegration ou des dommages et interets."
    )
    reform, _ = reformulateur(settings, reponse=bavard)
    with caplog.at_level(logging.WARNING):
        assert reform(QUESTION) == QUESTION

    assert "trop longue" in caplog.text


def test_reformulation_identique(settings):
    """Une question deja juridique doit ressortir telle quelle."""
    reform, _ = reformulateur(settings, reponse=QUESTION)
    assert reform(QUESTION) == QUESTION


def test_question_vide(settings):
    reform, faux = reformulateur(settings)
    assert reform("   ") == ""
    assert faux.appels == []


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_desactivee_par_configuration(settings):
    assert construire(replace(settings, reformulation_enabled=False)) is None


def test_absente_sans_modele(settings):
    """Sans cle d'API, la recherche doit fonctionner comme avant."""
    assert construire(replace(settings, groq_api_key=None)) is None


def test_construite_quand_tout_est_reuni(settings):
    assert isinstance(construire(settings), Reformulateur)


def test_le_generateur_est_partage(settings):
    """Eviter une seconde session HTTP et un second client Anthropic."""
    faux = FauxGenerator()
    reform = construire(settings, faux)

    reform(QUESTION)
    assert len(faux.appels) == 1


def test_sans_reglage_le_modele_par_defaut_est_utilise(settings):
    reform, faux = reformulateur(settings)
    reform(QUESTION)
    assert faux.appels[0]["modele"] is None


def test_le_modele_de_reformulation_est_transmis(settings):
    """Une tache mecanique : on peut la confier au modele le plus rapide."""
    reform, faux = reformulateur(replace(settings, reformulation_model="groq"))
    reform(QUESTION)
    assert faux.appels[0]["modele"] == "groq"
