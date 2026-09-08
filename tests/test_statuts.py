"""Tests du statut juridique des textes.

Un assistant qui presente l'ordonnance 88-144 comme le droit applicable
est pire qu'un assistant muet. Ces tests verrouillent le principe : on ne
suppose jamais qu'un texte est en vigueur.
"""

from __future__ import annotations

import json

import pytest

from legalrag.statuts import (
    FicheStatut,
    RegistreStatuts,
    Statut,
    mentionne_une_abrogation,
)


def ecrire_registre(chemin, documents: dict) -> None:
    chemin.write_text(
        json.dumps({"documents": documents}, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Le defaut est l'ignorance, jamais la validite
# ---------------------------------------------------------------------------


def test_document_absent_est_inconnu():
    """Sur du droit, l'absence d'information n'est pas une garantie."""
    assert RegistreStatuts().statut("nimporte-quoi") is Statut.INCONNU


def test_registre_absent_ne_leve_pas(tmp_path):
    registre = RegistreStatuts.charger(tmp_path / "rien.json")
    assert len(registre) == 0
    assert registre.statut("un-code") is Statut.INCONNU


def test_registre_illisible_ne_bloque_pas_mais_est_signale(tmp_path, caplog):
    import logging

    casse = tmp_path / "statuts.json"
    casse.write_text("{ ceci n'est pas du JSON", encoding="utf-8")

    with caplog.at_level(logging.ERROR):
        registre = RegistreStatuts.charger(casse)

    assert len(registre) == 0
    assert "illisible" in caplog.text


def test_statut_invalide_traite_comme_inconnu(tmp_path, caplog):
    import logging

    chemin = tmp_path / "statuts.json"
    ecrire_registre(chemin, {"un-code": {"statut": "peut-etre"}})

    with caplog.at_level(logging.WARNING):
        registre = RegistreStatuts.charger(chemin)

    assert registre.statut("un-code") is Statut.INCONNU
    assert "peut-etre" in caplog.text


# ---------------------------------------------------------------------------
# Lecture du registre
# ---------------------------------------------------------------------------


def test_lecture_des_trois_statuts(tmp_path):
    chemin = tmp_path / "statuts.json"
    ecrire_registre(
        chemin,
        {
            "code-en-vigueur": {"statut": "en_vigueur"},
            "ordonnance-88-144": {"statut": "abroge", "remplace_par": "loi 2015-017"},
            "texte-douteux": {"statut": "inconnu"},
        },
    )
    registre = RegistreStatuts.charger(chemin)

    assert registre.statut("code-en-vigueur") is Statut.EN_VIGUEUR
    assert registre.statut("ordonnance-88-144") is Statut.ABROGE
    assert registre.statut("texte-douteux") is Statut.INCONNU


def test_forme_abregee_acceptee(tmp_path):
    """Un statut peut s'ecrire directement, sans objet englobant."""
    chemin = tmp_path / "statuts.json"
    ecrire_registre(chemin, {"un-code": "en_vigueur"})
    assert RegistreStatuts.charger(chemin).statut("un-code") is Statut.EN_VIGUEUR


def test_liste_des_abroges(tmp_path):
    chemin = tmp_path / "statuts.json"
    ecrire_registre(
        chemin,
        {
            "a": {"statut": "abroge"},
            "b": {"statut": "en_vigueur"},
            "c": {"statut": "abroge"},
        },
    )
    assert RegistreStatuts.charger(chemin).abroges == ["a", "c"]


# ---------------------------------------------------------------------------
# Mentions accolees aux citations
# ---------------------------------------------------------------------------


def test_texte_en_vigueur_sans_mention():
    """Ne pas alourdir la citation quand tout va bien."""
    assert FicheStatut("x", Statut.EN_VIGUEUR).mention() == ""


def test_texte_abroge_signale():
    assert FicheStatut("x", Statut.ABROGE).mention() == " [ABROGÉ]"


def test_texte_abroge_avec_son_remplacant():
    fiche = FicheStatut("x", Statut.ABROGE, remplace_par="loi 2015-017")
    assert fiche.mention() == " [ABROGÉ, remplacé par loi 2015-017]"


def test_statut_inconnu_signale_comme_non_verifie():
    assert "non vérifié" in FicheStatut("x", Statut.INCONNU).mention()


@pytest.mark.parametrize(
    ("statut", "citable"),
    [(Statut.EN_VIGUEUR, True), (Statut.ABROGE, False), (Statut.INCONNU, False)],
)
def test_seul_un_texte_en_vigueur_est_citable_comme_droit_applicable(statut, citable):
    assert statut.citable_comme_droit_applicable is citable


# ---------------------------------------------------------------------------
# Abrogation mentionnee dans le corps d'un texte
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "texte",
    [
        "Sont abrogées toutes dispositions contraires a la presente loi.",
        "L'article 12 est abroge.",
        "Cette ordonnance n'est plus en vigueur depuis 2015.",
        "L'abrogation prend effet a compter de la publication.",
    ],
)
def test_abrogation_detectee_dans_le_texte(texte):
    """Un recueil en vigueur peut contenir des articles abroges."""
    assert mentionne_une_abrogation(texte)


@pytest.mark.parametrize(
    "texte",
    [
        "Article 326- Aucune personne acquittee legalement ne peut etre reprise.",
        "La presente loi entre en vigueur a compter de sa publication.",
    ],
)
def test_texte_ordinaire_non_signale(texte):
    assert not mentionne_une_abrogation(texte)


# ---------------------------------------------------------------------------
# Le statut arrive jusqu'a la citation
# ---------------------------------------------------------------------------


def test_citation_porte_la_mention_d_abrogation():
    from legalrag.retrieval import Passage

    passage = Passage(
        text="Article 1er de l'ordonnance.",
        document="Ordonnance 88-144",
        document_id="ordonnance-88-144",
        page_start=3,
        page_end=3,
        pages=(3,),
        timestamp="",
        chunk_id="c1",
        article="1",
        statut=Statut.ABROGE,
        mention_statut=" [ABROGÉ, remplacé par loi 2015-017]",
    )
    citation = passage.citation()

    assert "ABROGÉ" in citation
    assert "loi 2015-017" in citation
    assert passage.as_dict()["statut"] == "abroge"


def test_consigne_interdit_de_presenter_un_texte_abroge_comme_applicable():
    from legalrag.generation import SYSTEM_PROMPT

    consigne = " ".join(SYSTEM_PROMPT.split())
    assert "[ABROGE]" in consigne
    assert "JAMAIS" in consigne
    assert "statut non verifie" in consigne
