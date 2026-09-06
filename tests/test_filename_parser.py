"""Tests du parsing des noms de fichiers du corpus."""

from __future__ import annotations

import logging
from datetime import datetime

import pytest

from legalrag.filename_parser import (
    FilenameConventionError,
    parse_filename,
    parse_filename_or_none,
    slugify,
)


# --------------------------------------------------------------------------
# Cas nominaux
# --------------------------------------------------------------------------

def test_avec_horodatage():
    r = parse_filename("202311230853-Code de Procedure Penale-21.txt")
    assert r.document == "Code de Procedure Penale"
    assert r.page == 21
    assert r.timestamp == "202311230853"
    assert r.timestamp_dt == datetime(2023, 11, 23, 8, 53)


def test_sans_horodatage():
    r = parse_filename("Code de Procedure Penale-21.txt")
    assert r.document == "Code de Procedure Penale"
    assert r.page == 21
    # L'absence d'horodatage est normale, pas une erreur.
    assert r.timestamp is None
    assert r.timestamp_dt is None


def test_horodatage_avec_secondes():
    r = parse_filename("20231123085307-Code Civil-7.txt")
    assert r.timestamp == "20231123085307"
    assert r.timestamp_dt == datetime(2023, 11, 23, 8, 53, 7)
    assert r.document == "Code Civil"


# --------------------------------------------------------------------------
# Noms de documents contenant des tirets
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("filename", "document", "page"),
    [
        ("Code de l'entree et du sejour - CESEDA-12.txt", "Code de l'entree et du sejour - CESEDA", 12),
        ("Loi-Informatique-et-Libertes-3.txt", "Loi-Informatique-et-Libertes", 3),
        ("202401010000-Decret n-2024-115-9.txt", "Decret n-2024-115", 9),
    ],
)
def test_nom_a_tirets(filename, document, page):
    """Le titre peut contenir des tirets : l'ancrage se fait sur la page finale."""
    r = parse_filename(filename)
    assert r.document == document
    assert r.page == page


def test_ancrage_sur_le_dernier_bloc_numerique():
    """"Foo-12-34" : 34 est la page, "Foo-12" reste le titre."""
    r = parse_filename("Article 700-12-34.txt")
    assert r.document == "Article 700-12"
    assert r.page == 34


# --------------------------------------------------------------------------
# Numeros de page
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("filename", "page"),
    [
        ("Code Civil-1.txt", 1),
        ("Code Civil-42.txt", 42),
        ("Code Civil-1234.txt", 1234),
        ("Code Civil-10475.txt", 10475),
        ("Code Civil-007.txt", 7),  # zeros de tete normalises
    ],
)
def test_page_multi_chiffres(filename, page):
    assert parse_filename(filename).page == page


# --------------------------------------------------------------------------
# Le prefixe numerique n'est PAS toujours un horodatage
# --------------------------------------------------------------------------

def test_bloc_numerique_court_conserve_dans_le_titre():
    """"2023" n'a pas la longueur d'un horodatage : c'est du titre."""
    r = parse_filename("2023-Loi de finances-4.txt")
    assert r.document == "2023-Loi de finances"
    assert r.timestamp is None


def test_bloc_numerique_de_bonne_longueur_mais_date_invalide():
    """12 chiffres mais mois 99 : ce n'est pas une date, donc c'est du titre."""
    r = parse_filename("209913990853-Code-1.txt")
    assert r.document == "209913990853-Code"
    assert r.timestamp is None


def test_horodatage_sans_titre_restant():
    """Retirer l'horodatage laisserait un document sans nom : on le garde."""
    r = parse_filename("202311230853-21.txt")
    assert r.document == "202311230853"
    assert r.page == 21
    assert r.timestamp is None


# --------------------------------------------------------------------------
# Fichiers hors convention
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "filename",
    [
        "Code de Procedure Penale.txt",      # pas de numero de page
        "Code de Procedure Penale-.txt",     # page vide
        "Code de Procedure Penale-xii.txt",  # page non numerique
        "-21.txt",                           # pas de titre
        "21.txt",                            # ni titre ni separateur
        "notes.pdf",                         # extension non supportee
        "Code de Procedure Penale-21.txt.bak",
        ".txt",                              # nom vide
        "",                                  # chaine vide
        "   ",                               # blancs uniquement
    ],
)
def test_hors_convention_leve(filename):
    with pytest.raises(FilenameConventionError):
        parse_filename(filename)


def test_hors_convention_ne_casse_pas_le_job(caplog):
    """Le pipeline doit pouvoir sauter le fichier et continuer."""
    with caplog.at_level(logging.WARNING):
        assert parse_filename_or_none("notes.pdf") is None
    assert "hors convention" in caplog.text


def test_variante_non_levante_sur_cas_valide():
    assert parse_filename_or_none("Code Civil-3.txt").page == 3


# --------------------------------------------------------------------------
# Details de robustesse
# --------------------------------------------------------------------------

def test_extension_insensible_a_la_casse():
    assert parse_filename("Code Civil-3.TXT").page == 3


def test_espaces_de_bord_ignores():
    r = parse_filename("  Code Civil-3.txt  ")
    assert r.document == "Code Civil"
    assert r.source_filename == "Code Civil-3.txt"


def test_chemin_complet_accepte():
    r = parse_filename("corpus/2024/Code Civil-3.txt")
    assert r.document == "Code Civil"
    assert r.source_filename == "Code Civil-3.txt"


def test_metadata_horodatage_absent_est_une_chaine_vide():
    meta = parse_filename("Code Civil-3.txt").as_metadata()
    assert meta["timestamp"] == ""
    assert meta["document"] == "Code Civil"
    assert meta["page"] == 3


def test_document_id_est_un_slug_stable():
    a = parse_filename("202311230853-Code de Procedure Penale-21.txt").document_id
    b = parse_filename("Code de Procedure Penale-99.txt").document_id
    # Deux pages du meme document, avec et sans horodatage, partagent l'identifiant.
    assert a == b == "code-de-procedure-penale"


def test_slugify_accents_et_apostrophes():
    assert slugify("Code de l'Entrée & du Séjour") == "code-de-l-entree-du-sejour"
