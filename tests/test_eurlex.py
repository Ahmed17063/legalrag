"""Tests du collecteur EUR-Lex.

Aucun appel reseau : le transport est remplace par un double. Ce qui est
verifie ici, c'est le respect du delai impose par le robots.txt, la forme
des URL, et le refus d'ecrire un fichier qui n'est pas un PDF.
"""

from __future__ import annotations

import requests
import pytest

from legalrag.scraping.eurlex import (
    CRAWL_DELAY,
    PECHE_MARITIME,
    EurlexCollector,
    TexteEuropeen,
)

PDF_MINIMAL = b"%PDF-1.4\n% contenu de test\n"


@pytest.fixture
def collecteur(monkeypatch):
    """Un collecteur qui ne dort pas, pour que les tests restent rapides."""
    c = EurlexCollector()
    monkeypatch.setattr(c, "_patienter", lambda: None)
    return c


def texte(celex="32008R1005", fichier="UE-test") -> TexteEuropeen:
    return TexteEuropeen(celex, "Un reglement de test", fichier)


# ---------------------------------------------------------------------------
# URL
# ---------------------------------------------------------------------------


def test_url_par_celex(collecteur):
    url = collecteur.url_pdf("32008R1005")
    assert url.endswith("/FR/TXT/PDF/?uri=CELEX:32008R1005")


def test_format_pdf_et_non_doc(collecteur):
    """Le robots.txt d'EUR-Lex interdit le format DOC, pas le PDF."""
    url = collecteur.url_pdf("32009R1224")
    assert "/TXT/PDF/" in url
    assert "/TXT/DOC/" not in url


def test_langue_configurable():
    assert "/EN/TXT/PDF/" in EurlexCollector(langue="en").url_pdf("32008R1005")


# ---------------------------------------------------------------------------
# Politesse
# ---------------------------------------------------------------------------


def test_delai_par_defaut_conforme_au_robots_txt():
    """EUR-Lex impose Crawl-delay: 10. Le respecter n'est pas optionnel."""
    assert CRAWL_DELAY >= 10.0
    assert EurlexCollector()._delai >= 10.0


def test_agent_identifiable():
    entete = EurlexCollector()._session.headers["User-Agent"]
    assert "legalrag" in entete


def test_le_delai_est_bien_attendu(monkeypatch):
    """Sans temporisation, la collecte violerait le robots.txt."""
    c = EurlexCollector(delai=0.05)
    attentes = []
    monkeypatch.setattr("legalrag.scraping.eurlex.time.sleep", attentes.append)

    c._patienter()
    c._patienter()

    assert attentes, "aucune attente entre deux requetes"


# ---------------------------------------------------------------------------
# Telechargement
# ---------------------------------------------------------------------------


def test_ecriture_du_pdf(collecteur, monkeypatch, tmp_path):
    monkeypatch.setattr(collecteur, "_telecharger", lambda url: PDF_MINIMAL)

    chemin = collecteur.recuperer(texte(fichier="UE-reglement-test"), tmp_path)

    assert chemin is not None
    assert chemin.name == "UE-reglement-test.pdf"
    assert chemin.read_bytes() == PDF_MINIMAL


def test_reponse_html_refusee(collecteur, monkeypatch, tmp_path):
    """EUR-Lex renvoie parfois une page d'erreur avec un code 200."""
    monkeypatch.setattr(
        collecteur, "_telecharger", lambda url: b"<html><body>Not available</body></html>"
    )

    assert collecteur.recuperer(texte(), tmp_path) is None
    assert not list(tmp_path.glob("*.pdf"))


def test_fichier_deja_present_non_retelecharge(collecteur, monkeypatch, tmp_path):
    appels = []
    monkeypatch.setattr(collecteur, "_telecharger", lambda url: appels.append(url) or PDF_MINIMAL)

    t = texte(fichier="UE-deja-la")
    collecteur.recuperer(t, tmp_path)
    collecteur.recuperer(t, tmp_path)

    assert len(appels) == 1


def test_aucun_fichier_temporaire_laisse(collecteur, monkeypatch, tmp_path):
    monkeypatch.setattr(collecteur, "_telecharger", lambda url: PDF_MINIMAL)
    collecteur.recuperer(texte(), tmp_path)
    assert not list(tmp_path.glob("*.part"))


def test_echec_reseau_n_arrete_pas_la_collecte(collecteur, monkeypatch, tmp_path):
    """Un texte indisponible ne doit pas condamner les autres."""

    def telecharger(url):
        if "32009R1224" in url:
            raise requests.RequestException("indisponible")
        return PDF_MINIMAL

    monkeypatch.setattr(collecteur, "_telecharger", telecharger)

    recuperes, echoues = collecteur.collecter(PECHE_MARITIME, tmp_path)

    assert len(echoues) == 1
    assert echoues[0].celex == "32009R1224"
    assert len(recuperes) == len(PECHE_MARITIME) - 1


# ---------------------------------------------------------------------------
# La liste des textes
# ---------------------------------------------------------------------------


def test_celex_bien_formes():
    """3 = droit derive, l'annee, R = reglement, puis le numero."""
    import re

    for t in PECHE_MARITIME:
        assert re.fullmatch(r"3\d{4}[RLD]\d{4}", t.celex), t.celex


def test_noms_de_fichiers_distincts_et_parlants():
    """Le nom de fichier devient le titre du document apres ingestion."""
    noms = [t.fichier for t in PECHE_MARITIME]
    assert len(set(noms)) == len(noms)
    for nom in noms:
        assert nom.startswith("UE-")
        assert len(nom) > 12


def test_noms_sans_ambiguite_de_page():
    """Un nom finissant par un tiret et des chiffres brouillerait le parseur."""
    from legalrag.filename_parser import parse_filename

    for t in PECHE_MARITIME:
        nom_document = t.fichier.replace("-", " ")
        assert parse_filename(f"{nom_document}-3.txt").page == 3
