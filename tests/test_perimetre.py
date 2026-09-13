"""Tests de la restriction du corpus interrogeable.

Le perimetre est le seul mecanisme du projet qui *retire* de la matiere a
la recherche. Une erreur y est donc silencieuse par nature : on ne voit
pas ce qui n'est pas remonte. D'ou l'insistance sur les cas degrades —
fichier absent, liste vide, JSON casse — qui doivent tous retomber sur le
corpus entier plutot que sur un corpus vide.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace

import pytest

from legalrag.perimetre import Perimetre, PerimetreError


def ecrire(chemin, contenu) -> None:
    chemin.write_text(json.dumps(contenu, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------


def test_fichier_absent_laisse_le_corpus_entier(tmp_path):
    """Cas normal d'une installation qui interroge tout."""
    p = Perimetre.charger(tmp_path / "rien.json")

    assert not p
    assert p.autorise("n-importe-quel-document")


def test_chemin_nul_laisse_le_corpus_entier():
    assert not Perimetre.charger(None)


def test_perimetre_inactif_laisse_le_corpus_entier(tmp_path):
    """Le fichier reste en place, la restriction est simplement levee."""
    f = tmp_path / "perimetre.json"
    ecrire(f, {"actif": False, "documents": ["a", "b"]})

    p = Perimetre.charger(f)
    assert not p
    assert p.autorise("z")


def test_documents_sous_forme_de_liste(tmp_path):
    f = tmp_path / "perimetre.json"
    ecrire(f, {"actif": True, "documents": ["a", "b"]})

    p = Perimetre.charger(f)
    assert p.documents == frozenset({"a", "b"})


def test_documents_sous_forme_d_objet_avec_justification(tmp_path):
    """L'ecriture retenue dans le depot : chaque texte porte sa raison."""
    f = tmp_path / "perimetre.json"
    ecrire(f, {"actif": True, "documents": {"a": "parce que", "b": "aussi"}})

    p = Perimetre.charger(f)
    assert p.documents == frozenset({"a", "b"})


def test_libelle_conserve(tmp_path):
    f = tmp_path / "perimetre.json"
    ecrire(f, {"actif": True, "libelle": "Peche", "documents": ["a"]})

    assert Perimetre.charger(f).libelle == "Peche"


# ---------------------------------------------------------------------------
# Cas degrades : jamais un corpus vide
# ---------------------------------------------------------------------------


def test_perimetre_actif_mais_vide_est_ignore(tmp_path, caplog):
    """Un perimetre actif sans document ne ramenerait plus jamais rien.

    C'est presque surement une faute de frappe, pas une intention : on
    revient au corpus entier en le signalant.
    """
    f = tmp_path / "perimetre.json"
    ecrire(f, {"actif": True, "documents": []})

    with caplog.at_level(logging.WARNING):
        p = Perimetre.charger(f)

    assert not p
    assert p.autorise("quoi que ce soit")
    assert "aucun document" in caplog.text


def test_json_casse_refuse(tmp_path):
    f = tmp_path / "perimetre.json"
    f.write_text("{ceci n'est pas du JSON", encoding="utf-8")

    with pytest.raises(PerimetreError, match="illisible"):
        Perimetre.charger(f)


def test_racine_qui_n_est_pas_un_objet_refusee(tmp_path):
    f = tmp_path / "perimetre.json"
    ecrire(f, ["a", "b"])

    with pytest.raises(PerimetreError, match="mal forme"):
        Perimetre.charger(f)


def test_documents_du_mauvais_type_refuses(tmp_path):
    f = tmp_path / "perimetre.json"
    ecrire(f, {"actif": True, "documents": "un-seul-document"})

    with pytest.raises(PerimetreError, match="liste ou un objet"):
        Perimetre.charger(f)


# ---------------------------------------------------------------------------
# Autorisation
# ---------------------------------------------------------------------------


def test_autorise_ce_qui_est_liste():
    p = Perimetre(actif=True, documents=frozenset({"a"}))
    assert p.autorise("a")
    assert not p.autorise("b")


def test_inactif_autorise_tout():
    p = Perimetre(actif=False, documents=frozenset({"a"}))
    assert p.autorise("b")


# ---------------------------------------------------------------------------
# Effet sur la recherche
# ---------------------------------------------------------------------------


def _retriever(settings, monkeypatch, points, perimetre_json=None, tmp_path=None):
    from legalrag import retrieval
    from tests.test_retrieval import FauxQdrant, FauxTransport

    if perimetre_json is not None:
        f = tmp_path / "perimetre.json"
        ecrire(f, perimetre_json)
        settings = replace(settings, perimetre_file=f)

    faux = FauxQdrant(points)
    monkeypatch.setattr(retrieval, "_make_client", lambda s: faux)
    monkeypatch.setattr(retrieval, "VoyageTransport", lambda **kw: FauxTransport())
    return retrieval.Retriever(settings), faux


def _points():
    from tests.test_retrieval import FauxPoint, charge

    return [
        FauxPoint(charge("c1", "Article 84- Peche sans licence.", doc="Recueil 28"), 0.9),
        FauxPoint(charge("c2", "Article 12- Sans rapport.", doc="Recueil 03"), 0.5),
    ]


def test_le_filtre_restreint_aux_documents_du_perimetre(settings, monkeypatch, tmp_path):
    r, faux = _retriever(
        settings, monkeypatch, _points(),
        {"actif": True, "documents": ["recueil-28", "constitution"]}, tmp_path,
    )
    r.search("une question", rerank=False)

    condition = faux.filtres[-1].must[0]
    assert condition.key == "document_id"
    assert sorted(condition.match.any) == ["constitution", "recueil-28"]


def test_sans_perimetre_aucun_filtre(settings, monkeypatch):
    r, faux = _retriever(settings, monkeypatch, _points())
    r.search("une question", rerank=False)

    assert faux.filtres[-1] is None


def test_un_document_du_perimetre_reste_selectionnable(settings, monkeypatch, tmp_path):
    r, faux = _retriever(
        settings, monkeypatch, _points(),
        {"actif": True, "documents": ["recueil-28"]}, tmp_path,
    )
    r.search("une question", document_id="recueil-28", rerank=False)

    condition = faux.filtres[-1].must[0]
    assert condition.match.value == "recueil-28"


def test_un_document_hors_perimetre_est_refuse(settings, monkeypatch, tmp_path):
    """Le nommer explicitement ne doit pas contourner la restriction.

    Sans ce refus, l'appelant recevrait zero passage sans savoir pourquoi :
    les deux conditions se contrediraient en silence.
    """
    r, faux = _retriever(
        settings, monkeypatch, _points(),
        {"actif": True, "documents": ["recueil-28"]}, tmp_path,
    )
    with pytest.raises(ValueError, match="hors du perimetre"):
        r.search("une question", document_id="recueil-03", rerank=False)

    # Rien n'a ete demande a Qdrant : l'erreur precede l'appel.
    assert faux.filtres == []


def test_le_catalogue_suit_le_perimetre(settings, monkeypatch, tmp_path):
    """Proposer dans l'interface un document que la recherche refusera
    d'ouvrir ne menerait qu'a une erreur."""
    r, _ = _retriever(
        settings, monkeypatch, _points(),
        {"actif": True, "documents": ["recueil-28"]}, tmp_path,
    )
    assert [d["document_id"] for d in r.documents()] == ["recueil-28"]


def test_le_catalogue_entier_sans_perimetre(settings, monkeypatch):
    r, _ = _retriever(settings, monkeypatch, _points())
    assert sorted(d["document_id"] for d in r.documents()) == ["recueil-03", "recueil-28"]
