"""Tests de la source de corpus locale."""

from __future__ import annotations

import pytest

from legalrag.local_corpus import LocalCorpus


@pytest.fixture
def corpus(tmp_path):
    racine = tmp_path / "corpus"
    (racine / "fr").mkdir(parents=True)
    (racine / "ar").mkdir(parents=True)
    (racine / "fr" / "202506251725-Code des mines-1.txt").write_text(
        "Article premier du code minier.", encoding="utf-8"
    )
    (racine / "fr" / "202506251725-Code des mines-2.txt").write_text(
        "Article deux du code minier.", encoding="utf-8"
    )
    (racine / "ar" / "202703301117-Statut-1.txt").write_text("المادة الأولى", encoding="utf-8")
    (racine / "fr" / "notes.md").write_text("pas un txt", encoding="utf-8")
    return racine


def test_liste_les_txt(corpus):
    fichiers = LocalCorpus(corpus).list_text_files()
    assert len(fichiers) == 3
    assert all(f.name.endswith(".txt") for f in fichiers)


def test_ignore_les_autres_extensions(corpus):
    noms = {f.name for f in LocalCorpus(corpus).list_text_files()}
    assert "notes.md" not in noms


def test_lecture_du_contenu(corpus):
    source = LocalCorpus(corpus)
    fichier = next(f for f in source.list_text_files() if f.name.endswith("-1.txt") and "Mines" in f.name or "mines" in f.name)
    assert "code minier" in source.read_text(fichier)


def test_accents_et_arabe_relus_correctement(corpus):
    source = LocalCorpus(corpus)
    arabe = next(f for f in source.list_text_files() if "Statut" in f.name)
    assert source.read_text(arabe) == "المادة الأولى"


def test_identifiants_uniques_et_stables(corpus):
    source = LocalCorpus(corpus)
    premiers = {f.id for f in source.list_text_files()}
    seconds = {f.id for f in source.list_text_files()}
    assert len(premiers) == 3
    assert premiers == seconds


def test_sous_repertoire_seul(corpus):
    """Un index par langue : on pointe directement le sous-repertoire."""
    fichiers = LocalCorpus(corpus / "fr").list_text_files()
    assert len(fichiers) == 2


def test_sans_recursion(corpus):
    """A la racine, sans recursion, il n'y a aucun .txt."""
    assert LocalCorpus(corpus, recursive=False).list_text_files() == []


def test_repertoire_absent(tmp_path):
    with pytest.raises(FileNotFoundError, match="introuvable"):
        LocalCorpus(tmp_path / "rien")


def test_meme_surface_que_drive(corpus):
    """Le pipeline doit pouvoir substituer l'une a l'autre."""
    from legalrag.drive import DriveCorpus

    for methode in ("list_text_files", "read_text"):
        assert hasattr(LocalCorpus, methode)
        assert hasattr(DriveCorpus, methode)
