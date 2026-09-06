"""Tests du chunking, en particulier le recollement entre pages."""

from __future__ import annotations

import pytest

from legalrag.chunking import (
    PageText,
    chunk_pages,
    consecutive_runs,
    dedupe_pages,
    make_chunk_id,
)
from legalrag.filename_parser import parse_filename
from legalrag.tokenization import HeuristicTokenizer


TOKENIZER = HeuristicTokenizer()


def page(filename: str, text: str) -> PageText:
    return PageText(parsed=parse_filename(filename), text=text)


def test_article_coupe_entre_deux_pages_est_reconstitue():
    """Le coeur du besoin : un article coupe en fin de page 20.

    La page 20 se termine par le debut de l'article, la page 21 le termine.
    Au moins un chunk doit contenir la phrase entiere.
    """
    fin_page_20 = "Blabla. " * 60 + "Article 706-73. La procedure applicable"
    debut_page_21 = "a la criminalite organisee est definie par le present titre."

    chunks = chunk_pages(
        [page("Code-20.txt", fin_page_20), page("Code-21.txt", debut_page_21)],
        TOKENIZER,
        chunk_tokens=80,
        overlap_tokens=30,
    )

    # La page 20 se termine sans ponctuation finale : la phrase court sur la
    # page suivante, donc les deux moities doivent etre recollees par une
    # simple espace, sans coupure de paragraphe au milieu de l'article.
    phrase = "La procedure applicable a la criminalite organisee"
    assert any(phrase in c.text for c in chunks), "l'article n'a ete reconstitue dans aucun chunk"

    # Et le chunk qui la contient est correctement attribue aux deux pages.
    porteur = next(c for c in chunks if phrase in c.text)
    assert porteur.page_start == 20
    assert porteur.page_end == 21
    assert porteur.pages == (20, 21)


def test_chevauchement_effectif_entre_chunks_consecutifs():
    texte = " ".join(f"mot{i}" for i in range(400))
    chunks = chunk_pages(
        [page("Code-1.txt", texte)], TOKENIZER, chunk_tokens=100, overlap_tokens=40
    )

    assert len(chunks) > 1
    for precedent, suivant in zip(chunks, chunks[1:]):
        fin = set(precedent.text.split()[-20:])
        debut = set(suivant.text.split()[:40])
        assert fin & debut, "aucun recouvrement entre deux chunks consecutifs"


def test_pages_non_contigues_ne_sont_pas_recollees():
    """Une page 21 absente ne doit pas creer un faux voisinage 20-22."""
    chunks = chunk_pages(
        [
            page("Code-20.txt", "Contenu de la page vingt."),
            page("Code-22.txt", "Contenu de la page vingt-deux."),
        ],
        TOKENIZER,
        chunk_tokens=200,
        overlap_tokens=50,
    )
    for chunk in chunks:
        assert chunk.pages in {(20,), (22,)}


def test_metadonnees_portees_par_chaque_chunk():
    chunks = chunk_pages(
        [page("202311230853-Code de Procedure Penale-21.txt", "Un texte de test. " * 30)],
        TOKENIZER,
        chunk_tokens=50,
        overlap_tokens=10,
    )
    assert chunks
    for chunk in chunks:
        assert chunk.document == "Code de Procedure Penale"
        assert chunk.document_id == "code-de-procedure-penale"
        assert chunk.timestamp == "202311230853"
        assert chunk.source_filenames == ("202311230853-Code de Procedure Penale-21.txt",)
        assert chunk.token_count > 0


def test_horodatage_absent_donne_une_chaine_vide():
    chunks = chunk_pages(
        [page("Code Civil-1.txt", "Texte simple.")], TOKENIZER, 50, 10
    )
    assert chunks[0].timestamp == ""


def test_chunk_id_deterministe_et_sensible_au_contenu():
    assert make_chunk_id("code", 0, "texte") == make_chunk_id("code", 0, "texte")
    assert make_chunk_id("code", 0, "texte") != make_chunk_id("code", 1, "texte")
    assert make_chunk_id("code", 0, "texte") != make_chunk_id("code", 0, "texte modifie")


def test_deux_passes_donnent_les_memes_identifiants():
    """Condition de la reprise : rejouer le job ne doit rien replonger."""
    pages = [page("Code-1.txt", "Article premier. " * 40), page("Code-2.txt", "Article deux. " * 40)]
    premiers = chunk_pages(pages, TOKENIZER, 60, 20)
    seconds = chunk_pages(pages, TOKENIZER, 60, 20)
    assert [c.chunk_id for c in premiers] == [c.chunk_id for c in seconds]


def test_dedupe_garde_la_version_la_plus_recente():
    pages = [
        page("202301010000-Code-5.txt", "ancienne"),
        page("202406010000-Code-5.txt", "recente"),
    ]
    retenues = dedupe_pages(pages)
    assert len(retenues) == 1
    assert retenues[0].text == "recente"


def test_consecutive_runs():
    pages = [page(f"Code-{n}.txt", "x") for n in (1, 2, 3, 7, 8, 20)]
    runs = consecutive_runs(pages)
    assert [[p.page for p in run] for run in runs] == [[1, 2, 3], [7, 8], [20]]


def test_page_vide_ne_produit_pas_de_chunk():
    assert chunk_pages([page("Code-1.txt", "   \n  ")], TOKENIZER, 50, 10) == []


@pytest.mark.parametrize("overlap", [0, 1, 49])
def test_pas_de_boucle_infinie_selon_le_chevauchement(overlap):
    texte = "Article premier du code. " * 50
    chunks = chunk_pages([page("Code-1.txt", texte)], TOKENIZER, 50, overlap)
    assert chunks
    assert len(chunks) < 500  # progression garantie a chaque tour


def test_separateur_de_page_selon_la_ponctuation():
    from legalrag.chunking import page_separator

    # Phrase terminee : nouvelle page = nouveau paragraphe.
    assert page_separator("Fin de l'article.") == "\n\n"
    assert page_separator("Voir ci-dessous :") == "\n\n"
    # Phrase en cours : on recolle sans casser l'article.
    assert page_separator("La procedure applicable") == " "
    assert page_separator("") == "\n\n"
