"""Tests du retrait des en-tetes et pieds de page repetes."""

from __future__ import annotations

from legalrag.boilerplate import clean_pages, detect, normalize, strip
from legalrag.chunking import PageText, chunk_pages
from legalrag.filename_parser import parse_filename
from legalrag.tokenization import HeuristicTokenizer

TOKENIZER = HeuristicTokenizer()

EN_TETE = "Republique islamique de Mauritanie\nHonneur - Fraternite - Justice\nMinistere de la justice"


def page(numero: int, corps: str, suffixe_ocr: str = "", pied: str = "Code de Procedure Penale") -> PageText:
    """Une page avec son bandeau, tel qu'il ressort de l'OCR."""
    texte = f"{EN_TETE} {suffixe_ocr}\n{corps}\n{pied}"
    return PageText(parsed=parse_filename(f"Code de Procedure Penale-{numero}.txt"), text=texte)


def test_normalisation_absorbe_la_ponctuation_et_la_casse():
    assert normalize("Honneur - Fraternite - Justice : LS") == "honneur fraternite justice ls"


def test_detection_sur_pages_repetees():
    textes = [f"{EN_TETE}\nContenu unique {i}\nCode de Procedure Penale" for i in range(10)]
    cles = detect(textes)
    assert cles, "le bandeau aurait du etre detecte"
    assert any("honneur" in c for c in cles)
    assert any("code de proc" in c for c in cles)


def test_detection_tolere_le_bruit_ocr_en_fin_de_ligne():
    """Le meme bandeau ressort avec des suffixes differents selon les pages."""
    suffixes = ["LS", "D", "ARS", "T", "RS", "SE", "", "HR.", "LR", "SR"]
    textes = [f"{EN_TETE} {s}\nContenu {i}\nCode" for i, s in enumerate(suffixes)]
    cles = detect(textes)
    assert any("honneur fraternite justice" in c for c in cles)


def test_ligne_rare_non_supprimee():
    """Une ligne presente sur une seule page est du contenu, pas un bandeau."""
    textes = [f"{EN_TETE}\nContenu {i}" for i in range(10)]
    textes[0] = f"{EN_TETE}\nMention exceptionnelle qui n'apparait qu'ici"
    cles = detect(textes)
    assert not any("mention exceptionnelle" in c for c in cles)


def test_article_jamais_supprime_meme_si_frequent():
    """Garde-fou : une ligne d'article ne doit jamais partir."""
    textes = ["Article 1. Texte identique sur toutes les pages." for _ in range(10)]
    cles = detect(textes)
    resultat = strip(textes[0], cles)
    assert "Article 1." in resultat


def test_corps_de_page_preserve():
    """Le bandeau n'est cherche qu'aux extremites, pas au milieu.

    La ligne du milieu mentionne le ministere comme le fait l'en-tete,
    mais elle est du contenu : elle doit survivre.
    """
    textes = [
        f"{EN_TETE}\nOuverture {i}\n"
        "Le ministere de la justice est cite ici dans un article\n"
        f"Suite {i}\nEncore {i}\nFin {i}\nCode de Procedure Penale"
        for i in range(10)
    ]
    cles = detect(textes)
    resultat = strip(textes[0], cles)
    assert "cite ici dans un article" in resultat


def test_strip_retire_effectivement_le_bandeau():
    textes = [f"{EN_TETE}\nArticle {i}. Le contenu.\nCode de Procedure Penale" for i in range(10)]
    cles = detect(textes)
    resultat = strip(textes[3], cles)

    assert "Republique islamique" not in resultat
    assert "Honneur" not in resultat
    assert "Article 3. Le contenu." in resultat


def test_clean_pages_par_document():
    """Chaque document a son propre bandeau : la detection est cloisonnee."""
    a = [
        PageText(
            parsed=parse_filename(f"Code A-{i}.txt"),
            text=f"MINISTERE DE LA JUSTICE\nDisposition numero {i} du premier code",
        )
        for i in range(1, 11)
    ]
    b = [
        PageText(
            parsed=parse_filename(f"Code B-{i}.txt"),
            text=f"CONSEIL CONSTITUTIONNEL\nDisposition numero {i} du second code",
        )
        for i in range(1, 11)
    ]
    nettoyees = clean_pages(a + b)

    for page_nettoyee in nettoyees:
        assert "MINISTERE DE LA JUSTICE" not in page_nettoyee.text
        assert "CONSEIL CONSTITUTIONNEL" not in page_nettoyee.text
        assert page_nettoyee.text.startswith("Disposition numero")


def test_document_sans_bandeau_inchange():
    sujets = [
        "De la garde a vue et de ses conditions",
        "Des perquisitions effectuees de nuit",
        "Du role du procureur de la Republique",
        "Des nullites de la procedure penale",
        "De la detention provisoire et sa duree",
        "Du controle judiciaire et ses obligations",
        "De l'instruction preparatoire",
        "Des expertises ordonnees par le juge",
        "De la chambre de l'instruction",
        "Des voies de recours extraordinaires",
    ]
    pages = [
        PageText(parsed=parse_filename(f"Code-{i}.txt"), text=sujet)
        for i, sujet in enumerate(sujets, start=1)
    ]
    assert [p.text for p in clean_pages(pages)] == [p.text for p in pages]


def test_trop_peu_de_pages_pour_conclure():
    """Sur deux pages, une repetition n'est pas une preuve."""
    assert detect(["Meme ligne\nA", "Meme ligne\nB"]) == set()


def test_metadonnees_conservees():
    pages = [page(i, f"Article {i}.") for i in range(1, 11)]
    nettoyees = clean_pages(pages)
    for avant, apres in zip(pages, nettoyees):
        assert apres.parsed == avant.parsed
        assert apres.page == avant.page


# -- l'effet recherche : le recollement entre pages ------------------------


def test_le_bandeau_ne_coupe_plus_les_articles_entre_pages():
    """Le vrai objectif : un article a cheval sur deux pages doit se
    reconstituer sans bandeau injecte au milieu."""
    pages = [page(i, f"Blabla page {i}.") for i in range(1, 21)]
    # Un article coupe entre la page 10 et la page 11.
    pages[9] = page(10, "Article 326. Aucune personne acquittee legalement ne peut")
    pages[10] = page(11, "plus etre reprise ou accusee a raison des memes faits.")

    nettoyees = clean_pages(pages)
    chunks = chunk_pages(nettoyees, TOKENIZER, chunk_tokens=400, overlap_tokens=100)

    phrase = "ne peut plus etre reprise ou accusee"
    assert any(phrase in c.text for c in chunks), (
        "l'article n'a pas ete reconstitue : le bandeau s'est intercale"
    )
    porteur = next(c for c in chunks if phrase in c.text)
    assert "Republique islamique" not in porteur.text


def test_page_jamais_videe_par_le_retrait():
    """Si tout serait supprime, on garde la page telle quelle."""
    cles = detect([f"Ligne de bandeau identique partout" for _ in range(10)])
    assert cles
    assert strip("Ligne de bandeau identique partout", cles) == "Ligne de bandeau identique partout"


def test_numeros_conserves_dans_la_cle():
    """Deux lignes ne differant que par un numero restent distinctes."""
    from legalrag.boilerplate import _key

    assert _key("Article 12 du present code") != _key("Article 13 du present code")


def test_bruit_ocr_final_absorbe_par_la_cle():
    from legalrag.boilerplate import _key

    assert _key("Honneur - Fraternite - Justice : LS") == _key("Honneur - Fraternite - Justice D")
