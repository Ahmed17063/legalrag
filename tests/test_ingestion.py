"""Tests de l'ingestion de PDF vers des pages .txt."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from legalrag.filename_parser import parse_filename
from legalrag.ingestion import (
    IngestionError,
    detecter_langue,
    ingerer_pdf,
    ingerer_repertoire,
    nom_de_document,
)


# ---------------------------------------------------------------------------
# Detection de langue
# ---------------------------------------------------------------------------


def test_francais_detecte():
    texte = (
        "Article premier. La presente loi porte code minier et fixe les conditions "
        "de prospection, de recherche et d'exploitation des substances minerales."
    )
    assert detecter_langue(texte) == "fr"


def test_arabe_detecte():
    texte = "المادة الأولى تحدد هذه المدونة شروط التنقيب والبحث واستغلال المواد المعدنية في البلاد" * 2
    assert detecter_langue(texte) == "ar"


def test_texte_trop_court_indetermine():
    assert detecter_langue("Article 1.") == "indetermine"


def test_melange_sans_dominante():
    """Un document reellement bilingue ne doit pas etre range d'office."""
    texte = "Article premier du present code minier " * 3 + "المادة الأولى من هذه المدونة " * 4
    assert detecter_langue(texte) in {"ar", "fr", "indetermine"}


# ---------------------------------------------------------------------------
# Nom de document
# ---------------------------------------------------------------------------


def test_nom_deduit_du_fichier():
    assert nom_de_document(Path("code_des_mines.pdf")) == "Code des mines"


@pytest.mark.parametrize(
    ("fichier", "attendu"),
    [
        ("code_foncier_arabe.pdf", "Code foncier"),
        ("code_des_peches_ar-2.pdf", "Code des peches"),
        ("constitution_consolidee_fr-2.pdf", "Constitution"),
        ("statutpersonnelarabe-4.pdf", "Statutpersonnel"),
    ],
)
def test_suffixes_parasites_retires(fichier, attendu):
    """Les sites empilent langue et numero de version dans le nom."""
    assert nom_de_document(Path(fichier)) == attendu


def test_titre_de_metadonnees_credible_retenu():
    titre = "Loi portant Code minier de la Republique"
    assert nom_de_document(Path("x.pdf"), titre) == titre


@pytest.mark.parametrize(
    "titre",
    [
        "Microsoft Word - Loi 99-013.doc",
        "doc",
        "Untitled1",
        "sans titre",
        "rapport.pdf",
        "abc",
    ],
)
def test_titre_de_metadonnees_douteux_ecarte(titre):
    """Ces champs sont remplis par le logiciel, pas par un juriste."""
    assert nom_de_document(Path("code_des_mines.pdf"), titre) == "Code des mines"


def test_nom_ne_cree_pas_d_ambiguite_de_page():
    """Un titre finissant par '-12' brouillerait l'ancrage sur la page."""
    nom = nom_de_document(Path("x.pdf"), "Ordonnance relative aux societes-12")
    assert not nom.endswith("-12")
    # Et le nom de page reste analysable.
    assert parse_filename(f"{nom}-3.txt").page == 3


def test_caracteres_interdits_neutralises():
    nom = nom_de_document(Path("x.pdf"), 'Loi n° 99/013 : dispositions "diverses"')
    for interdit in '<>:"/\\|?*':
        assert interdit not in nom


# ---------------------------------------------------------------------------
# Ingestion, avec un PDF construit pour le test
# ---------------------------------------------------------------------------


def ecrire_pdf(chemin: Path, pages: list[str]) -> Path:
    """Fabrique un PDF minimal mais valide, dont le texte est extractible.

    Ecrit a la main plutot que via pypdf : une page produite par
    `add_blank_page` n'a pas de ressource de police, et rien n'en est alors
    extractible — ce qui ferait passer tous les PDF de test pour des
    documents images.
    """
    objets: list[bytes] = []

    def ajouter(corps: bytes) -> int:
        objets.append(corps)
        return len(objets)  # les numeros d'objet commencent a 1

    police = ajouter(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    numeros_pages: list[int] = []
    identifiant_pages = len(pages) * 2 + 3  # reserve pour le noeud /Pages

    for texte in pages:
        lignes = []
        for position, morceau in enumerate(texte.split(". ")):
            echappe = morceau.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            lignes.append(f"BT /F1 11 Tf 40 {780 - position * 16} Td ({echappe}) Tj ET")
        flux = "\n".join(lignes).encode("latin-1", "replace")
        contenu = ajouter(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(flux), flux))
        numeros_pages.append(
            ajouter(
                b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 595 842] "
                b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>"
                % (identifiant_pages, police, contenu)
            )
        )

    kids = b" ".join(b"%d 0 R" % n for n in numeros_pages)
    noeud_pages = ajouter(
        b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, len(numeros_pages))
    )
    catalogue = ajouter(b"<< /Type /Catalog /Pages %d 0 R >>" % noeud_pages)

    sortie = bytearray(b"%PDF-1.4\n")
    positions = [0]
    for numero, corps in enumerate(objets, 1):
        positions.append(len(sortie))
        sortie += b"%d 0 obj\n%s\nendobj\n" % (numero, corps)

    debut_xref = len(sortie)
    sortie += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objets) + 1)
    for position in positions[1:]:
        sortie += b"%010d 00000 n \n" % position
    sortie += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objets) + 1,
        catalogue,
        debut_xref,
    )

    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_bytes(bytes(sortie))
    return chemin


@pytest.fixture
def pdf_francais(tmp_path):
    pages = [
        "Article premier de la presente loi portant code minier national. "
        "Elle fixe les conditions de prospection et de recherche miniere.",
        "Article deux relatif aux titres miniers delivres par le ministre "
        "charge des mines dans les conditions prevues par decret.",
    ]
    return ecrire_pdf(tmp_path / "code_des_mines.pdf", pages)


def test_pages_ecrites_selon_la_convention(pdf_francais, tmp_path):
    rapport = ingerer_pdf(pdf_francais, tmp_path / "corpus")

    assert rapport.pages_ecrites == 2
    assert rapport.document == "Code des mines"

    for fichier in rapport.fichiers:
        analyse = parse_filename(fichier.name)
        assert analyse.document == "Code des mines"
        assert 1 <= analyse.page <= 2
        assert analyse.timestamp  # horodatage present


def test_rangement_par_langue(pdf_francais, tmp_path):
    rapport = ingerer_pdf(pdf_francais, tmp_path / "corpus")
    assert rapport.destination == tmp_path / "corpus" / "fr"
    assert all(f.parent.name == "fr" for f in rapport.fichiers)


def test_sans_rangement_par_langue(pdf_francais, tmp_path):
    rapport = ingerer_pdf(pdf_francais, tmp_path / "corpus", par_langue=False)
    assert rapport.destination == tmp_path / "corpus"


def test_titre_impose(pdf_francais, tmp_path):
    rapport = ingerer_pdf(pdf_francais, tmp_path / "c", document="Code minier 1999")
    assert rapport.document == "Code minier 1999"
    assert parse_filename(rapport.fichiers[0].name).document == "Code minier 1999"


def test_contenu_conserve(pdf_francais, tmp_path):
    rapport = ingerer_pdf(pdf_francais, tmp_path / "corpus")
    premiere = next(f for f in rapport.fichiers if f.name.endswith("-1.txt"))
    assert "code minier" in premiere.read_text(encoding="utf-8").lower()


def test_pdf_image_signale_sans_rien_ecrire(tmp_path):
    """Un PDF sans texte extractible produirait des pages vides."""
    vide = ecrire_pdf(tmp_path / "scan.pdf", ["", "", ""])
    rapport = ingerer_pdf(vide, tmp_path / "corpus")

    assert rapport.ocr_necessaire
    assert rapport.pages_ecrites == 0
    assert not (tmp_path / "corpus").exists() or not list((tmp_path / "corpus").rglob("*.txt"))


def test_fichier_absent(tmp_path):
    with pytest.raises(IngestionError, match="introuvable"):
        ingerer_pdf(tmp_path / "rien.pdf", tmp_path / "corpus")


def test_fichier_illisible(tmp_path):
    faux = tmp_path / "casse.pdf"
    faux.write_bytes(b"ceci n'est pas un PDF")
    with pytest.raises(IngestionError):
        ingerer_pdf(faux, tmp_path / "corpus")


def test_ingestion_idempotente(pdf_francais, tmp_path):
    """Rejouer l'ingestion ne doit pas dupliquer les pages."""
    sortie = tmp_path / "corpus"
    ingerer_pdf(pdf_francais, sortie)
    ingerer_pdf(pdf_francais, sortie)
    assert len(list(sortie.rglob("*.txt"))) == 2


def test_aucun_fichier_temporaire_laisse(pdf_francais, tmp_path):
    sortie = tmp_path / "corpus"
    ingerer_pdf(pdf_francais, sortie)
    assert not list(sortie.rglob("*.part"))


# ---------------------------------------------------------------------------
# Ingestion d'un repertoire
# ---------------------------------------------------------------------------


def test_repertoire_complet(tmp_path):
    source = tmp_path / "pdfs"
    ecrire_pdf(source / "code_des_mines.pdf", ["Article premier du code minier national."])
    ecrire_pdf(source / "code_de_commerce.pdf", ["Article premier du code de commerce."])

    rapports = ingerer_repertoire(source, tmp_path / "corpus")

    assert len(rapports) == 2
    assert {r.document for r in rapports} == {"Code des mines", "Code de commerce"}


def test_un_pdf_casse_n_arrete_pas_le_lot(tmp_path):
    source = tmp_path / "pdfs"
    ecrire_pdf(source / "bon.pdf", ["Article premier du code minier national."])
    (source / "casse.pdf").write_bytes(b"pas un PDF")

    rapports = ingerer_repertoire(source, tmp_path / "corpus")
    assert len(rapports) == 1


def test_repertoire_vide(tmp_path):
    vide = tmp_path / "rien"
    vide.mkdir()
    assert ingerer_repertoire(vide, tmp_path / "corpus") == []
