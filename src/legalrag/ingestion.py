"""Ingestion de PDF : un document PDF devient des pages .txt.

Le pipeline attend des fichiers nommes `[horodatage-]document-page.txt`,
un par page. Ce module produit exactement cela a partir de PDF, quelle que
soit leur provenance : telecharges a la main depuis le Journal Officiel,
recuperes sur le site justice, ou deposes par un tiers.

Trois points meritent attention :

* **La langue se detecte sur le contenu, jamais sur le nom.** Sur le site
  justice, `code_des_mines.pdf` est en francais alors que rien dans son nom
  ne l'indique, tandis que `code_foncier_arabe.pdf` est bien en arabe. Les
  pages sont donc rangees dans un sous-repertoire par langue, ce qui permet
  d'alimenter des index separes.
* **Un PDF sans texte extractible est un PDF image.** Il faudrait l'OCR
  pour l'exploiter. Le module ne le fait pas : il le signale clairement
  plutot que de produire des pages vides qui pollueraient l'index.
* **L'horodatage vient du document**, pas de l'instant de l'ingestion :
  c'est ce qui permet de distinguer deux versions d'un meme texte.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# En dessous, une page est consideree comme vide : un PDF image renvoie
# souvent quelques caracteres parasites plutot que rien du tout.
MIN_CHARS_PAGE = 40

# Proportion de pages vides au-dela de laquelle le document entier est
# juge non exploitable sans OCR.
SEUIL_PAGES_VIDES = 0.8

# Suffixes que les sites ajoutent aux noms de fichiers et qui ne font pas
# partie du titre du texte.
_SUFFIXES_PARASITES = re.compile(
    r"(?i)[_-]*(arabe|arab|_ar|-ar|fr|francais|français|final|consolidee?|v?\d+)$"
)

# Caracteres interdits dans un nom de fichier Windows.
_INTERDITS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class IngestionError(RuntimeError):
    """Le PDF ne peut pas etre exploite."""


@dataclass
class RapportIngestion:
    """Ce qu'une ingestion a produit."""

    document: str = ""
    langue: str = "indetermine"
    pages_totales: int = 0
    pages_ecrites: int = 0
    pages_vides: int = 0
    horodatage: str = ""
    destination: Path | None = None
    fichiers: list[Path] = field(default_factory=list)
    ocr_necessaire: bool = False

    def resume(self) -> str:
        etat = "OCR NECESSAIRE" if self.ocr_necessaire else "ok"
        return (
            f"{self.document!r} [{self.langue}] : {self.pages_ecrites}/{self.pages_totales} "
            f"page(s) ecrite(s), {self.pages_vides} vide(s) — {etat}"
        )


def detecter_langue(texte: str) -> str:
    """Script dominant : 'fr' (latin), 'ar' (arabe), ou 'indetermine'.

    On raisonne sur le script et non sur la langue : distinguer francais et
    anglais n'a pas d'interet ici, alors que separer latin et arabe
    determine dans quel index le texte ira.
    """
    compte: Counter[str] = Counter()
    for caractere in texte:
        if not caractere.isalpha():
            continue
        nom = unicodedata.name(caractere, "")
        if "ARABIC" in nom:
            compte["ar"] += 1
        elif "LATIN" in nom:
            compte["fr"] += 1

    total = sum(compte.values())
    if total < 50:
        return "indetermine"

    dominant, occurrences = compte.most_common(1)[0]
    return dominant if occurrences / total > 0.6 else "indetermine"


def nom_de_document(chemin: Path, titre_pdf: str | None = None) -> str:
    """Deduit un titre lisible, a partir des metadonnees ou du nom de fichier.

    Les metadonnees priment quand elles existent et sont credibles : bien
    des PDF portent un titre de logiciel ("Microsoft Word - doc1") qui ne
    vaut pas mieux que le nom de fichier.
    """
    if titre_pdf and _titre_credible(titre_pdf):
        return _nettoyer(titre_pdf)

    base = chemin.stem
    # Les suffixes s'empilent : "statutpersonnelarabe-4" porte a la fois un
    # numero de version et une mention de langue. On retire jusqu'a
    # stabilite plutot qu'une seule fois.
    for _ in range(4):
        raccourci = _SUFFIXES_PARASITES.sub("", base)
        if raccourci == base or not raccourci:
            break
        base = raccourci
    base = base.replace("_", " ").replace("-", " ")
    base = re.sub(r"\s+", " ", base).strip()
    if not base:
        base = chemin.stem or "document"
    return _nettoyer(base[:1].upper() + base[1:])


def _titre_credible(titre: str) -> bool:
    """Un titre de metadonnees PDF merite-t-il d'etre retenu ?

    Ces champs sont remplis par le logiciel de production et contiennent
    le plus souvent des restes : "doc", "Microsoft Word - sans titre",
    "Untitled1". Un intitule de texte juridique fait plusieurs mots. On
    exige donc au moins deux mots et une longueur credible, ce qui elimine
    l'essentiel du bruit sans rejeter les vrais titres.
    """
    propre = titre.strip()
    if len(propre) < 10 or len(propre.split()) < 2:
        return False

    minuscule = propre.lower()
    rebuts = ("microsoft word", "untitled", "sans titre", "document1", "adobe", ".doc", ".pdf")
    return not any(rebut in minuscule for rebut in rebuts)


def _nettoyer(titre: str) -> str:
    """Rend un titre utilisable comme nom de fichier.

    Le numero de page final servant d'ancrage au parseur, un titre qui se
    terminerait lui-meme par un tiret suivi de chiffres creerait une
    ambiguite : on le neutralise.
    """
    titre = _INTERDITS.sub(" ", titre)
    titre = re.sub(r"\s+", " ", titre).strip(" .-")
    titre = re.sub(r"-(\d+)$", r" \1", titre)
    return titre or "document"


def _horodatage(chemin: Path, metadonnees) -> str:
    """Horodatage du document, au format YYYYMMDDHHMM.

    On prefere la date de modification declaree par le PDF a celle du
    fichier sur disque : un telechargement remet la seconde a zero et
    ferait passer deux versions differentes pour la meme.
    """
    brut = None
    if metadonnees is not None:
        brut = getattr(metadonnees, "modification_date", None) or getattr(
            metadonnees, "creation_date", None
        )

    if isinstance(brut, datetime):
        return brut.strftime("%Y%m%d%H%M")

    return datetime.fromtimestamp(chemin.stat().st_mtime).strftime("%Y%m%d%H%M")


def ingerer_pdf(
    source: Path,
    destination: Path,
    document: str | None = None,
    par_langue: bool = True,
    horodater: bool = True,
) -> RapportIngestion:
    """Decoupe un PDF en fichiers .txt, une page par fichier.

    Args:
        source: le PDF a lire.
        destination: repertoire racine de sortie.
        document: titre a utiliser ; deduit du PDF si absent.
        par_langue: ranger dans un sous-repertoire `fr/` ou `ar/`.
        horodater: prefixer les noms de fichiers par l'horodatage.

    Raises:
        IngestionError: fichier illisible ou sans aucun texte extractible.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependance declaree
        raise IngestionError("pypdf est requis pour lire les PDF") from exc

    if not source.is_file():
        raise IngestionError(f"fichier introuvable : {source}")

    try:
        lecteur = PdfReader(str(source))
        textes = [(page.extract_text() or "") for page in lecteur.pages]
    except Exception as exc:  # noqa: BLE001 - pypdf leve des types varies
        raise IngestionError(f"{source.name} illisible : {exc}") from exc

    rapport = RapportIngestion(pages_totales=len(textes))
    if not textes:
        raise IngestionError(f"{source.name} ne contient aucune page")

    rapport.pages_vides = sum(1 for t in textes if len(t.strip()) < MIN_CHARS_PAGE)
    rapport.ocr_necessaire = rapport.pages_vides >= len(textes) * SEUIL_PAGES_VIDES

    if rapport.ocr_necessaire:
        logger.warning(
            "%s : %d/%d pages sans texte extractible. C'est un PDF image, "
            "un OCR serait necessaire. Aucune page ecrite.",
            source.name,
            rapport.pages_vides,
            len(textes),
        )
        rapport.document = nom_de_document(source, _titre_metadonnees(lecteur))
        return rapport

    rapport.langue = detecter_langue("".join(textes[:10]))
    rapport.document = document or nom_de_document(source, _titre_metadonnees(lecteur))
    rapport.horodatage = _horodatage(source, getattr(lecteur, "metadata", None))

    dossier = destination / rapport.langue if par_langue else destination
    dossier.mkdir(parents=True, exist_ok=True)
    rapport.destination = dossier

    prefixe = f"{rapport.horodatage}-" if horodater and rapport.horodatage else ""

    for numero, texte in enumerate(textes, 1):
        if len(texte.strip()) < MIN_CHARS_PAGE:
            continue
        cible = dossier / f"{prefixe}{rapport.document}-{numero}.txt"
        # Ecriture atomique : une ingestion interrompue ne laisse pas de
        # page tronquee qui serait indexee telle quelle.
        temporaire = cible.with_suffix(".txt.part")
        temporaire.write_text(texte, encoding="utf-8")
        temporaire.replace(cible)
        rapport.fichiers.append(cible)
        rapport.pages_ecrites += 1

    logger.info("%s", rapport.resume())
    return rapport


def _titre_metadonnees(lecteur) -> str | None:
    metadonnees = getattr(lecteur, "metadata", None)
    if metadonnees is None:
        return None
    return getattr(metadonnees, "title", None)


def ingerer_repertoire(
    source: Path,
    destination: Path,
    par_langue: bool = True,
) -> list[RapportIngestion]:
    """Ingere tous les PDF d'un repertoire.

    Un PDF illisible n'interrompt pas le lot : il est signale et le
    traitement continue.
    """
    pdfs = sorted(p for p in source.rglob("*.pdf") if p.is_file())
    if not pdfs:
        logger.warning("Aucun PDF dans %s", source)
        return []

    rapports = []
    for pdf in pdfs:
        try:
            rapports.append(ingerer_pdf(pdf, destination, par_langue=par_langue))
        except IngestionError as exc:
            logger.error("Ignore : %s", exc)

    ecrites = sum(r.pages_ecrites for r in rapports)
    ocr = [r.document for r in rapports if r.ocr_necessaire]
    logger.info(
        "%d PDF traite(s), %d page(s) ecrite(s)%s",
        len(rapports),
        ecrites,
        f", {len(ocr)} necessitant un OCR" if ocr else "",
    )
    return rapports
