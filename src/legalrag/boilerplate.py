"""Retrait des en-tetes et pieds de page repetes.

Les corpus issus de numerisation portent un bandeau sur chaque page
(nom de l'institution, titre du document, numerotation). Ce bandeau pose
deux problemes pour un RAG :

1. Il casse le recollement des pages. Un article coupe entre les pages 128
   et 129 se retrouve, une fois les pages mises bout a bout, avec quatre
   lignes de bandeau injectees au milieu de la phrase. C'est precisement la
   coupure que le recollement cherchait a supprimer.
2. Il dilue chaque chunk a cheval sur deux pages, qui se met a partager du
   texte identique avec tous les autres chunks de frontiere.

La detection est statistique et non codee en dur : une ligne est un bandeau
si elle apparait en tete ou en pied d'une proportion importante des pages
du meme document. Rien n'est donc specifique a un corpus particulier.

Le bruit d'OCR empeche une comparaison exacte : la meme ligne ressort en
"Honneur - Fraternite - Justice : LS", "... : D" ou "... ARS" selon les
pages. Les lignes sont donc comparees apres normalisation et retrait des
jetons courts de fin de ligne, qui sont le bruit typique de l'OCR.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict

from .chunking import PageText

logger = logging.getLogger(__name__)

# Zone examinee a chaque extremite d'une page. Un bandeau vit la, pas au
# milieu du texte : chercher plus loin risquerait de supprimer du contenu.
HEAD_LINES = 4
TAIL_LINES = 3

# Proportion de pages ou la ligne doit apparaitre pour etre un bandeau.
MIN_RATIO = 0.5

# En deca, une ligne est trop courte pour etre identifiee de facon fiable.
MIN_NORMALIZED_LENGTH = 10

# Longueur maximale d'un jeton de bruit d'OCR en fin de ligne ("LS", "D",
# "T", "RS"...). Ces jetons sont retires pour construire la cle, ce qui
# reunit les variantes d'un meme bandeau. Les nombres sont conserves : ils
# distinguent des lignes de contenu par ailleurs identiques.
MAX_NOISE_TOKEN = 2

# Une ligne qui ouvre un article n'est jamais un bandeau, meme si elle est
# frequente. Garde-fou contre une suppression catastrophique.
_PROTECTED = re.compile(r"^\s*(article|art\.|titre|chapitre|section)\b", re.IGNORECASE)


def normalize(line: str) -> str:
    """Reduit une ligne a ses lettres et ses chiffres, casse et ponctuation
    mises de cote.

    Les chiffres sont conserves volontairement : sans eux, "Article 12" et
    "Article 13" deviendraient identiques, et une suite de lignes de contenu
    ne differant que par un numero serait prise pour un bandeau.
    """
    lowered = re.sub(r"[^a-zà-ÿ0-9 ]+", " ", line.lower())
    return re.sub(r"\s+", " ", lowered).strip()


def _key(line: str) -> str | None:
    """Cle de regroupement : la ligne normalisee, moins le bruit final.

    Le meme bandeau ressort en "... justice ls", "... justice d" ou
    "... justice t" selon les pages. Retirer les jetons courts de fin
    reunit ces variantes sans rapprocher des lignes reellement distinctes.
    """
    tokens = normalize(line).split()
    while tokens and len(tokens[-1]) <= MAX_NOISE_TOKEN and not tokens[-1].isdigit():
        tokens.pop()

    key = " ".join(tokens)
    if len(key) < MIN_NORMALIZED_LENGTH:
        return None
    return key


def detect(texts: list[str], min_ratio: float = MIN_RATIO) -> set[str]:
    """Cles des lignes de bandeau d'un document."""
    if len(texts) < 3:
        # Trop peu de pages pour distinguer une repetition d'une coincidence.
        return set()

    counts: Counter[str] = Counter()
    for text in texts:
        lines = [line for line in text.splitlines() if line.strip()]
        zone = lines[:HEAD_LINES] + lines[-TAIL_LINES:]
        # Une meme ligne presente deux fois sur la page ne compte qu'une fois.
        for key in {k for line in zone if (k := _key(line)) and not _PROTECTED.match(line)}:
            counts[key] += 1

    threshold = len(texts) * min_ratio
    return {key for key, count in counts.items() if count >= threshold}


def strip(text: str, keys: set[str]) -> str:
    """Retire les lignes de bandeau aux extremites de la page."""
    if not keys:
        return text

    lines = text.splitlines()
    # Index des lignes non vides : les zones se comptent en lignes de contenu,
    # pour qu'une ligne blanche en tete ne decale pas la fenetre.
    filled = [i for i, line in enumerate(lines) if line.strip()]
    if not filled:
        return text

    zone = set(filled[:HEAD_LINES]) | set(filled[-TAIL_LINES:])
    kept = [
        line
        for i, line in enumerate(lines)
        if i not in zone or _key(line) not in keys or _PROTECTED.match(line)
    ]

    result = "\n".join(kept).strip()

    # Filet de securite : si la detection a vu du bandeau partout, c'est
    # qu'elle s'est trompee. Mieux vaut une page avec son bandeau qu'une
    # page vide.
    if not result:
        logger.warning("Retrait annule sur une page : il n'en serait rien reste.")
        return text

    return result


def clean_pages(pages: list[PageText], min_ratio: float = MIN_RATIO) -> list[PageText]:
    """Detecte puis retire les bandeaux, document par document.

    La detection se fait par document : deux documents d'un meme corpus
    n'ont pas forcement le meme bandeau, et le titre en pied de page est
    par nature propre a chacun.
    """
    by_document: dict[str, list[PageText]] = defaultdict(list)
    for page in pages:
        by_document[page.document_id].append(page)

    cleaned: list[PageText] = []
    total_removed = 0

    for document_id, doc_pages in by_document.items():
        keys = detect([page.text for page in doc_pages], min_ratio)
        if not keys:
            cleaned.extend(doc_pages)
            continue

        removed_here = 0
        for page in doc_pages:
            new_text = strip(page.text, keys)
            removed_here += len(page.text) - len(new_text)
            cleaned.append(
                PageText(
                    parsed=page.parsed,
                    text=new_text,
                    drive_file_id=page.drive_file_id,
                )
            )

        total_removed += removed_here
        logger.info(
            "%s : %d motif(s) de bandeau retire(s), %d caracteres supprimes",
            doc_pages[0].parsed.document,
            len(keys),
            removed_here,
        )
        for key in sorted(keys):
            logger.debug("  motif : %r", key)

    if total_removed:
        original = sum(len(page.text) for page in pages)
        logger.info(
            "Bandeaux : %d caracteres retires sur %d (%.1f%% du corpus)",
            total_removed,
            original,
            total_removed / original * 100 if original else 0,
        )

    # L'ordre d'origine n'a pas d'importance pour la suite (le chunking
    # regroupe et trie), mais le conserver rend les logs lisibles.
    return sorted(cleaned, key=lambda p: (p.document_id, p.page))
