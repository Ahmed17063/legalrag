"""Decoupage du corpus en chunks, metadonnees conservees.

Principe : on ne decoupe pas page par page. Les pages consecutives d'un
meme document sont d'abord recollees en un texte continu, puis decoupees
par fenetre glissante avec chevauchement.

C'est ce qui repond au probleme des articles coupes en fin de page : un
article a cheval sur les pages 20 et 21 se retrouve entier dans au moins
un chunk, au lieu de produire deux moities inexploitables. Chaque chunk
porte la liste des pages qu'il couvre, donc la citation reste precise.

Les pages manquantes coupent la continuite : si le corpus contient les
pages 20 et 22 mais pas la 21, on ne recolle pas 20 a 22 (cela fabriquerait
un faux voisinage). Chaque plage continue est traitee separement.
"""

from __future__ import annotations

import hashlib
import logging
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby

from .filename_parser import ParsedFilename

logger = logging.getLogger(__name__)

# Insere entre deux pages recollees. Compte dans le texte du chunk, et est
# rattache a la page qui precede pour le calcul de couverture.
PAGE_SEPARATOR = "\n\n"

# Ponctuations qui marquent une fin d'unite de sens. Si une page se termine
# par l'une d'elles, la page suivante commence un nouveau paragraphe. Sinon
# la phrase court d'une page a l'autre et on se contente d'une espace :
# inserer une coupure de paragraphe au milieu d'un article degraderait le
# texte recolle.
_SENTENCE_END = frozenset('.!?:;"»)]')

# Fraction de la fin d'un chunk dans laquelle on cherche une frontiere
# naturelle (paragraphe, phrase) plutot que de couper au milieu d'un mot.
_SNAP_WINDOW_RATIO = 0.20

# Frontieres candidates, de la plus souhaitable a la moins souhaitable.
_SNAP_PATTERNS = ("\n\n", "\n", ". ", " ")


@dataclass(frozen=True, slots=True)
class PageText:
    """Le contenu d'une page, avec les metadonnees issues de son nom."""

    parsed: ParsedFilename
    text: str
    drive_file_id: str = ""

    @property
    def page(self) -> int:
        return self.parsed.page

    @property
    def document_id(self) -> str:
        return self.parsed.document_id


@dataclass(frozen=True, slots=True)
class Chunk:
    """Un fragment pret a etre plonge, avec sa tracabilite."""

    chunk_id: str
    document: str
    document_id: str
    chunk_index: int
    text: str
    page_start: int
    page_end: int
    pages: tuple[int, ...]
    timestamp: str
    source_filenames: tuple[str, ...]
    token_count: int

    def as_record(self) -> dict[str, object]:
        """Projection a plat, ecrite telle quelle dans l'index."""
        return {
            "chunk_id": self.chunk_id,
            "document": self.document,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "pages": list(self.pages),
            "timestamp": self.timestamp,
            "source_filenames": list(self.source_filenames),
            "token_count": self.token_count,
        }


def make_chunk_id(document_id: str, chunk_index: int, text: str) -> str:
    """Identifiant deterministe, derive du contenu.

    Le meme texte au meme rang redonne le meme identifiant : c'est ce qui
    permet a une reprise de sauter ce qui est deja plonge. A l'inverse, si
    le texte source change, l'identifiant change et le chunk est replonge.
    """
    payload = f"{document_id}|{chunk_index}|{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def dedupe_pages(pages: list[PageText]) -> list[PageText]:
    """Une seule version par (document, page) : la plus recemment horodatee.

    Le corpus peut contenir la meme page exportee plusieurs fois. Sans
    horodatage pour departager, on garde la premiere rencontree.
    """
    best: dict[tuple[str, int], PageText] = {}
    duplicates = 0

    for page in pages:
        key = (page.document_id, page.page)
        current = best.get(key)
        if current is None:
            best[key] = page
            continue

        duplicates += 1
        new_ts = page.parsed.timestamp_dt
        old_ts = current.parsed.timestamp_dt
        if new_ts is not None and (old_ts is None or new_ts > old_ts):
            best[key] = page

    if duplicates:
        logger.info(
            "%d page(s) en double ecartee(s), version la plus recente conservee",
            duplicates,
        )

    return sorted(best.values(), key=lambda p: (p.document_id, p.page))


def consecutive_runs(pages: list[PageText]) -> list[list[PageText]]:
    """Regroupe les pages d'un document en plages de numeros consecutifs."""
    ordered = sorted(pages, key=lambda p: p.page)
    runs: list[list[PageText]] = []
    for _, group in groupby(enumerate(ordered), key=lambda pair: pair[1].page - pair[0]):
        runs.append([page for _, page in group])
    return runs


def _snap_end(text: str, start_char: int, ideal_end: int) -> int:
    """Recule la fin du chunk jusqu'a une frontiere naturelle proche."""
    span = ideal_end - start_char
    if span <= 0:
        return ideal_end

    floor = start_char + int(span * (1 - _SNAP_WINDOW_RATIO))
    for pattern in _SNAP_PATTERNS:
        found = text.rfind(pattern, floor, ideal_end)
        if found > start_char:
            return found + len(pattern)
    return ideal_end


def _first_token_at_or_after(token_starts: list[int], char_pos: int) -> int:
    return bisect_left(token_starts, char_pos)


def page_separator(previous_text: str) -> str:
    """Choisit la jointure entre une page et la suivante."""
    stripped = previous_text.rstrip()
    if not stripped:
        return PAGE_SEPARATOR
    return PAGE_SEPARATOR if stripped[-1] in _SENTENCE_END else " "


def _pages_covering(
    page_spans: list[tuple[int, int, int]], start: int, end: int
) -> tuple[int, ...]:
    """Pages dont le span de caracteres croise [start, end)."""
    covered = [
        page
        for page, span_start, span_end in page_spans
        if span_start < end and span_end > start
    ]
    return tuple(sorted(set(covered)))


def chunk_run(
    run: list[PageText],
    tokenizer,
    chunk_tokens: int,
    overlap_tokens: int,
    start_index: int = 0,
) -> list[Chunk]:
    """Decoupe une plage de pages consecutives d'un meme document."""
    parts: list[str] = []
    page_spans: list[tuple[int, int, int]] = []  # (page, debut, fin) en caracteres
    cursor = 0

    for position, page in enumerate(run):
        body = page.text
        if position < len(run) - 1:
            body += page_separator(body)
        parts.append(body)
        page_spans.append((page.page, cursor, cursor + len(body)))
        cursor += len(body)

    run_text = "".join(parts)
    if not run_text.strip():
        logger.debug(
            "Plage vide ignoree : %s pages %d-%d",
            run[0].parsed.document,
            run[0].page,
            run[-1].page,
        )
        return []

    offsets = tokenizer.offsets(run_text)
    if not offsets:
        return []

    token_starts = [span[0] for span in offsets]
    total_tokens = len(offsets)
    step = max(1, chunk_tokens - overlap_tokens)

    by_page = {page.page: page for page in run}
    chunks: list[Chunk] = []
    chunk_index = start_index
    start_tok = 0

    while start_tok < total_tokens:
        end_tok = min(start_tok + chunk_tokens, total_tokens)
        is_last = end_tok >= total_tokens

        start_char = offsets[start_tok][0]
        ideal_end = offsets[end_tok - 1][1]
        end_char = ideal_end if is_last else _snap_end(run_text, start_char, ideal_end)

        raw = run_text[start_char:end_char]
        text = raw.strip()

        if text:
            # Les pages sont calculees sur le texte reellement retenu, apres
            # strip, pour ne pas attribuer une page dont seul le separateur
            # aurait ete inclus.
            offset_left = len(raw) - len(raw.lstrip())
            offset_right = len(raw) - len(raw.rstrip())
            pages = _pages_covering(
                page_spans, start_char + offset_left, end_char - offset_right
            )
            if not pages:
                pages = (run[0].page,)

            first_page = by_page[pages[0]]
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(first_page.document_id, chunk_index, text),
                    document=first_page.parsed.document,
                    document_id=first_page.document_id,
                    chunk_index=chunk_index,
                    text=text,
                    page_start=pages[0],
                    page_end=pages[-1],
                    pages=pages,
                    timestamp=first_page.parsed.timestamp or "",
                    source_filenames=tuple(
                        by_page[p].parsed.source_filename for p in pages if p in by_page
                    ),
                    token_count=end_tok - start_tok,
                )
            )
            chunk_index += 1

        if is_last:
            break

        # Le snapping a pu reculer la fin : on repart du token reellement
        # atteint, sinon le chevauchement demande ne serait pas respecte.
        reached_tok = max(start_tok + 1, _first_token_at_or_after(token_starts, end_char))
        start_tok = max(start_tok + step, reached_tok - overlap_tokens)

    return chunks


def chunk_pages(
    pages: list[PageText],
    tokenizer,
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Decoupe l'ensemble du corpus, document par document."""
    unique = dedupe_pages(pages)

    by_document: dict[str, list[PageText]] = defaultdict(list)
    for page in unique:
        by_document[page.document_id].append(page)

    all_chunks: list[Chunk] = []
    for document_id, doc_pages in sorted(by_document.items()):
        runs = consecutive_runs(doc_pages)
        if len(runs) > 1:
            logger.warning(
                "%s : %d plages de pages non contigues (%s). Pas de recollement entre elles.",
                doc_pages[0].parsed.document,
                len(runs),
                ", ".join(f"{r[0].page}-{r[-1].page}" for r in runs),
            )

        index = 0
        for run in runs:
            run_chunks = chunk_run(
                run, tokenizer, chunk_tokens, overlap_tokens, start_index=index
            )
            index += len(run_chunks)
            all_chunks.extend(run_chunks)

        logger.debug("%s : %d pages -> %d chunks", document_id, len(doc_pages), index)

    logger.info(
        "Chunking : %d pages -> %d chunks (%d documents)",
        len(unique),
        len(all_chunks),
        len(by_document),
    )
    return all_chunks
