"""Orchestration du job d'indexation.

Enchainement : listing Drive -> parsing des noms -> telechargement (cache)
-> chunking -> filtrage de ce qui est deja plonge -> embeddings par lots
-> ecriture Parquet fragment par fragment.

Chaque lot est persiste des qu'il revient de l'API. Si le job meurt, tout
ce qui a ete ecrit est conserve et la reprise repart du lot suivant.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from .chunking import Chunk, PageText, chunk_pages
from .config import Settings
from .drive import DriveCorpus
from .embeddings import VoyageEmbedder
from .filename_parser import parse_filename_or_none
from .index_store import IndexStore, make_manifest
from .tokenization import get_tokenizer

logger = logging.getLogger(__name__)

# Tarifs publics au 6 septembre 2026, en dollars par million de tokens.
# Servent uniquement a afficher une estimation avant de lancer le job.
PRICE_PER_MTOK: dict[str, float] = {
    "voyage-context-4": 0.12,
    "voyage-4-large": 0.12,
    "voyage-4": 0.06,
    "voyage-4-lite": 0.02,
    "voyage-law-2": 0.12,
}


@dataclass
class RunReport:
    """Ce qui s'est passe pendant le job."""

    drive_files: int = 0
    skipped_filenames: list[str] = field(default_factory=list)
    pages_read: int = 0
    chunks_total: int = 0
    chunks_already_indexed: int = 0
    chunks_embedded: int = 0
    tokens_billed: int = 0
    parts_written: int = 0

    def summary(self) -> str:
        lines = [
            "----- Recapitulatif -----",
            f"Fichiers listes sur Drive     : {self.drive_files}",
            f"Fichiers hors convention      : {len(self.skipped_filenames)}",
            f"Pages lues                    : {self.pages_read}",
            f"Chunks produits               : {self.chunks_total}",
            f"  deja indexes (non repayes)  : {self.chunks_already_indexed}",
            f"  nouvellement plonges        : {self.chunks_embedded}",
            f"Fragments Parquet ecrits      : {self.parts_written}",
            f"Tokens factures (API)         : {self.tokens_billed}",
        ]
        return "\n".join(lines)


def estimate_cost(tokens: int, model: str) -> str:
    price = PRICE_PER_MTOK.get(model)
    if price is None:
        return "tarif inconnu pour ce modele"
    return f"~{tokens / 1_000_000 * price:.4f} USD ({tokens} tokens a {price} USD/Mtok)"


def collect_pages(settings: Settings, report: RunReport, limit: int | None = None) -> list[PageText]:
    """Liste, filtre et telecharge les pages du corpus."""
    corpus = DriveCorpus(
        credentials_file=settings.google_credentials_file,
        cache_dir=settings.cache_dir,
        recursive=settings.drive_recursive,
    )

    files = corpus.list_text_files(settings.drive_folder_id)
    report.drive_files = len(files)
    logger.info("%d fichier(s) .txt trouve(s) sur Drive", len(files))

    pages: list[PageText] = []
    for drive_file in files:
        parsed = parse_filename_or_none(drive_file.name)
        if parsed is None:
            # Un fichier hors convention ne doit pas interrompre le job.
            report.skipped_filenames.append(drive_file.name)
            continue

        try:
            text = corpus.read_text(drive_file)
        except Exception as exc:  # noqa: BLE001 - un fichier illisible ne bloque pas
            logger.error("Lecture impossible, fichier ignore : %s (%s)", drive_file.name, exc)
            report.skipped_filenames.append(drive_file.name)
            continue

        pages.append(PageText(parsed=parsed, text=text, drive_file_id=drive_file.id))

        if limit is not None and len(pages) >= limit:
            logger.info("Limite de %d page(s) atteinte (--limit)", limit)
            break

    report.pages_read = len(pages)
    if report.skipped_filenames:
        logger.warning(
            "%d fichier(s) ignore(s). Premiers exemples : %s",
            len(report.skipped_filenames),
            ", ".join(report.skipped_filenames[:5]),
        )
    return pages


def group_by_document(chunks: list[Chunk]) -> list[list[Chunk]]:
    """Regroupe les chunks par document, dans l'ordre.

    Le mode contextualise a besoin de ce regroupement : c'est ce qui permet
    au modele de voir le document entier autour de chaque chunk.
    """
    grouped: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.document_id].append(chunk)
    for group in grouped.values():
        group.sort(key=lambda c: c.chunk_index)
    return [grouped[key] for key in sorted(grouped)]


def run(settings: Settings, limit: int | None = None, dry_run: bool = False) -> RunReport:
    """Execute le job complet."""
    report = RunReport()

    tokenizer = get_tokenizer(settings.embedding_model)
    pages = collect_pages(settings, report, limit=limit)
    if not pages:
        logger.warning("Aucune page exploitable : rien a faire.")
        return report

    chunks = chunk_pages(
        pages,
        tokenizer,
        chunk_tokens=settings.chunk_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    report.chunks_total = len(chunks)
    if not chunks:
        return report

    store = IndexStore(
        settings.index_dir,
        make_manifest(
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
            mode=settings.embedding_mode,
            chunk_tokens=settings.chunk_tokens,
            overlap=settings.chunk_overlap_tokens,
        ),
    )

    done = store.existing_chunk_ids()
    pending = [chunk for chunk in chunks if chunk.chunk_id not in done]
    report.chunks_already_indexed = len(chunks) - len(pending)

    if not pending:
        logger.info("Index deja a jour : aucun appel a l'API.")
        return report

    tokens_to_bill = sum(chunk.token_count for chunk in pending)
    logger.info(
        "%d chunk(s) a plonger, %d deja indexe(s). Cout estime : %s",
        len(pending),
        report.chunks_already_indexed,
        estimate_cost(tokens_to_bill, settings.embedding_model),
    )
    if not tokenizer.exact:
        logger.info("Estimation approximative : le tokenizer exact n'etait pas disponible.")

    if dry_run:
        logger.info("--dry-run : arret avant tout appel a l'API.")
        return report

    embedder = VoyageEmbedder(
        api_key=settings.voyage_api_key,
        base_url=settings.voyage_base_url,
        model=settings.embedding_model,
        dimension=settings.embedding_dimension,
        contextualized=settings.is_contextualized,
        exact_tokenizer=tokenizer.exact,
        max_requests_per_minute=settings.max_requests_per_minute,
        max_tokens_per_minute=settings.max_tokens_per_minute,
    )

    groups = group_by_document(pending)
    for batch in embedder.embed(groups):
        store.append(batch)
        report.parts_written += 1
        report.chunks_embedded += len(batch)
        logger.info(
            "Progression : %d/%d chunk(s) plonge(s)", report.chunks_embedded, len(pending)
        )

    report.tokens_billed = embedder.total_tokens_billed
    return report
