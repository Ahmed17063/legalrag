"""Ecriture de l'index d'embeddings et reprise apres incident.

Format retenu : **Parquet**, ecrit en fragments (`parts/part-*.parquet`).

Pourquoi Parquet plutot que JSON ou NPY :

* colonnaire et typé : un vecteur de 1024 float32 pese 4 Ko, contre ~20 Ko
  en JSON une fois les flottants ecrits en texte ;
* le texte et les metadonnees vivent dans le meme fichier que les vecteurs,
  donc l'index reste auto-portant ;
* pyarrow lit un repertoire de fragments comme une seule table, et sait ne
  charger qu'une colonne, ce qui rend la reprise quasi instantanee.

Pourquoi des fragments plutot qu'un seul fichier : un fichier Parquet
tronque par un arret brutal est illisible dans son ensemble. Avec des
fragments ecrits atomiquement (fichier temporaire puis `rename`), un lot
interrompu ne laisse rien derriere lui, et tous les lots deja ecrits
restent valides.

La reprise repose sur `chunk_id`, qui derive du contenu : au redemarrage on
relit les identifiants deja presents et on ne renvoie a l'API que ce qui
manque. Rien n'est repaye.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .embeddings import EmbeddedChunk

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
PARTS_DIRNAME = "parts"
COMPACTED_NAME = "embeddings.parquet"


class IndexError_(RuntimeError):
    """Index incoherent avec la configuration demandee."""


def build_schema(dimension: int) -> pa.Schema:
    """Schema de l'index. La dimension du vecteur est figee par le modele."""
    return pa.schema(
        [
            pa.field("chunk_id", pa.string(), nullable=False),
            pa.field("document", pa.string(), nullable=False),
            pa.field("document_id", pa.string(), nullable=False),
            pa.field("chunk_index", pa.int32(), nullable=False),
            pa.field("text", pa.string(), nullable=False),
            pa.field("page_start", pa.int32(), nullable=False),
            pa.field("page_end", pa.int32(), nullable=False),
            pa.field("pages", pa.list_(pa.int32()), nullable=False),
            # Chaine vide quand le nom de fichier ne portait pas d'horodatage.
            pa.field("timestamp", pa.string(), nullable=False),
            pa.field("source_filenames", pa.list_(pa.string()), nullable=False),
            pa.field("token_count", pa.int32(), nullable=False),
            pa.field("embedding", pa.list_(pa.float32(), dimension), nullable=False),
        ]
    )


@dataclass(frozen=True, slots=True)
class IndexManifest:
    """Ce avec quoi l'index a ete construit. Sert de garde-fou a la reprise."""

    model: str
    dimension: int
    mode: str
    chunk_tokens: int
    chunk_overlap_tokens: int
    created_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "dimension": self.dimension,
            "mode": self.mode,
            "chunk_tokens": self.chunk_tokens,
            "chunk_overlap_tokens": self.chunk_overlap_tokens,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> IndexManifest:
        return cls(
            model=data["model"],
            dimension=int(data["dimension"]),
            mode=data["mode"],
            chunk_tokens=int(data.get("chunk_tokens", 0)),
            chunk_overlap_tokens=int(data.get("chunk_overlap_tokens", 0)),
            created_at=data.get("created_at", ""),
        )


class IndexStore:
    """Index d'embeddings sur disque local, resistant aux interruptions."""

    def __init__(self, index_dir: Path, manifest: IndexManifest) -> None:
        self._dir = index_dir
        self._parts_dir = index_dir / PARTS_DIRNAME
        self._manifest = manifest
        self._schema = build_schema(manifest.dimension)

        self._parts_dir.mkdir(parents=True, exist_ok=True)
        self._check_or_write_manifest()

    # -- manifeste ---------------------------------------------------------

    def _check_or_write_manifest(self) -> None:
        path = self._dir / MANIFEST_NAME
        if not path.is_file():
            path.write_text(
                json.dumps(self._manifest.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return

        existing = IndexManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
        incompatible = (
            existing.model != self._manifest.model
            or existing.dimension != self._manifest.dimension
            or existing.mode != self._manifest.mode
        )
        if incompatible:
            raise IndexError_(
                f"L'index de {self._dir} a ete construit avec "
                f"{existing.model} / {existing.dimension}d / mode {existing.mode}, "
                f"incompatible avec {self._manifest.model} / {self._manifest.dimension}d / "
                f"mode {self._manifest.mode}.\n"
                "Melanger deux modeles dans un meme index rendrait les distances "
                "incomparables. Utilisez un INDEX_DIR different, ou supprimez l'existant."
            )

        if (
            existing.chunk_tokens != self._manifest.chunk_tokens
            or existing.chunk_overlap_tokens != self._manifest.chunk_overlap_tokens
        ):
            logger.warning(
                "Parametres de chunking differents de ceux de l'index existant "
                "(%d/%d puis %d/%d) : les chunks changent d'identifiant et seront replonges.",
                existing.chunk_tokens,
                existing.chunk_overlap_tokens,
                self._manifest.chunk_tokens,
                self._manifest.chunk_overlap_tokens,
            )

    # -- reprise -----------------------------------------------------------

    def existing_chunk_ids(self) -> set[str]:
        """Identifiants deja plonges, relus depuis les fragments Parquet."""
        done: set[str] = set()
        for part in self._iter_parts():
            try:
                table = pq.read_table(part, columns=["chunk_id"])
            except Exception as exc:  # noqa: BLE001 - fragment illisible
                logger.error(
                    "Fragment illisible, ignore (il sera reconstruit) : %s (%s)", part.name, exc
                )
                continue
            done.update(table.column("chunk_id").to_pylist())

        if done:
            logger.info("Reprise : %d chunk(s) deja presents dans l'index", len(done))
        return done

    def _iter_parts(self) -> list[Path]:
        return sorted(self._parts_dir.glob("part-*.parquet"))

    def _next_part_path(self) -> Path:
        existing = self._iter_parts()
        index = 0
        if existing:
            index = int(existing[-1].stem.split("-")[-1]) + 1
        return self._parts_dir / f"part-{index:06d}.parquet"

    # -- ecriture ----------------------------------------------------------

    def append(self, embedded: Sequence[EmbeddedChunk]) -> Path | None:
        """Ecrit un lot dans un nouveau fragment, de facon atomique."""
        if not embedded:
            return None

        for item in embedded:
            if len(item.embedding) != self._manifest.dimension:
                raise IndexError_(
                    f"vecteur de dimension {len(item.embedding)} recu, "
                    f"{self._manifest.dimension} attendue (chunk {item.chunk.chunk_id})"
                )

        records = [item.chunk.as_record() for item in embedded]
        columns = {
            name: [record[name] for record in records]
            for name in self._schema.names
            if name != "embedding"
        }
        columns["embedding"] = [item.embedding for item in embedded]

        table = pa.Table.from_pydict(columns, schema=self._schema)

        target = self._next_part_path()
        tmp = target.with_suffix(".parquet.part")
        pq.write_table(table, tmp, compression="zstd")

        # `os.replace` est atomique : le fragment apparait complet ou pas du tout.
        os.replace(tmp, target)
        logger.debug("Fragment ecrit : %s (%d chunks)", target.name, len(embedded))
        return target

    # -- lecture / compactage ---------------------------------------------

    def count(self) -> int:
        return sum(pq.read_metadata(part).num_rows for part in self._iter_parts())

    def prune(self, valid_chunk_ids: set[str]) -> int:
        """Supprime les lignes qui ne correspondent plus au corpus courant.

        L'index est cumulatif : il conserve tout ce qui y a ete ecrit. Si
        le texte source change, ou si un premier passage a tourne sur un
        sous-ensemble (--limit) avec un decoupage different, les anciens
        chunks restent en place sans plus correspondre a rien. Ils feraient
        remonter a la recherche des fragments d'une version perimee.

        Returns:
            Le nombre de lignes supprimees.
        """
        removed = 0

        for part in self._iter_parts():
            table = pq.read_table(part, schema=self._schema)
            keep = [cid in valid_chunk_ids for cid in table.column("chunk_id").to_pylist()]

            if all(keep):
                continue

            removed += keep.count(False)
            filtered = table.filter(pa.array(keep))

            if filtered.num_rows == 0:
                part.unlink()
                continue

            tmp = part.with_suffix(".parquet.part")
            pq.write_table(filtered, tmp, compression="zstd")
            os.replace(tmp, part)

        if removed:
            logger.info("Elagage : %d ligne(s) obsolete(s) supprimee(s)", removed)
        return removed

    def compact(self, remove_parts: bool = False) -> Path | None:
        """Fusionne les fragments en un fichier unique, plus pratique a livrer."""
        parts = self._iter_parts()
        if not parts:
            logger.warning("Aucun fragment a compacter")
            return None

        target = self._dir / COMPACTED_NAME
        tmp = target.with_suffix(".parquet.part")

        with pq.ParquetWriter(tmp, self._schema, compression="zstd") as writer:
            for part in parts:
                writer.write_table(pq.read_table(part, schema=self._schema))

        os.replace(tmp, target)
        rows = pq.read_metadata(target).num_rows
        logger.info("Index compacte : %s (%d chunks)", target, rows)

        if remove_parts:
            for part in parts:
                part.unlink(missing_ok=True)
            logger.info("%d fragment(s) supprime(s)", len(parts))

        return target


def make_manifest(
    model: str, dimension: int, mode: str, chunk_tokens: int, overlap: int
) -> IndexManifest:
    return IndexManifest(
        model=model,
        dimension=dimension,
        mode=mode,
        chunk_tokens=chunk_tokens,
        chunk_overlap_tokens=overlap,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
