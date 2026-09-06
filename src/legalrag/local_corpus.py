"""Corpus lu depuis un repertoire local.

Alternative a Google Drive, exposant volontairement la meme surface :
`list_text_files()` puis `read_text()`. Le pipeline ne sait pas d'ou
viennent les fichiers, et n'a pas a le savoir.

Sert aux pages produites par l'ingestion de PDF, qu'ils aient ete
telecharges a la main depuis le Journal Officiel ou collectes sur un site
qui l'autorise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LocalFile:
    """Un fichier local, avec la meme surface qu'un fichier Drive."""

    id: str
    name: str
    path: Path
    modified_time: str = ""
    size: int | None = None


class LocalCorpus:
    """Lecture d'un corpus de `.txt` depuis un repertoire.

    Les arguments inutilises (`cache_dir`) sont acceptes pour que la classe
    reste interchangeable avec `DriveCorpus` : le pipeline construit sa
    source de la meme facon dans les deux cas.
    """

    def __init__(
        self,
        corpus_dir: Path,
        cache_dir: Path | None = None,
        recursive: bool = True,
    ) -> None:
        self._dir = Path(corpus_dir).expanduser()
        self._recursive = recursive

        if not self._dir.is_dir():
            raise FileNotFoundError(
                f"Repertoire de corpus introuvable : {self._dir}\n"
                "Verifiez LOCAL_CORPUS_DIR, ou lancez d'abord l'ingestion des PDF."
            )
        logger.info("Corpus local : %s", self._dir)

    def list_text_files(self, folder_id: str | None = None) -> list[LocalFile]:
        """Liste les `.txt`. `folder_id` est ignore, present par symetrie."""
        motif = "**/*.txt" if self._recursive else "*.txt"
        fichiers = sorted(p for p in self._dir.glob(motif) if p.is_file())

        return [
            LocalFile(
                # Le chemin relatif sert d'identifiant : stable, et unique
                # meme si deux sous-repertoires portent le meme nom de page.
                id=str(chemin.relative_to(self._dir)).replace("\\", "/"),
                name=chemin.name,
                path=chemin,
                modified_time=str(int(chemin.stat().st_mtime)),
                size=chemin.stat().st_size,
            )
            for chemin in fichiers
        ]

    def read_text(self, fichier: LocalFile) -> str:
        """Contenu du fichier. L'ingestion ecrit en UTF-8."""
        try:
            return fichier.path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("%s : pas de l'UTF-8, relecture en cp1252", fichier.name)
            return fichier.path.read_text(encoding="cp1252", errors="replace")
