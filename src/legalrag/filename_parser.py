"""Parsing des noms de fichiers du corpus.

Convention (l'horodatage est optionnel) :

    [<horodatage>-]<nom_du_document>-<numero_de_page>.txt

    202311230853-Code de Procedure Penale-21.txt
        horodatage = 202311230853, document = "Code de Procedure Penale", page = 21
    Code de Procedure Penale-21.txt
        horodatage = None,         document = "Code de Procedure Penale", page = 21

Deux regles guident l'implementation :

1. On ancre sur le numero de page en FIN de nom. Un `split("-")` casserait
   tous les documents dont le titre contient un tiret ("Code de l'entree
   et du sejour des etrangers - CESEDA-12.txt").
2. Le prefixe horodatage n'est retire que si c'est une suite de chiffres
   de longueur d'horodatage QUI FORME UNE DATE VALIDE. Un simple bloc
   numerique en tete ("2023-Loi de finances-4.txt") est du titre, pas un
   horodatage, et doit etre conserve.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# Greedy : `.+` consomme le maximum, donc `-(\d+)$` s'accroche au DERNIER
# groupe de chiffres du nom. "Foo-12-34" -> document "Foo-12", page 34.
_PAGE_RE = re.compile(r"^(?P<head>.+)-(?P<page>\d+)$")

# Longueur d'horodatage -> format strptime associe.
_TIMESTAMP_FORMATS: dict[int, str] = {
    12: "%Y%m%d%H%M",     # YYYYMMDDHHMM   (convention du corpus)
    14: "%Y%m%d%H%M%S",   # YYYYMMDDHHMMSS (tolere)
}
# Alternance construite du plus long au plus court : sur "20231123085300-X-1"
# il faut essayer 14 chiffres avant 12, sinon le reste du nom serait tronque.
_TIMESTAMP_RE = re.compile(
    "^(?P<ts>%s)-(?P<rest>.+)$"
    % "|".join(rf"\d{{{n}}}" for n in sorted(_TIMESTAMP_FORMATS, reverse=True))
)

_SUPPORTED_SUFFIXES = (".txt",)


class FilenameConventionError(ValueError):
    """Le nom de fichier ne respecte pas la convention du corpus."""


@dataclass(frozen=True, slots=True)
class ParsedFilename:
    """Metadonnees extraites d'un nom de fichier."""

    source_filename: str
    document: str
    page: int
    timestamp: str | None = None
    timestamp_dt: datetime | None = None

    @property
    def document_id(self) -> str:
        """Identifiant stable et sans espaces, utilisable en nom de fichier."""
        return slugify(self.document)

    def as_metadata(self) -> dict[str, object]:
        """Projection destinee aux metadonnees portees par chaque chunk."""
        return {
            "document": self.document,
            "document_id": self.document_id,
            "page": self.page,
            # Chaine vide plutot que None : un horodatage absent est normal,
            # et les colonnes Parquet restent typees.
            "timestamp": self.timestamp or "",
            "source_filename": self.source_filename,
        }


def slugify(value: str) -> str:
    """Reduit un titre a un identifiant ascii minuscule separe par des tirets."""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_only).strip("-").lower()
    return slug or "document"


def _split_timestamp(head: str) -> tuple[str | None, datetime | None, str]:
    """Detache un eventuel prefixe horodatage de la partie titre.

    Renvoie ``(horodatage, datetime, titre)``. En l'absence d'horodatage
    valide, le ``head`` est renvoye intact comme titre.
    """
    match = _TIMESTAMP_RE.match(head)
    if match is None:
        return None, None, head

    raw = match.group("ts")
    rest = match.group("rest").strip()

    # "202311230853-21.txt" : apres retrait il ne reste plus de titre. On
    # prefere conserver le bloc numerique comme titre plutot que produire
    # un document sans nom.
    if not rest:
        return None, None, head

    try:
        parsed = datetime.strptime(raw, _TIMESTAMP_FORMATS[len(raw)])
    except ValueError:
        # Bonne longueur mais date impossible (mois 34, jour 99...) : ce
        # n'est pas un horodatage, c'est du titre.
        logger.debug("Bloc numerique %r non reconnu comme horodatage, conserve dans le titre", raw)
        return None, None, head

    return raw, parsed, rest


def parse_filename(filename: str) -> ParsedFilename:
    """Extrait document / page / horodatage d'un nom de fichier.

    Raises:
        FilenameConventionError: si le nom ne respecte pas la convention.
    """
    if not isinstance(filename, str) or not filename.strip():
        raise FilenameConventionError("nom de fichier vide")

    name = filename.strip()

    # On travaille sur le nom seul : un chemin complet est accepte.
    name = name.replace("\\", "/").rsplit("/", 1)[-1]

    lowered = name.lower()
    suffix = next((s for s in _SUPPORTED_SUFFIXES if lowered.endswith(s)), None)
    if suffix is None:
        raise FilenameConventionError(
            f"extension non supportee (attendu {' ou '.join(_SUPPORTED_SUFFIXES)}) : {filename!r}"
        )

    stem = name[: -len(suffix)].strip()
    if not stem:
        raise FilenameConventionError(f"nom vide avant l'extension : {filename!r}")

    page_match = _PAGE_RE.match(stem)
    if page_match is None:
        raise FilenameConventionError(
            f"numero de page absent en fin de nom : {filename!r}"
        )

    head = page_match.group("head").strip()
    page = int(page_match.group("page"))  # "007" -> 7

    if not head:
        raise FilenameConventionError(f"nom de document vide : {filename!r}")

    timestamp, timestamp_dt, document = _split_timestamp(head)

    document = document.strip(" -")
    if not document:
        raise FilenameConventionError(f"nom de document vide : {filename!r}")

    return ParsedFilename(
        source_filename=name,
        document=document,
        page=page,
        timestamp=timestamp,
        timestamp_dt=timestamp_dt,
    )


def parse_filename_or_none(filename: str) -> ParsedFilename | None:
    """Variante non levante : logge un avertissement et renvoie ``None``.

    C'est la porte d'entree utilisee par le pipeline : un fichier hors
    convention ne doit jamais interrompre le job.
    """
    try:
        return parse_filename(filename)
    except FilenameConventionError as exc:
        logger.warning("Fichier ignore (hors convention) : %s", exc)
        return None
