"""Lecture du corpus sur Google Drive via un compte de service.

Le compte de service n'a besoin que du scope lecture seule : ce pipeline
n'ecrit jamais sur Drive. L'index est produit en local (voir index_store).

Les fichiers telecharges sont mis en cache sur disque, cle par identifiant
Drive et par version. Une reprise apres incident ne retelecharge donc que
ce qui a change en amont.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

_FOLDER_MIME = "application/vnd.google-apps.folder"

# Codes qui valent la peine d'etre rejoues : quota momentane et pannes
# transitoires cote Google. Un 403 de permission, lui, ne se resout pas
# en reessayant.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

_LIST_FIELDS = "nextPageToken, files(id, name, mimeType, size, modifiedTime, md5Checksum)"


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, HttpError):
        return getattr(exc.resp, "status", None) in _RETRYABLE_STATUS
    return isinstance(exc, (TimeoutError, ConnectionError, OSError))


_retry_drive = retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=1, max=30),
    reraise=True,
)


@dataclass(frozen=True, slots=True)
class DriveFile:
    """Un fichier tel que liste par l'API Drive."""

    id: str
    name: str
    modified_time: str
    md5_checksum: str | None = None
    size: int | None = None

    @property
    def cache_key(self) -> str:
        """Change des que le contenu distant change."""
        return self.md5_checksum or self.modified_time.replace(":", "").replace("-", "")


class DriveCorpus:
    """Acces en lecture au dossier Drive contenant le corpus."""

    def __init__(
        self,
        credentials_file: Path,
        cache_dir: Path,
        recursive: bool = True,
    ) -> None:
        self._cache_dir = cache_dir
        self._recursive = recursive
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        credentials = service_account.Credentials.from_service_account_file(
            str(credentials_file), scopes=SCOPES
        )
        # cache_discovery=False : evite un avertissement bruyant et un cache
        # disque inutile pour un job batch.
        self._service = build("drive", "v3", credentials=credentials, cache_discovery=False)
        logger.info("Client Drive initialise (compte de service : %s)", credentials.service_account_email)

    # -- listing -----------------------------------------------------------

    @_retry_drive
    def _list_page(self, folder_id: str, page_token: str | None) -> dict:
        return (
            self._service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields=_LIST_FIELDS,
                pageSize=1000,
                pageToken=page_token,
                orderBy="name",
                # Necessaire si le corpus vit sur un Drive partage.
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )

    def list_text_files(self, folder_id: str) -> list[DriveFile]:
        """Liste les .txt du dossier, en descendant dans les sous-dossiers."""
        found: list[DriveFile] = []
        subfolders: list[str] = []
        page_token: str | None = None

        while True:
            response = self._list_page(folder_id, page_token)
            for item in response.get("files", []):
                if item["mimeType"] == _FOLDER_MIME:
                    subfolders.append(item["id"])
                    continue
                if not item["name"].lower().endswith(".txt"):
                    logger.debug("Ignore (pas un .txt) : %s", item["name"])
                    continue
                size = item.get("size")
                found.append(
                    DriveFile(
                        id=item["id"],
                        name=item["name"],
                        modified_time=item.get("modifiedTime", ""),
                        md5_checksum=item.get("md5Checksum"),
                        size=int(size) if size is not None else None,
                    )
                )
            page_token = response.get("nextPageToken")
            if not page_token:
                break

        if self._recursive:
            for sub in subfolders:
                found.extend(self.list_text_files(sub))
        elif subfolders:
            logger.info("%d sous-dossier(s) ignore(s) (DRIVE_RECURSIVE=false)", len(subfolders))

        return found

    # -- telechargement ----------------------------------------------------

    def _cache_path(self, drive_file: DriveFile) -> Path:
        return self._cache_dir / f"{drive_file.id}.{drive_file.cache_key}.txt"

    @_retry_drive
    def _download_bytes(self, file_id: str) -> bytes:
        buffer = io.BytesIO()
        request = self._service.files().get_media(fileId=file_id, supportsAllDrives=True)
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return buffer.getvalue()

    def read_text(self, drive_file: DriveFile) -> str:
        """Contenu du fichier, depuis le cache si disponible."""
        cached = self._cache_path(drive_file)
        if cached.is_file():
            logger.debug("Cache : %s", drive_file.name)
            return cached.read_text(encoding="utf-8")

        raw = self._download_bytes(drive_file.id)
        text = _decode(raw, drive_file.name)

        # Ecriture atomique : un job interrompu ne laisse pas un fichier de
        # cache tronque qui serait relu tel quel au tour suivant.
        tmp = cached.with_suffix(".part")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(cached)

        # Les versions precedentes du meme fichier n'ont plus d'utilite.
        for stale in self._cache_dir.glob(f"{drive_file.id}.*.txt"):
            if stale != cached:
                stale.unlink(missing_ok=True)

        return text


def _decode(raw: bytes, filename: str) -> str:
    """Decode un .txt en essayant les encodages courants d'un export Windows."""
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if encoding not in ("utf-8-sig", "utf-8"):
            logger.warning("%s : decode en %s (pas de l'UTF-8)", filename, encoding)
        return text

    # latin-1 accepte tout octet, donc on n'arrive ici qu'en cas d'anomalie.
    logger.error("%s : decodage impossible, caracteres invalides remplaces", filename)
    return raw.decode("utf-8", errors="replace")
