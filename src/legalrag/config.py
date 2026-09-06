"""Chargement de la configuration depuis l'environnement.

Aucune valeur secrete n'est ecrite en dur : tout vient de l'environnement,
alimente par un fichier `.env` local (ignore par git) ou par les variables
du systeme de CI/deploiement.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Modeles connus -> (mode d'appel, dimensions autorisees, contexte max par chunk).
KNOWN_MODELS: dict[str, tuple[str, frozenset[int], int]] = {
    "voyage-context-4": ("contextualized", frozenset({256, 512, 1024, 2048}), 32_000),
    "voyage-4-large": ("standard", frozenset({256, 512, 1024, 2048}), 32_000),
    "voyage-4": ("standard", frozenset({256, 512, 1024, 2048}), 32_000),
    "voyage-4-lite": ("standard", frozenset({256, 512, 1024, 2048}), 32_000),
    "voyage-law-2": ("standard", frozenset({1024}), 16_000),
}


class ConfigError(RuntimeError):
    """Configuration absente ou incoherente."""


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} doit etre un entier, recu {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "y", "oui"}


@dataclass(frozen=True, slots=True)
class Settings:
    """Parametres du job d'indexation."""

    voyage_api_key: str
    google_credentials_file: Path
    drive_folder_id: str
    drive_recursive: bool
    embedding_model: str
    embedding_mode: str
    embedding_dimension: int
    model_max_tokens: int
    chunk_tokens: int
    chunk_overlap_tokens: int
    max_requests_per_minute: int
    max_tokens_per_minute: int
    index_dir: Path
    cache_dir: Path

    @property
    def is_contextualized(self) -> bool:
        return self.embedding_mode == "contextualized"


def load_settings(env_file: Path | None = None) -> Settings:
    """Lit `.env` puis l'environnement, valide, et renvoie les parametres.

    Raises:
        ConfigError: variable manquante ou valeur incoherente.
    """
    _load_dotenv(env_file)

    missing: list[str] = []

    api_key = _env("VOYAGE_API_KEY")
    if not api_key:
        missing.append("VOYAGE_API_KEY")

    folder_id = _env("DRIVE_FOLDER_ID")
    if not folder_id:
        missing.append("DRIVE_FOLDER_ID")

    creds_raw = _env("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_raw:
        missing.append("GOOGLE_APPLICATION_CREDENTIALS")

    if missing:
        raise ConfigError(
            "Variables d'environnement manquantes : "
            + ", ".join(missing)
            + ".\nCopiez .env.example en .env et renseignez-les."
        )

    credentials_file = Path(creds_raw).expanduser()
    if not credentials_file.is_file():
        raise ConfigError(
            f"Fichier de credentials Google introuvable : {credentials_file}\n"
            "Voir la section 'Credentials Google Drive' du README."
        )

    model = _env("EMBEDDING_MODEL", "voyage-context-4")
    known = KNOWN_MODELS.get(model)
    if known is None:
        logger.warning(
            "Modele %r inconnu de ce projet : le mode et les limites sont pris "
            "dans l'environnement sans validation.",
            model,
        )
        default_mode, allowed_dims, max_tokens = "standard", None, 32_000
    else:
        default_mode, allowed_dims, max_tokens = known

    mode = _env("EMBEDDING_MODE", default_mode)
    if mode not in {"contextualized", "standard"}:
        raise ConfigError(f"EMBEDDING_MODE doit valoir 'contextualized' ou 'standard', recu {mode!r}")
    if known is not None and mode != default_mode:
        raise ConfigError(
            f"Le modele {model!r} s'appelle en mode {default_mode!r}, pas {mode!r}."
        )

    dimension = _env_int("EMBEDDING_DIMENSION", 1024)
    if allowed_dims is not None and dimension not in allowed_dims:
        raise ConfigError(
            f"EMBEDDING_DIMENSION={dimension} non supportee par {model!r} "
            f"(valeurs possibles : {sorted(allowed_dims)})"
        )

    chunk_tokens = _env_int("CHUNK_TOKENS", 800)
    overlap = _env_int("CHUNK_OVERLAP_TOKENS", 150)
    if chunk_tokens <= 0:
        raise ConfigError("CHUNK_TOKENS doit etre strictement positif")
    if not 0 <= overlap < chunk_tokens:
        raise ConfigError(
            f"CHUNK_OVERLAP_TOKENS ({overlap}) doit etre >= 0 et < CHUNK_TOKENS ({chunk_tokens}) ; "
            "sinon la fenetre n'avance pas et le chunking boucle."
        )
    if chunk_tokens > max_tokens:
        raise ConfigError(
            f"CHUNK_TOKENS ({chunk_tokens}) depasse le contexte de {model!r} ({max_tokens})"
        )

    return Settings(
        voyage_api_key=api_key,
        google_credentials_file=credentials_file,
        drive_folder_id=folder_id,
        drive_recursive=_env_bool("DRIVE_RECURSIVE", True),
        embedding_model=model,
        embedding_mode=mode,
        embedding_dimension=dimension,
        model_max_tokens=max_tokens,
        chunk_tokens=chunk_tokens,
        chunk_overlap_tokens=overlap,
        max_requests_per_minute=_env_int("MAX_REQUESTS_PER_MINUTE", 200),
        max_tokens_per_minute=_env_int("MAX_TOKENS_PER_MINUTE", 800_000),
        index_dir=Path(_env("INDEX_DIR", "./data/index")).expanduser(),
        cache_dir=Path(_env("CACHE_DIR", "./data/cache")).expanduser(),
    )


def _load_dotenv(env_file: Path | None) -> None:
    """Charge un .env s'il existe. Les variables deja definies gagnent."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dependance declaree dans requirements
        logger.debug("python-dotenv absent : lecture directe de l'environnement")
        return

    path = env_file or Path(".env")
    if path.is_file():
        load_dotenv(path, override=False)
        logger.debug("Configuration chargee depuis %s", path)
