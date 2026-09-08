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


# Voyage AI appartient a MongoDB. Il existe deux portes d'entree, et la
# bonne depend de l'endroit ou la cle a ete creee :
#   - cle creee dans MongoDB Atlas  -> ai.mongodb.com (cas des comptes recents)
#   - cle d'un ancien compte Voyage -> api.voyageai.com (prefixe "pa-")
# Une cle limitee a une geographie doit viser l'endpoint correspondant,
# sinon l'appel echoue.
VOYAGE_LEGACY_URL = "https://api.voyageai.com/v1"
VOYAGE_ATLAS_URL = "https://ai.mongodb.com/v1"
VOYAGE_ATLAS_EU_URL = "https://eu.ai.mongodb.com/v1"
VOYAGE_ATLAS_US_URL = "https://us.ai.mongodb.com/v1"


class ConfigError(RuntimeError):
    """Configuration absente ou incoherente."""


def resolve_base_url(api_key: str, override: str | None = None) -> str:
    """Deduit l'URL de l'API a partir du prefixe de la cle.

    Reproduit le routage automatique du client officiel. `override`
    (VOYAGE_API_BASE_URL) court-circuite la deduction si besoin.
    """
    if override:
        return override.rstrip("/")

    if api_key.startswith("pa-"):
        return VOYAGE_LEGACY_URL
    if api_key.startswith("al-eu-"):
        return VOYAGE_ATLAS_EU_URL
    if api_key.startswith("al-us-"):
        return VOYAGE_ATLAS_US_URL

    # Cle Atlas sans restriction de geographie : pas de prefixe distinctif.
    return VOYAGE_ATLAS_URL


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
    voyage_base_url: str
    google_credentials_file: Path
    drive_folder_id: str
    drive_recursive: bool
    corpus_source: str
    local_corpus_dir: Path
    embedding_model: str
    embedding_mode: str
    embedding_dimension: int
    model_max_tokens: int
    chunk_tokens: int
    chunk_overlap_tokens: int
    strip_boilerplate: bool
    max_requests_per_minute: int
    max_tokens_per_minute: int
    index_dir: Path
    cache_dir: Path
    qdrant_collection: str
    qdrant_url: str | None
    qdrant_api_key: str | None
    qdrant_path: Path
    qdrant_batch: int
    rerank_model: str
    rerank_enabled: bool
    retrieval_top_k: int
    retrieval_top_n: int
    statuts_file: Path
    groq_api_key: str | None
    generation_model: str
    generation_max_tokens: int

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

    source = _env("CORPUS_SOURCE", "drive")
    if source not in {"drive", "local"}:
        raise ConfigError(f"CORPUS_SOURCE doit valoir 'drive' ou 'local', recu {source!r}")

    # Les credentials Google n'ont de sens que si le corpus vient de Drive.
    folder_id = _env("DRIVE_FOLDER_ID") or ""
    creds_raw = _env("GOOGLE_APPLICATION_CREDENTIALS") or ""
    if source == "drive":
        if not folder_id:
            missing.append("DRIVE_FOLDER_ID")
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
        voyage_base_url=resolve_base_url(api_key, _env("VOYAGE_API_BASE_URL")),
        google_credentials_file=credentials_file,
        drive_folder_id=folder_id,
        drive_recursive=_env_bool("DRIVE_RECURSIVE", True),
        corpus_source=source,
        local_corpus_dir=Path(_env("LOCAL_CORPUS_DIR", "./data/corpus/fr")).expanduser(),
        embedding_model=model,
        embedding_mode=mode,
        embedding_dimension=dimension,
        model_max_tokens=max_tokens,
        chunk_tokens=chunk_tokens,
        chunk_overlap_tokens=overlap,
        strip_boilerplate=_env_bool("STRIP_BOILERPLATE", True),
        max_requests_per_minute=_env_int("MAX_REQUESTS_PER_MINUTE", 200),
        max_tokens_per_minute=_env_int("MAX_TOKENS_PER_MINUTE", 800_000),
        index_dir=Path(_env("INDEX_DIR", "./data/index")).expanduser(),
        cache_dir=Path(_env("CACHE_DIR", "./data/cache")).expanduser(),
        qdrant_collection=_env("QDRANT_COLLECTION", "legalrag"),
        qdrant_url=_env("QDRANT_URL"),
        qdrant_api_key=_env("QDRANT_API_KEY"),
        qdrant_path=Path(_env("QDRANT_PATH", "./data/qdrant")).expanduser(),
        qdrant_batch=_env_int("QDRANT_BATCH", 128),
        rerank_model=_env("RERANK_MODEL", "rerank-3"),
        rerank_enabled=_env_bool("RERANK_ENABLED", True),
        retrieval_top_k=_env_int("RETRIEVAL_TOP_K", 25),
        retrieval_top_n=_env_int("RETRIEVAL_TOP_N", 5),
        statuts_file=Path(_env("STATUTS_FILE", "./data/statuts.json")).expanduser(),
        groq_api_key=_env("GROQ_API_KEY"),
        generation_model=_env("GENERATION_MODEL", "openai/gpt-oss-120b"),
        generation_max_tokens=_env_int("GENERATION_MAX_TOKENS", 2048),
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
