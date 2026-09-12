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

# Niveaux d'effort acceptes par l'API Anthropic pour le raisonnement.
EFFORTS_ANTHROPIC = frozenset({"low", "medium", "high", "xhigh", "max"})


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


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} doit etre un nombre, recu {raw!r}") from exc


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

    # Temperature de redaction : basse, on restitue du droit. Ne s'applique
    # qu'a Groq — Claude Sonnet 5 rejette en 400 toute valeur non par
    # defaut, quelle qu'elle soit. Voir generation.py.
    generation_temperature: float = 0.1

    # Reformulation de la question avant la recherche vectorielle. La
    # temperature y est plus haute : on cherche de la variete lexicale,
    # l'inverse du besoin de la redaction. Voir reformulation.py.
    reformulation_enabled: bool = True
    reformulation_temperature: float = 0.4
    # Large pour une phrase, et c'est voulu : les modeles a raisonnement
    # (gpt-oss chez Groq) depensent ce budget en raisonnement AVANT d'ecrire.
    # A 200, le raisonnement consommait tout et la reformulation revenait
    # vide. Mesure sur gpt-oss-120b : 145 a 284 tokens de raisonnement, 360
    # au total dans le pire cas.
    reformulation_max_tokens: int = 800
    # Modele charge de la reformulation. Vide = le defaut de generation.
    # Une tache mecanique et courte : le modele le plus rapide suffit.
    reformulation_model: str = ""

    # Deuxieme fournisseur de generation. Les champs portent une valeur par
    # defaut : les sites de construction existants (tests compris) n'ont pas
    # a etre repris, et une installation sans cle Anthropic fonctionne comme
    # avant, avec Groq seul.
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5"
    # Ce plafond couvre le raisonnement *et* la reponse : Claude Sonnet 5
    # active le raisonnement adaptatif par defaut, et `max_tokens` borne le
    # total. Une valeur calquee sur GENERATION_MAX_TOKENS tronquerait la
    # reponse au milieu d'une phrase.
    anthropic_max_tokens: int = 4096
    anthropic_effort: str = "low"
    generation_default: str = "groq"

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

    # Les acces Drive ne sont requis qu'a l'indexation. Les exiger ici
    # ferait echouer `serve`, `search` et `ask`, qui interrogent Qdrant par
    # le reseau et ne touchent jamais a Drive — c'est ce qui empechait le
    # conteneur de demarrer. La verification est reportee au moment ou le
    # corpus est reellement construit (voir pipeline.collect_pages).
    folder_id = _env("DRIVE_FOLDER_ID") or ""
    creds_raw = _env("GOOGLE_APPLICATION_CREDENTIALS") or ""

    if missing:
        raise ConfigError(
            "Variables d'environnement manquantes : "
            + ", ".join(missing)
            + ".\nCopiez .env.example en .env et renseignez-les."
        )

    # L'existence du fichier n'est pas verifiee ici : voir
    # `verifier_acces_corpus`, appelee au moment de construire le corpus.
    credentials_file = Path(creds_raw).expanduser() if creds_raw else Path()

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

    temperature = _env_float("GENERATION_TEMPERATURE", 0.1)
    temperature_reformulation = _env_float("REFORMULATION_TEMPERATURE", 0.4)
    for nom, valeur in (
        ("GENERATION_TEMPERATURE", temperature),
        ("REFORMULATION_TEMPERATURE", temperature_reformulation),
    ):
        if not 0.0 <= valeur <= 2.0:
            raise ConfigError(
                f"{nom}={valeur} hors bornes : l'API n'accepte qu'une valeur "
                "entre 0 et 2."
            )

    # Niveau d'effort de raisonnement d'Anthropic. Une valeur inconnue est
    # refusee par l'API : mieux vaut le dire ici que perdre l'appel.
    effort = _env("ANTHROPIC_EFFORT", "low")
    if effort not in EFFORTS_ANTHROPIC:
        raise ConfigError(
            f"ANTHROPIC_EFFORT={effort!r} inconnu "
            f"(valeurs possibles : {', '.join(sorted(EFFORTS_ANTHROPIC))})"
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
        generation_temperature=temperature,
        reformulation_enabled=_env_bool("REFORMULATION_ENABLED", True),
        reformulation_temperature=temperature_reformulation,
        reformulation_max_tokens=_env_int("REFORMULATION_MAX_TOKENS", 800),
        reformulation_model=_env("REFORMULATION_MODEL", "") or "",
        anthropic_api_key=_env("ANTHROPIC_API_KEY"),
        anthropic_model=_env("ANTHROPIC_MODEL", "claude-sonnet-5"),
        anthropic_max_tokens=_env_int("ANTHROPIC_MAX_TOKENS", 4096),
        anthropic_effort=effort,
        generation_default=_env("GENERATION_DEFAULT", "groq"),
    )


def verifier_acces_corpus(settings: Settings) -> None:
    """Verifie que la source du corpus est utilisable.

    Appelee au moment de construire le corpus, et non au chargement de la
    configuration : servir le chatbot ne demande ni Drive ni corpus local,
    et exiger ces reglages pour toute commande empechait le conteneur de
    demarrer.

    Raises:
        ConfigError: reglage manquant ou fichier introuvable.
    """
    if settings.corpus_source == "local":
        if not settings.local_corpus_dir.is_dir():
            raise ConfigError(
                f"Repertoire de corpus introuvable : {settings.local_corpus_dir}\n"
                "Verifiez LOCAL_CORPUS_DIR, ou lancez d'abord l'ingestion des PDF."
            )
        return

    manquants = [
        nom
        for nom, valeur in (
            ("DRIVE_FOLDER_ID", settings.drive_folder_id),
            ("GOOGLE_APPLICATION_CREDENTIALS", str(settings.google_credentials_file)),
        )
        if not valeur or valeur == "."
    ]
    if manquants:
        raise ConfigError(
            "Indexation depuis Drive : variables manquantes : "
            + ", ".join(manquants)
            + ".\nVoir docs/configuration-google-drive.md."
        )

    if not settings.google_credentials_file.is_file():
        raise ConfigError(
            f"Fichier de credentials Google introuvable : {settings.google_credentials_file}\n"
            "Voir la section 'Credentials Google Drive' du README."
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
