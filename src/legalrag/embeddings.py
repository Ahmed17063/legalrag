"""Appels a l'API d'embeddings Voyage, par lots, avec garde-fous.

Deux modes :

* `contextualized` (voyage-context-4) : on envoie les chunks d'un document
  groupes ensemble, et le modele produit un vecteur par chunk qui tient
  compte du reste du document. C'est le mode recommande ici : sur du code
  juridique, un chunk isole perd le titre de section ou l'article auquel
  il se rattache, et le contexte le lui rend.
* `standard` : un vecteur par texte, independamment (voyage-4-*, voyage-law-2).

Les resultats sont produits au fil de l'eau (`yield`) et non accumules :
le pipeline peut ainsi les persister lot par lot, ce qui rend la reprise
apres incident possible sans repayer les lots deja ecrits.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .chunking import Chunk

logger = logging.getLogger(__name__)

# URL par defaut. Le pipeline la remplace par celle deduite de la cle
# (voir config.resolve_base_url) : Atlas et l'ancienne plateforme Voyage
# n'exposent pas le meme domaine.
DEFAULT_API_ROOT = "https://api.voyageai.com/v1"

# Limites documentees par Voyage pour un appel. On applique en plus une
# marge (voir SAFETY_RATIO) car le comptage local peut differer legerement
# du comptage serveur.
MAX_INPUTS_PER_REQUEST = 1000
MAX_CHUNKS_PER_REQUEST = 16_000
MAX_TOKENS_PER_REQUEST = 120_000
MAX_TOKENS_PER_DOCUMENT = 32_000

# Marge appliquee aux budgets. Plus large quand le tokenizer local n'est
# qu'une approximation : mieux vaut un lot plus petit qu'un aller-retour
# rejete par le serveur.
SAFETY_RATIO_EXACT = 0.95
SAFETY_RATIO_APPROX = 0.75


class EmbeddingError(RuntimeError):
    """Echec d'appel a l'API d'embeddings."""


class RetryableAPIError(EmbeddingError):
    """Erreur transitoire : quota momentane, panne serveur, reseau."""


class PayloadTooLargeError(EmbeddingError):
    """Le lot a ete refuse pour cause de taille : il faut le scinder."""


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """Un chunk et son vecteur."""

    chunk: Chunk
    embedding: list[float]


class RateLimiter:
    """Limiteur a fenetre glissante sur les requetes et les tokens.

    Sert de garde-fou cote client : il vaut mieux lisser le debit que
    declencher des 429 en rafale et passer son temps en backoff.
    """

    def __init__(self, max_requests_per_minute: int, max_tokens_per_minute: int) -> None:
        self._max_requests = max_requests_per_minute
        self._max_tokens = max_tokens_per_minute
        self._requests: deque[float] = deque()
        self._tokens: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        horizon = now - 60.0
        while self._requests and self._requests[0] < horizon:
            self._requests.popleft()
        while self._tokens and self._tokens[0][0] < horizon:
            self._tokens.popleft()

    def acquire(self, tokens: int) -> None:
        """Bloque jusqu'a ce que le budget de la minute glissante le permette."""
        while True:
            with self._lock:
                now = time.monotonic()
                self._prune(now)

                used_tokens = sum(count for _, count in self._tokens)
                requests_ok = len(self._requests) < self._max_requests
                tokens_ok = used_tokens + tokens <= self._max_tokens or not self._tokens

                if requests_ok and tokens_ok:
                    self._requests.append(now)
                    self._tokens.append((now, tokens))
                    return

                oldest = min(
                    [self._requests[0]] if self._requests else [],
                    default=now,
                )
                if self._tokens:
                    oldest = min(oldest, self._tokens[0][0])
                wait = max(0.05, 60.0 - (now - oldest))

            logger.debug("Debit plafonne, attente de %.1fs", wait)
            time.sleep(wait)


class VoyageEmbedder:
    """Client HTTP minimal pour les endpoints d'embeddings Voyage."""

    def __init__(
        self,
        api_key: str,
        model: str,
        dimension: int,
        contextualized: bool,
        exact_tokenizer: bool,
        max_requests_per_minute: int = 200,
        max_tokens_per_minute: int = 800_000,
        timeout: float = 120.0,
        base_url: str = DEFAULT_API_ROOT,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._max_tokens_per_minute = max_tokens_per_minute
        self._model = model
        self._dimension = dimension
        self._contextualized = contextualized
        self._timeout = timeout
        self._ratio = SAFETY_RATIO_EXACT if exact_tokenizer else SAFETY_RATIO_APPROX
        self._limiter = RateLimiter(max_requests_per_minute, max_tokens_per_minute)

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

        self.total_tokens_billed = 0

    # -- budgets -----------------------------------------------------------

    @property
    def _token_budget(self) -> int:
        """Taille maximale d'un lot, en tokens.

        Bornee par la limite d'un appel, mais aussi par le debit autorise
        par minute : un lot plus gros que le quota d'une minute serait
        rejete a chaque tentative, et aucun reessai ne le sauverait. Cela
        arrive sur les comptes sans moyen de paiement, plafonnes tres bas.
        """
        return int(min(MAX_TOKENS_PER_REQUEST, self._max_tokens_per_minute) * self._ratio)

    @property
    def _document_token_budget(self) -> int:
        return int(MAX_TOKENS_PER_DOCUMENT * self._ratio)

    # -- transport ---------------------------------------------------------

    # Backoff calibre sur un compte tres plafonne : a 3 requetes par
    # minute, repartir au bout de 2 secondes ne fait que reconsommer le
    # quota. On attend donc d'emblee l'ordre de grandeur d'une fenetre.
    @retry(
        retry=retry_if_exception_type(RetryableAPIError),
        stop=stop_after_attempt(8),
        wait=wait_exponential_jitter(initial=20, max=120),
        reraise=True,
    )
    def _post(self, endpoint: str, payload: dict) -> dict:
        try:
            response = self._session.post(
                f"{self._base_url}/{endpoint}", json=payload, timeout=self._timeout
            )
        except requests.RequestException as exc:
            raise RetryableAPIError(f"erreur reseau : {exc}") from exc

        if response.status_code == 200:
            return response.json()

        body = response.text[:500]

        if response.status_code == 429:
            # Cas frequent : compte sans moyen de paiement, plafonne a
            # 3 requetes et 10 000 tokens par minute. Il faut alors
            # baisser MAX_REQUESTS_PER_MINUTE et MAX_TOKENS_PER_MINUTE
            # pour que le limiteur cote client respecte ce plafond.
            raise RetryableAPIError(
                f"debit refuse (429). Budget configure : "
                f"{self._max_tokens_per_minute} tokens/min. Reponse : {body}"
            )
        if response.status_code >= 500:
            raise RetryableAPIError(f"erreur serveur ({response.status_code}) : {body}")
        if response.status_code in (400, 413) and _mentions_size(body):
            raise PayloadTooLargeError(body)
        if response.status_code in (401, 403):
            raise EmbeddingError(
                f"authentification refusee ({response.status_code}). "
                "Verifiez VOYAGE_API_KEY. Detail : " + body
            )
        raise EmbeddingError(f"appel refuse ({response.status_code}) : {body}")

    # -- mode contextualise ------------------------------------------------

    def _split_oversized(self, group: Sequence[Chunk]) -> list[list[Chunk]]:
        """Scinde un document trop long pour partir d'un seul tenant.

        Deux plafonds s'appliquent : le contexte du modele, et ce qu'une
        requete peut transporter. Les chunks d'un document doivent voyager
        dans la meme liste pour partager leur contexte, donc cette liste ne
        peut pas depasser le budget d'une requete non plus. Sans cette
        borne, un document unique plus gros que le budget partait entier et
        se faisait refuser indefiniment.
        """
        budget = min(self._document_token_budget, self._token_budget)
        parts: list[list[Chunk]] = []
        current: list[Chunk] = []
        total = 0

        for chunk in group:
            cost = min(chunk.token_count, budget)
            if current and total + cost > budget:
                parts.append(current)
                current, total = [], 0
            current.append(chunk)
            total += cost

        if current:
            parts.append(current)
        return parts

    def _pack_contextual(self, groups: Sequence[Sequence[Chunk]]) -> Iterator[list[list[Chunk]]]:
        """Assemble des documents en requetes respectant les limites."""
        batch: list[list[Chunk]] = []
        batch_tokens = 0
        batch_chunks = 0

        for group in groups:
            for part in self._split_oversized(group):
                cost = sum(c.token_count for c in part)
                too_many_inputs = len(batch) + 1 > MAX_INPUTS_PER_REQUEST
                too_many_chunks = batch_chunks + len(part) > MAX_CHUNKS_PER_REQUEST
                too_many_tokens = batch_tokens + cost > self._token_budget

                if batch and (too_many_inputs or too_many_chunks or too_many_tokens):
                    yield batch
                    batch, batch_tokens, batch_chunks = [], 0, 0

                batch.append(list(part))
                batch_tokens += cost
                batch_chunks += len(part)

        if batch:
            yield batch

    def _embed_contextual_batch(self, batch: list[list[Chunk]]) -> list[EmbeddedChunk]:
        payload = {
            "inputs": [[chunk.text for chunk in group] for group in batch],
            "model": self._model,
            "input_type": "document",
            "output_dimension": self._dimension,
        }

        try:
            data = self._post("contextualizedembeddings", payload)
        except PayloadTooLargeError:
            if len(batch) == 1 and len(batch[0]) == 1:
                raise
            middle = max(1, len(batch) // 2)
            logger.warning("Lot refuse pour sa taille, scission en deux (%d documents)", len(batch))
            left = self._embed_contextual_batch(batch[:middle])
            right = self._embed_contextual_batch(batch[middle:])
            return left + right

        self.total_tokens_billed += int(data.get("usage", {}).get("total_tokens", 0))
        return _pair_contextual(batch, data)

    # -- mode standard -----------------------------------------------------

    def _pack_standard(self, chunks: Sequence[Chunk]) -> Iterator[list[Chunk]]:
        batch: list[Chunk] = []
        batch_tokens = 0

        for chunk in chunks:
            cost = chunk.token_count
            if batch and (
                len(batch) + 1 > MAX_INPUTS_PER_REQUEST
                or batch_tokens + cost > self._token_budget
            ):
                yield batch
                batch, batch_tokens = [], 0
            batch.append(chunk)
            batch_tokens += cost

        if batch:
            yield batch

    def _embed_standard_batch(self, batch: list[Chunk]) -> list[EmbeddedChunk]:
        payload = {
            "input": [chunk.text for chunk in batch],
            "model": self._model,
            "input_type": "document",
            "output_dimension": self._dimension,
        }

        try:
            data = self._post("embeddings", payload)
        except PayloadTooLargeError:
            if len(batch) == 1:
                raise
            middle = len(batch) // 2
            logger.warning("Lot refuse pour sa taille, scission en deux (%d textes)", len(batch))
            return self._embed_standard_batch(batch[:middle]) + self._embed_standard_batch(
                batch[middle:]
            )

        self.total_tokens_billed += int(data.get("usage", {}).get("total_tokens", 0))
        return _pair_standard(batch, data)

    # -- API publique ------------------------------------------------------

    def embed(self, groups: Sequence[Sequence[Chunk]]) -> Iterator[list[EmbeddedChunk]]:
        """Plonge les chunks, lot par lot.

        `groups` est une liste de documents, chacun etant sa liste de chunks
        dans l'ordre. En mode standard le regroupement est ignore.

        Yields:
            Les lots de resultats, dans l'ordre, au fur et a mesure.
        """
        if self._contextualized:
            batches = self._pack_contextual(groups)
            for index, batch in enumerate(batches, start=1):
                tokens = sum(c.token_count for group in batch for c in group)
                self._limiter.acquire(tokens)
                logger.info(
                    "Lot %d : %d document(s), %d chunk(s), ~%d tokens",
                    index,
                    len(batch),
                    sum(len(g) for g in batch),
                    tokens,
                )
                yield self._embed_contextual_batch(batch)
        else:
            flat = [chunk for group in groups for chunk in group]
            for index, batch in enumerate(self._pack_standard(flat), start=1):
                tokens = sum(c.token_count for c in batch)
                self._limiter.acquire(tokens)
                logger.info("Lot %d : %d chunk(s), ~%d tokens", index, len(batch), tokens)
                yield self._embed_standard_batch(batch)


# -- utilitaires de decodage des reponses ----------------------------------


def _mentions_size(body: str) -> bool:
    lowered = body.lower()
    return any(word in lowered for word in ("token", "too large", "limit", "exceed"))


def _pair_standard(batch: Sequence[Chunk], data: dict) -> list[EmbeddedChunk]:
    items = data.get("data") or []
    if len(items) != len(batch):
        raise EmbeddingError(
            f"reponse incoherente : {len(items)} vecteurs pour {len(batch)} textes"
        )
    ordered = sorted(items, key=lambda item: item.get("index", 0))
    return [
        EmbeddedChunk(chunk=chunk, embedding=item["embedding"])
        for chunk, item in zip(batch, ordered)
    ]


def _pair_contextual(batch: Sequence[Sequence[Chunk]], data: dict) -> list[EmbeddedChunk]:
    outer = data.get("data") or []
    if len(outer) != len(batch):
        raise EmbeddingError(
            f"reponse incoherente : {len(outer)} documents pour {len(batch)} envoyes"
        )

    paired: list[EmbeddedChunk] = []
    for group, entry in zip(batch, sorted(outer, key=lambda item: item.get("index", 0))):
        inner = entry.get("data") or []
        if len(inner) != len(group):
            raise EmbeddingError(
                f"reponse incoherente : {len(inner)} vecteurs pour {len(group)} chunks"
            )
        ordered = sorted(inner, key=lambda item: item.get("index", 0))
        paired.extend(
            EmbeddedChunk(chunk=chunk, embedding=item["embedding"])
            for chunk, item in zip(group, ordered)
        )
    return paired
