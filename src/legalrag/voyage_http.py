"""Transport HTTP partage vers l'API Voyage.

Extrait de `embeddings` pour que l'indexation et la recherche partagent la
meme gestion des erreurs, des reessais et du debit. Les plafonds sont bas
sur un compte sans moyen de paiement (3 requetes et 10 000 tokens par
minute) et concernent aussi bien les embeddings que le reranking.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)

DEFAULT_API_ROOT = "https://api.voyageai.com/v1"


class VoyageError(RuntimeError):
    """Echec d'appel a l'API Voyage."""


class RetryableAPIError(VoyageError):
    """Erreur transitoire : quota momentane, panne serveur, reseau."""


class PayloadTooLargeError(VoyageError):
    """Charge refusee pour sa taille : il faut la scinder."""


class RateLimiter:
    """Limiteur a fenetre glissante sur les requetes et les tokens.

    Garde-fou cote client : il vaut mieux lisser le debit que declencher
    des 429 en rafale et passer son temps en backoff.
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

                oldest = min([self._requests[0]] if self._requests else [], default=now)
                if self._tokens:
                    oldest = min(oldest, self._tokens[0][0])
                wait = max(0.05, 60.0 - (now - oldest))

            logger.debug("Debit plafonne, attente de %.1fs", wait)
            time.sleep(wait)


def _mentions_size(body: str) -> bool:
    lowered = body.lower()
    return any(mot in lowered for mot in ("token", "too large", "limit", "exceed"))


class VoyageTransport:
    """Session HTTP vers l'API Voyage, avec reessais et limitation de debit."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_API_ROOT,
        max_requests_per_minute: int = 200,
        max_tokens_per_minute: int = 800_000,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_tokens_per_minute = max_tokens_per_minute
        self._timeout = timeout
        self.limiter = RateLimiter(max_requests_per_minute, max_tokens_per_minute)

        self._session = requests.Session()
        self._session.headers.update(
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        )

    # Backoff calibre sur un compte tres plafonne : a 3 requetes par minute,
    # repartir au bout de 2 secondes ne fait que reconsommer le quota. On
    # attend d'emblee l'ordre de grandeur d'une fenetre.
    @retry(
        retry=retry_if_exception_type(RetryableAPIError),
        stop=stop_after_attempt(8),
        wait=wait_exponential_jitter(initial=20, max=120),
        reraise=True,
    )
    def post(self, endpoint: str, payload: dict) -> dict:
        try:
            response = self._session.post(
                f"{self.base_url}/{endpoint}", json=payload, timeout=self._timeout
            )
        except requests.RequestException as exc:
            raise RetryableAPIError(f"erreur reseau : {exc}") from exc

        if response.status_code == 200:
            return response.json()

        body = response.text[:500]

        if response.status_code == 429:
            raise RetryableAPIError(
                f"debit refuse (429). Budget configure : "
                f"{self.max_tokens_per_minute} tokens/min. Reponse : {body}"
            )
        if response.status_code >= 500:
            raise RetryableAPIError(f"erreur serveur ({response.status_code}) : {body}")
        if response.status_code in (400, 413) and _mentions_size(body):
            raise PayloadTooLargeError(body)
        if response.status_code in (401, 403):
            raise VoyageError(
                f"authentification refusee ({response.status_code}). "
                "Verifiez VOYAGE_API_KEY. Detail : " + body
            )
        raise VoyageError(f"appel refuse ({response.status_code}) : {body}")
