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
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from .chunking import Chunk
from .voyage_http import (
    DEFAULT_API_ROOT,
    PayloadTooLargeError,
    RateLimiter,
    RetryableAPIError,
    VoyageError,
    VoyageTransport,
)

logger = logging.getLogger(__name__)

# Reexportes : le reste du projet et les tests importent ces noms depuis ici.
EmbeddingError = VoyageError

__all__ = [
    "EmbeddedChunk",
    "EmbeddingError",
    "PayloadTooLargeError",
    "RateLimiter",
    "RetryableAPIError",
    "VoyageEmbedder",
    "MAX_INPUTS_PER_REQUEST",
    "MAX_CHUNKS_PER_REQUEST",
    "MAX_TOKENS_PER_REQUEST",
    "MAX_TOKENS_PER_DOCUMENT",
]

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

# Au-dela, l'incident n'est plus passager : inutile de defiler des
# milliers d'erreurs pendant des heures sans qu'aucun lot n'aboutisse.
MAX_ECHECS_CONSECUTIFS = 5


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """Un chunk et son vecteur."""

    chunk: Chunk
    embedding: list[float]


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
        self._transport = VoyageTransport(
            api_key=api_key,
            base_url=base_url,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            timeout=timeout,
        )
        self._max_tokens_per_minute = max_tokens_per_minute
        self._model = model
        self._dimension = dimension
        self._contextualized = contextualized
        self._timeout = timeout
        self._ratio = SAFETY_RATIO_EXACT if exact_tokenizer else SAFETY_RATIO_APPROX
        self._limiter = self._transport.limiter

        self.total_tokens_billed = 0
        # Un lot peut echouer sans condamner le job : on compte les
        # pertes pour les signaler a la fin.
        self.failed_batches = 0
        self.failed_chunks = 0
        self._echecs_consecutifs = 0

    # -- budgets -----------------------------------------------------------

    @property
    def _token_budget(self) -> int:
        """Taille maximale d'un lot, en tokens.

        Bornee par la limite d'un appel, mais aussi par le debit autorise
        par minute : un lot plus gros que le quota d'une minute serait
        rejete a chaque tentative, et aucun reessai ne le sauverait. Cela
        arrive sur les comptes sans moyen de paiement, plafonnes tres bas.

        La valeur est relue a chaque lot plutot que figee : un refus du
        serveur reduit le budget du limiteur, et les lots suivants doivent
        retrecir en consequence.
        """
        debit = self._transport.limiter.budget_tokens
        return int(min(MAX_TOKENS_PER_REQUEST, debit) * self._ratio)

    @property
    def _document_token_budget(self) -> int:
        return int(MAX_TOKENS_PER_DOCUMENT * self._ratio)

    # -- transport ---------------------------------------------------------

    def _post(self, endpoint: str, payload: dict) -> dict:
        """Delegue au transport partage.

        Reste une methode : les tests la remplacent pour eviter le reseau.
        """
        return self._transport.post(endpoint, payload)

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
                resultat = self._tenter(
                    self._embed_contextual_batch, batch, sum(len(g) for g in batch), index
                )
                if resultat:
                    yield resultat
        else:
            flat = [chunk for group in groups for chunk in group]
            for index, batch in enumerate(self._pack_standard(flat), start=1):
                tokens = sum(c.token_count for c in batch)
                self._limiter.acquire(tokens)
                logger.info("Lot %d : %d chunk(s), ~%d tokens", index, len(batch), tokens)
                resultat = self._tenter(self._embed_standard_batch, batch, len(batch), index)
                if resultat:
                    yield resultat

        if self.failed_chunks:
            logger.warning(
                "%d lot(s) abandonne(s), %d chunk(s) non plonge(s). "
                "Relancer la commande les reprendra : rien n'est perdu.",
                self.failed_batches,
                self.failed_chunks,
            )

    def _tenter(self, traitement, batch, nb_chunks: int, index: int) -> list[EmbeddedChunk]:
        """Traite un lot, en encaissant son echec eventuel.

        Un job de plusieurs heures traverse forcement un incident : veille
        de la machine, coupure reseau, quota refuse malgre les reessais.
        Faire echouer tout le job pour un lot serait absurde alors que
        l'index est incremental et que la reprise sait ou reprendre.

        Seuls les incidents **transitoires** sont encaisses. Une reponse
        incoherente ou une authentification refusee sont systematiques :
        les avaler ferait defiler des milliers d'erreurs pendant des heures
        sans qu'aucun lot n'aboutisse. Elles remontent donc, et le job
        s'arrete tout de suite.
        """
        try:
            resultat = traitement(batch)
        except RetryableAPIError as exc:
            self.failed_batches += 1
            self.failed_chunks += nb_chunks
            self._echecs_consecutifs += 1
            logger.error(
                "Lot %d abandonne (%d chunk(s)) : %s. Le job continue.",
                index,
                nb_chunks,
                exc,
            )
            if self._echecs_consecutifs >= MAX_ECHECS_CONSECUTIFS:
                raise EmbeddingError(
                    f"{self._echecs_consecutifs} lots consecutifs en echec : "
                    "le probleme n'est pas passager. Job interrompu, la reprise "
                    "repartira des chunks deja ecrits."
                ) from exc
            return []

        self._echecs_consecutifs = 0
        return resultat


# -- utilitaires de decodage des reponses ----------------------------------


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
