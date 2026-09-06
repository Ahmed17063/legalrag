"""Comptage de tokens avec correspondance vers les positions de caracteres.

Le chunker a besoin de deux choses : compter les tokens (pour respecter la
taille de chunk et les limites de l'API) et savoir a quel endroit du texte
chaque token commence (pour decouper au bon caractere).

Le tokenizer exact de Voyage est utilise s'il est disponible. Sinon on
retombe sur une approximation : le job continue de tourner, mais les
budgets d'appel sont appliques avec une marge de securite.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol

logger = logging.getLogger(__name__)

# Un "mot" au sens large : suite alphanumerique, ou signe isole.
_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# Nombre moyen de caracteres par token sur du francais juridique.
_CHARS_PER_TOKEN = 4


class Tokenizer(Protocol):
    """Interface minimale attendue par le chunker."""

    name: str
    exact: bool

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Spans `(debut, fin)` en caracteres, un par token."""

    def count(self, text: str) -> int:
        """Nombre de tokens du texte."""


class HeuristicTokenizer:
    """Approximation locale, sans dependance ni appel reseau.

    Chaque mot est decoupe en pseudo-tokens de ~4 caracteres, ce qui donne
    a la fois un comptage plausible et des positions exploitables.
    """

    name = "heuristique"
    exact = False

    def offsets(self, text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        for match in _WORD_RE.finditer(text):
            start, end = match.span()
            length = end - start
            pieces = max(1, -(-length // _CHARS_PER_TOKEN))  # division entiere par exces
            if pieces == 1:
                spans.append((start, end))
                continue
            step = -(-length // pieces)
            for offset in range(0, length, step):
                spans.append((start + offset, min(start + offset + step, end)))
        return spans

    def count(self, text: str) -> int:
        return len(self.offsets(text))


class VoyageTokenizer:
    """Tokenizer exact expose par le SDK `voyageai`."""

    exact = True

    def __init__(self, model: str) -> None:
        import voyageai  # import tardif : dependance optionnelle

        self.name = f"voyage:{model}"
        self._model = model
        self._client = voyageai.Client()

    def offsets(self, text: str) -> list[tuple[int, int]]:
        encoding = self._client.tokenize([text], model=self._model)[0]
        # `tokenizers.Encoding.offsets` donne les spans caracteres des tokens.
        return [(int(a), int(b)) for a, b in encoding.offsets]

    def count(self, text: str) -> int:
        return len(self.offsets(text))


def get_tokenizer(model: str) -> Tokenizer:
    """Renvoie le tokenizer exact si possible, sinon l'approximation."""
    try:
        tokenizer = VoyageTokenizer(model)
        # Verification a chaud : le SDK peut etre installe mais l'API du
        # tokenizer avoir change de forme. Mieux vaut le decouvrir ici que
        # au milieu du job.
        tokenizer.offsets("Article 1er du Code civil.")
    except Exception as exc:  # noqa: BLE001 - on veut degrader, pas echouer
        logger.warning(
            "Tokenizer exact Voyage indisponible (%s) : repli sur l'approximation. "
            "Les tailles de chunk et les budgets d'appel seront appliques avec une marge.",
            exc,
        )
        return HeuristicTokenizer()

    logger.info("Tokenizer exact actif (%s)", tokenizer.name)
    return tokenizer
