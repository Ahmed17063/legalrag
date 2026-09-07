"""Generation de la reponse a partir des passages retrouves.

Le dernier maillon du RAG : les passages classes par le retrieval sont
soumis a un modele de langage qui redige une reponse.

C'est aussi le maillon le plus risque sur du juridique. Un modele qui
reformule approximativement une disposition legale, ou qui comble un trou
en inventant, produit quelque chose de pire qu'inutile : une reponse
plausible et fausse. Trois garde-fous, tous verifiables :

1. **Passages numerotes, citation obligatoire.** Le modele doit rattacher
   chaque affirmation a un numero de passage.
2. **Temperature nulle.** Aucune raison de vouloir de la variete quand on
   restitue du droit.
3. **Verification apres coup.** Les numeros cites sont controles : hors
   plage, ils sont signales. Une reponse sans aucune citation est marquee
   comme non etayee, et l'appelant peut la refuser.

Aucun de ces garde-fous ne rend le modele infaillible. La reponse est
donc toujours rendue **avec** les passages qui l'ont nourrie, pour que la
verification humaine reste possible.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .config import Settings
from .retrieval import Passage

logger = logging.getLogger(__name__)

# Groq expose une API compatible OpenAI.
GROQ_API_ROOT = "https://api.groq.com/openai/v1"

# Formule exacte attendue quand la reponse n'est pas dans le corpus. Elle
# est verifiee cote code : mieux vaut un aveu d'ignorance qu'une invention.
AUCUNE_REPONSE = "La réponse ne figure pas dans les documents fournis."

SYSTEM_PROMPT = f"""Tu es un assistant juridique. Tu reponds UNIQUEMENT a partir des extraits de textes de loi qui te sont fournis.

Regles absolues :
1. N'utilise QUE les extraits fournis. N'ajoute aucune connaissance exterieure, meme si tu penses la connaitre.
2. Cite tes sources avec le numero de l'extrait entre crochets, par exemple [1] ou [2]. Chaque affirmation doit porter sa citation.
3. Si les extraits ne permettent pas de repondre, ecris exactement : {AUCUNE_REPONSE}
4. Ne reformule pas une disposition legale de facon approximative. Cite le texte quand la formulation exacte compte.
5. Reponds en francais, de maniere concise et precise.
6. Si les extraits se contredisent ou sont ambigus, dis-le plutot que de trancher.
7. Precise toujours de quel ordre juridique releve chaque texte cite : droit
   mauritanien, droit de l'Union europeenne, ou convention internationale.
8. Un texte europeen ne s'applique pas de plein droit en Mauritanie. Si tu ne
   disposes que d'extraits europeens ou internationaux pour repondre, dis-le
   explicitement et signale que le droit mauritanien applicable n'a pas ete
   retrouve, au lieu de presenter la regle etrangere comme le droit en vigueur.
"""


class GenerationError(RuntimeError):
    """Echec d'appel a l'API de generation."""


class RetryableGenerationError(GenerationError):
    """Erreur transitoire : quota momentane, panne serveur, reseau."""


@dataclass
class Answer:
    """Une reponse redigee, et de quoi la verifier."""

    question: str
    text: str
    passages: list[Passage] = field(default_factory=list)
    cited: tuple[int, ...] = ()
    invalid_citations: tuple[int, ...] = ()
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def grounded(self) -> bool:
        """La reponse s'appuie-t-elle sur au moins un passage reel ?"""
        return bool(self.cited) and not self.invalid_citations

    @property
    def declined(self) -> bool:
        """Le modele a-t-il reconnu ne pas trouver la reponse ?"""
        return AUCUNE_REPONSE.lower() in self.text.lower()

    def sources(self) -> list[str]:
        """Citations des passages effectivement invoques."""
        return [
            f"[{n}] {self.passages[n - 1].citation()}"
            for n in self.cited
            if 1 <= n <= len(self.passages)
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.text,
            "grounded": self.grounded,
            "declined": self.declined,
            "cited": list(self.cited),
            "invalid_citations": list(self.invalid_citations),
            "sources": self.sources(),
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "passages": [p.as_dict() for p in self.passages],
        }


def build_context(passages: list[Passage]) -> str:
    """Met en forme les passages numerotes soumis au modele."""
    blocs = []
    for i, passage in enumerate(passages, 1):
        texte = " ".join(passage.text.split())
        blocs.append(f"[{i}] {passage.citation()}\n{texte}")
    return "\n\n".join(blocs)


def extract_citations(text: str, nb_passages: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Numeros cites, separes en valides et hors plage.

    Un modele peut citer [7] alors que seuls six passages ont ete fournis.
    C'est le signe d'une reponse qui derive : on le remonte.
    """
    numeros = [int(n) for n in re.findall(r"\[(\d{1,2})\]", text)]
    valides = sorted({n for n in numeros if 1 <= n <= nb_passages})
    invalides = sorted({n for n in numeros if not 1 <= n <= nb_passages})
    return tuple(valides), tuple(invalides)


class Generator:
    """Redige une reponse a partir des passages, via l'API Groq."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        if not settings.groq_api_key:
            raise GenerationError(
                "GROQ_API_KEY absente. Renseignez-la dans .env, ou lancez la "
                "recherche sans generation (commande `search`)."
            )

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {settings.groq_api_key}",
                "Content-Type": "application/json",
            }
        )

    @retry(
        retry=retry_if_exception_type(RetryableGenerationError),
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=2, max=30),
        reraise=True,
    )
    def _post(self, payload: dict) -> dict:
        try:
            response = self._session.post(
                f"{GROQ_API_ROOT}/chat/completions", json=payload, timeout=120
            )
        except requests.RequestException as exc:
            raise RetryableGenerationError(f"erreur reseau : {exc}") from exc

        if response.status_code == 200:
            return response.json()

        corps = response.text[:500]
        if response.status_code == 429:
            raise RetryableGenerationError(f"quota atteint (429) : {corps}")
        if response.status_code >= 500:
            raise RetryableGenerationError(f"erreur serveur ({response.status_code}) : {corps}")
        if response.status_code in (401, 403):
            raise GenerationError(
                f"authentification refusee ({response.status_code}). "
                "Verifiez GROQ_API_KEY. Detail : " + corps
            )
        if response.status_code == 404:
            raise GenerationError(
                f"modele {self._settings.generation_model!r} introuvable. "
                "Groq a peut-etre depreciee ce modele : consultez "
                "https://console.groq.com/docs/deprecations. Detail : " + corps
            )
        raise GenerationError(f"appel refuse ({response.status_code}) : {corps}")

    def answer(self, question: str, passages: list[Passage]) -> Answer:
        """Redige une reponse etayee par les passages fournis."""
        if not passages:
            return Answer(
                question=question,
                text=AUCUNE_REPONSE,
                model=self._settings.generation_model,
            )

        payload = {
            "model": self._settings.generation_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Extraits de textes de loi :\n\n{build_context(passages)}\n\n"
                        f"Question : {question}"
                    ),
                },
            ],
            # Aucune raison de vouloir de la variete quand on restitue du droit.
            "temperature": 0,
            "max_tokens": self._settings.generation_max_tokens,
        }

        data = self._post(payload)
        texte = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})

        valides, invalides = extract_citations(texte, len(passages))

        reponse = Answer(
            question=question,
            text=texte,
            passages=passages,
            cited=valides,
            invalid_citations=invalides,
            model=data.get("model", self._settings.generation_model),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )

        if invalides:
            logger.warning(
                "Le modele cite des extraits inexistants : %s (seulement %d fournis). "
                "Reponse a verifier.",
                invalides,
                len(passages),
            )
        elif not valides and not reponse.declined:
            logger.warning(
                "Reponse sans aucune citation : rien ne la rattache aux extraits. "
                "A verifier avant usage."
            )

        return reponse
