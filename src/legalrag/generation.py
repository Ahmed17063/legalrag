"""Generation de la reponse a partir des passages retrouves.

Le dernier maillon du RAG : les passages classes par le retrieval sont
soumis a un modele de langage qui redige une reponse.

C'est aussi le maillon le plus risque sur du juridique. Un modele qui
reformule approximativement une disposition legale, ou qui comble un trou
en inventant, produit quelque chose de pire qu'inutile : une reponse
plausible et fausse. Trois garde-fous, tous verifiables :

1. **Passages numerotes, citation obligatoire.** Le modele doit rattacher
   chaque affirmation a un numero de passage.
2. **Echantillonnage bride.** On ne cherche pas de la variete quand on
   restitue du droit. `GENERATION_TEMPERATURE` vaut 0,3 par defaut : assez
   bas pour rester fidele au texte, assez haut pour eviter les boucles et
   les formulations figees que produit parfois une temperature nulle. Voir
   la reserve sur Claude plus bas.
3. **Verification apres coup.** Les numeros cites sont controles : hors
   plage, ils sont signales. Une reponse sans aucune citation est marquee
   comme non etayee, et l'appelant peut la refuser.

Aucun de ces garde-fous ne rend le modele infaillible. La reponse est
donc toujours rendue **avec** les passages qui l'ont nourrie, pour que la
verification humaine reste possible.

Deux fournisseurs
-----------------

Le module en expose deux, choisis par l'appelant appel par appel :

* **Groq**, API compatible OpenAI, appelee en HTTP direct — rapide et
  gratuite dans les limites du palier d'essai ;
* **Claude** (Anthropic), via le SDK officiel — plus lent, mais il
  raisonne avant de repondre, ce qui aide quand les extraits se
  contredisent ou melangent plusieurs ordres juridiques.

Une reserve sur le garde-fou n° 2 : Claude Sonnet 5 **refuse** tout
parametre d'echantillonnage dont la valeur n'est pas celle par defaut —
`temperature=0` comme `temperature=0.3`, l'appel est rejete en 400 dans
les deux cas. `GENERATION_TEMPERATURE` ne s'applique donc qu'a Groq, et
aucune temperature n'est transmise a Claude. La consigne systeme, elle,
reste identique pour les deux.
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
9. Un extrait dont la reference porte la mention [ABROGE] ne doit JAMAIS etre
   presente comme le droit applicable. Tu peux l'exposer si la question porte
   sur le droit anterieur, en indiquant clairement qu'il est abroge.
10. Un extrait portant [statut non verifie] doit etre cite avec cette reserve :
   son maintien en vigueur n'a pas ete etabli.
"""


class GenerationError(RuntimeError):
    """Echec d'appel a l'API de generation."""


class RetryableGenerationError(GenerationError):
    """Erreur transitoire : quota momentane, panne serveur, reseau."""


@dataclass(frozen=True, slots=True)
class ModeleGeneration:
    """Un modele proposable a l'utilisateur.

    `cle` est l'identifiant stable echange avec le navigateur et la ligne
    de commande ; `modele_api` est ce que le fournisseur attend, et peut
    changer par variable d'environnement sans casser les appelants.
    """

    cle: str
    libelle: str
    fournisseur: str
    modele_api: str
    description: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "cle": self.cle,
            "libelle": self.libelle,
            "fournisseur": self.fournisseur,
            "modele": self.modele_api,
            "description": self.description,
        }


def modeles_disponibles(settings: Settings) -> dict[str, ModeleGeneration]:
    """Modeles reellement utilisables, c'est-a-dire ceux dont la cle existe.

    Proposer dans l'interface un modele sans cle d'API ne menerait qu'a une
    erreur au premier appel : mieux vaut ne pas l'afficher.
    """
    modeles: dict[str, ModeleGeneration] = {}

    if settings.groq_api_key:
        modeles["groq"] = ModeleGeneration(
            cle="groq",
            libelle=f"Groq — {settings.generation_model}",
            fournisseur="groq",
            modele_api=settings.generation_model,
            description="Rapide. Repond directement, sans etape de raisonnement.",
        )

    if settings.anthropic_api_key:
        modeles["claude"] = ModeleGeneration(
            cle="claude",
            libelle=f"Claude — {settings.anthropic_model}",
            fournisseur="anthropic",
            modele_api=settings.anthropic_model,
            description=(
                "Plus lent. Raisonne avant de repondre : utile quand les "
                "extraits se contredisent ou melangent plusieurs ordres juridiques."
            ),
        )

    return modeles


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
    fournisseur: str = ""
    # Le modele a-t-il ete coupe par le plafond de sortie ? Une reponse
    # juridique tronquee au milieu d'une phrase doit se voir.
    tronquee: bool = False

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
            "fournisseur": self.fournisseur,
            "tronquee": self.tronquee,
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


@dataclass(frozen=True, slots=True)
class _Brut:
    """Ce que renvoie un fournisseur, avant analyse des citations.

    Les deux API ne se ressemblent pas — Groq nomme `prompt_tokens` ce
    qu'Anthropic appelle `input_tokens`, et decrit l'arret par
    `finish_reason` la ou l'autre parle de `stop_reason`. Cette structure
    absorbe l'ecart pour que `answer` n'ait plus qu'un seul cas a traiter.
    """

    texte: str
    modele: str
    prompt_tokens: int
    completion_tokens: int
    tronquee: bool


class Generator:
    """Redige une reponse a partir des passages, via Groq ou Claude."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._modeles = modeles_disponibles(settings)
        if not self._modeles:
            raise GenerationError(
                "Aucune cle de generation. Renseignez GROQ_API_KEY ou "
                "ANTHROPIC_API_KEY dans .env, ou lancez la recherche sans "
                "generation (commande `search`)."
            )

        defaut = settings.generation_default
        if defaut not in self._modeles:
            # Le defaut configure n'a pas de cle : on prend le premier
            # utilisable plutot que d'echouer au premier appel.
            defaut = next(iter(self._modeles))
            if settings.generation_default in {"groq", "claude"}:
                logger.info(
                    "GENERATION_DEFAULT=%r sans cle d'API : %r pris par defaut.",
                    settings.generation_default,
                    defaut,
                )
        self._defaut = defaut

        self._session: requests.Session | None = None
        self._client_anthropic: Any = None

    # -- Modeles -----------------------------------------------------------

    def modeles(self) -> list[ModeleGeneration]:
        """Modeles proposables, le defaut en tete."""
        ordre = sorted(self._modeles.values(), key=lambda m: m.cle != self._defaut)
        return ordre

    @property
    def defaut(self) -> str:
        return self._defaut

    def _resoudre(self, cle: str | None) -> ModeleGeneration:
        if not cle:
            return self._modeles[self._defaut]
        modele = self._modeles.get(cle)
        if modele is None:
            raise GenerationError(
                f"Modele {cle!r} indisponible. Modeles configures : "
                + ", ".join(sorted(self._modeles))
                + "."
            )
        return modele

    # -- Groq : API compatible OpenAI, en HTTP direct ----------------------

    def _session_groq(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(
                {
                    "Authorization": f"Bearer {self._settings.groq_api_key}",
                    "Content-Type": "application/json",
                }
            )
        return self._session

    @retry(
        retry=retry_if_exception_type(RetryableGenerationError),
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=2, max=30),
        reraise=True,
    )
    def _post(self, payload: dict) -> dict:
        try:
            response = self._session_groq().post(
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

    def _appeler_groq(self, modele: ModeleGeneration, invite: str) -> _Brut:
        payload = {
            "model": modele.modele_api,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": invite},
            ],
            "temperature": self._settings.generation_temperature,
            "max_tokens": self._settings.generation_max_tokens,
        }

        data = self._post(payload)
        choix = data["choices"][0]
        usage = data.get("usage", {})

        return _Brut(
            texte=choix["message"]["content"].strip(),
            modele=data.get("model", modele.modele_api),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            tronquee=choix.get("finish_reason") == "length",
        )

    # -- Claude : SDK officiel Anthropic -----------------------------------

    def _client(self) -> Any:
        """Client Anthropic, construit au premier appel.

        L'import est differe : une installation qui n'utilise que Groq n'a
        pas a charger le SDK, et le serveur demarre sans lui.
        """
        if self._client_anthropic is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depend de l'install
                raise GenerationError(
                    "Le paquet `anthropic` est absent alors qu'ANTHROPIC_API_KEY "
                    "est renseignee. Installez-le (`uv sync`) ou retirez la cle "
                    "pour n'utiliser que Groq."
                ) from exc

            self._client_anthropic = anthropic.Anthropic(
                api_key=self._settings.anthropic_api_key,
                # Le SDK reessaie seul les 429, 5xx et coupures reseau, avec
                # un repli exponentiel. Inutile de superposer tenacity ici.
                max_retries=4,
                timeout=180.0,
            )
        return self._client_anthropic

    def _appeler_claude(self, modele: ModeleGeneration, invite: str) -> _Brut:
        import anthropic

        client = self._client()
        try:
            message = client.messages.create(
                model=modele.modele_api,
                max_tokens=self._settings.anthropic_max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": invite}],
                # Raisonnement adaptatif : actif par defaut sur Sonnet 5, et
                # c'est voulu — les extraits se contredisent souvent. L'effort
                # reste bas, la tache etant de citer, non de resoudre.
                output_config={"effort": self._settings.anthropic_effort},
                # Pas de `temperature` : Sonnet 5 rejette en 400 tout parametre
                # d'echantillonnage dont la valeur n'est pas celle par defaut.
                # GENERATION_TEMPERATURE ne vaut donc que pour Groq — envoyer
                # 0.3 ici ferait echouer chaque appel. Voir l'entete du module.
            )
        except anthropic.AuthenticationError as exc:
            raise GenerationError(
                "Authentification Anthropic refusee. Verifiez ANTHROPIC_API_KEY. "
                f"Detail : {exc}"
            ) from exc
        except anthropic.NotFoundError as exc:
            raise GenerationError(
                f"Modele {modele.modele_api!r} introuvable chez Anthropic. "
                "Verifiez ANTHROPIC_MODEL. Detail : " + str(exc)
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise RetryableGenerationError(f"erreur reseau : {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise GenerationError(
                f"appel refuse ({exc.status_code}) : {exc}"
            ) from exc

        # `content` est une liste de blocs typés : avec le raisonnement
        # adaptatif, les blocs `thinking` y figurent aussi. Seul le texte
        # nous interesse.
        texte = "".join(
            bloc.text for bloc in message.content if bloc.type == "text"
        ).strip()

        if message.stop_reason == "refusal":
            raise GenerationError(
                "Le modele a refuse de repondre a cette question."
            )

        return _Brut(
            texte=texte,
            modele=message.model,
            prompt_tokens=message.usage.input_tokens,
            completion_tokens=message.usage.output_tokens,
            tronquee=message.stop_reason == "max_tokens",
        )

    # -- Point d'entree ----------------------------------------------------

    def answer(
        self, question: str, passages: list[Passage], modele: str | None = None
    ) -> Answer:
        """Redige une reponse etayee par les passages fournis.

        `modele` est la cle d'un modele configure (`groq`, `claude`) ; sans
        elle, le modele par defaut est utilise.
        """
        choisi = self._resoudre(modele)

        if not passages:
            return Answer(
                question=question,
                text=AUCUNE_REPONSE,
                model=choisi.modele_api,
                fournisseur=choisi.fournisseur,
            )

        invite = (
            f"Extraits de textes de loi :\n\n{build_context(passages)}\n\n"
            f"Question : {question}"
        )

        if choisi.fournisseur == "anthropic":
            brut = self._appeler_claude(choisi, invite)
        else:
            brut = self._appeler_groq(choisi, invite)

        valides, invalides = extract_citations(brut.texte, len(passages))

        reponse = Answer(
            question=question,
            text=brut.texte,
            passages=passages,
            cited=valides,
            invalid_citations=invalides,
            model=brut.modele,
            fournisseur=choisi.fournisseur,
            prompt_tokens=brut.prompt_tokens,
            completion_tokens=brut.completion_tokens,
            tronquee=brut.tronquee,
        )

        if brut.tronquee:
            logger.warning(
                "Reponse coupee par le plafond de sortie (%s). Elle peut "
                "s'arreter au milieu d'une phrase.",
                choisi.cle,
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
