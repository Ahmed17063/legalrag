"""Reformulation de la question avant la recherche vectorielle.

Le probleme : l'utilisateur ecrit « est-ce qu'on peut licencier une femme
enceinte ? », le code ecrit « la rupture du contrat de travail ne peut
etre notifiee pendant les periodes de suspension liees a la maternite ».
Les deux parlent de la meme chose, mais le vocabulaire ne se recouvre pas
— et c'est le vocabulaire que compare un modele d'embedding.

La parade classique est de reformuler la question dans la langue du
corpus avant de l'envoyer chercher. Trois choix ont ete faits ici, tous
dictes par des contraintes reelles du projet :

**Une seule reformulation, pas plusieurs.** La technique courante produit
N variantes et cherche avec chacune. Impossible ici : le compte Voyage est
plafonne a deux requetes par minute, et une recherche en consomme deja
deux (un embedding, un reranking). Une variante de plus, et chaque
question attendrait une minute supplementaire. La reformulation est donc
unique, et elle *conserve* les termes de l'utilisateur au lieu de les
remplacer : on elargit le filet sans changer de riviere.

**La recherche vectorielle utilise la reformulation, le reranking la
question d'origine.** Le premier cherche du vocabulaire, le second juge la
pertinence : lui donner la question telle qu'elle a ete posee le garde
aligne sur l'intention reelle.

**L'echec est sans consequence.** Si le modele est indisponible, lent ou
repond n'importe quoi, la question d'origine part telle quelle. Une
reformulation ratee ne doit jamais empecher de repondre.

La temperature y est plus haute qu'a la redaction (0,4 contre 0,1) : on
cherche ici de la variete lexicale, pas de la fidelite au texte. C'est
exactement l'inverse du besoin de l'etape suivante.
"""

from __future__ import annotations

import logging
import re

from .config import Settings
from .generation import GenerationError, Generator

logger = logging.getLogger(__name__)

# Bornes de bon sens. Une reformulation beaucoup plus longue que la
# question signale un modele qui a commence a repondre au lieu de
# reformuler ; beaucoup plus courte, qu'il a tronque.
LONGUEUR_MAX = 400
RAPPORT_MAX = 6

CONSIGNE = """Tu prepares une recherche documentaire dans un corpus de textes de loi mauritaniens rediges en francais juridique.

On te donne la question d'un utilisateur, souvent formulee en langage courant. Tu la reecris de facon a maximiser les chances de retrouver les articles pertinents par similarite semantique.

Regles :
1. Conserve tous les termes importants de la question d'origine. Tu ajoutes du vocabulaire, tu n'en retires pas.
2. Ajoute les termes juridiques que le texte de loi emploierait probablement : le nom de l'institution, la qualification juridique de l'acte, les synonymes du code.
3. Si la question designe une notion par un mot courant, ajoute son equivalent juridique (licencier -> rupture du contrat de travail, licenciement ; voler -> soustraction frauduleuse, vol).
4. Ne reponds PAS a la question. Ne fais aucune supposition sur ce que dit la loi.
5. N'invente aucun numero d'article, aucune date, aucune reference de texte.
6. Ecris une seule phrase ou groupe nominal, en francais, sans guillemets, sans preambule, sans liste.
7. Si la question est deja formulee en termes juridiques precis, renvoie-la presque inchangee.

Tu renvoies uniquement la requete reecrite, rien d'autre."""


def _nettoyer(brut: str) -> str:
    """Retire ce qu'un modele ajoute malgre la consigne."""
    texte = brut.strip()

    # Certains modeles annoncent leur reponse ; d'autres l'encadrent.
    texte = re.sub(
        r"^(requ[eê]te\s+r[eé][eé]crite|reformulation|r[eé][eé]criture)\s*:\s*",
        "",
        texte,
        flags=re.IGNORECASE,
    )
    texte = texte.strip().strip('"').strip("«»").strip()

    # Une reponse multiligne est un debut de liste ou d'explication : on ne
    # garde que la premiere ligne utile.
    lignes = [ligne.strip() for ligne in texte.splitlines() if ligne.strip()]
    return lignes[0] if lignes else ""


class Reformulateur:
    """Reecrit une question dans le vocabulaire du corpus."""

    def __init__(self, settings: Settings, generator: Generator) -> None:
        self._settings = settings
        self._generator = generator

    def __call__(self, question: str) -> str:
        """Renvoie la requete a envoyer chercher.

        En cas de doute, la question d'origine : cette fonction ne leve
        jamais, et n'est jamais une raison de ne pas repondre.
        """
        question = question.strip()
        if not question:
            return question

        try:
            brut = self._generator.completer(
                CONSIGNE,
                f"Question de l'utilisateur : {question}",
                temperature=self._settings.reformulation_temperature,
                max_tokens=self._settings.reformulation_max_tokens,
                modele=self._settings.reformulation_model or None,
            )
        except GenerationError as exc:
            logger.warning(
                "Reformulation impossible (%s). La question d'origine est "
                "utilisee telle quelle.",
                exc,
            )
            return question

        reecrite = _nettoyer(brut)

        if not self._plausible(question, reecrite):
            return question

        if reecrite.lower() == question.lower():
            logger.debug("Reformulation identique a la question.")
            return question

        logger.info("Question reformulee pour la recherche : %s", reecrite)
        return reecrite

    def _plausible(self, question: str, reecrite: str) -> bool:
        """Garde-fou : une reformulation aberrante est ecartee en silence."""
        if not reecrite:
            logger.warning("Reformulation vide : question d'origine conservee.")
            return False

        if len(reecrite) > LONGUEUR_MAX or len(reecrite) > len(question) * RAPPORT_MAX:
            logger.warning(
                "Reformulation trop longue (%d caracteres pour une question de %d) : "
                "le modele a probablement commence a repondre. Question d'origine "
                "conservee.",
                len(reecrite),
                len(question),
            )
            return False

        return True


def construire(
    settings: Settings, generator: Generator | None = None
) -> Reformulateur | None:
    """Le reformulateur a utiliser, ou None s'il n'y a pas lieu.

    Renvoie None si la reformulation est desactivee, ou si aucun modele
    n'est configure — la recherche fonctionne alors comme avant, sans que
    l'appelant ait a s'en soucier.
    """
    if not settings.reformulation_enabled:
        return None

    if generator is None:
        try:
            generator = Generator(settings)
        except GenerationError as exc:
            logger.info(
                "Reformulation desactivee : aucun modele disponible (%s).", exc
            )
            return None

    return Reformulateur(settings, generator)
