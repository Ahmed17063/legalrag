"""Statut juridique des textes : en vigueur, abroge, ou inconnu.

Un assistant juridique doit pouvoir repondre sur le droit anterieur sans
jamais le presenter comme applicable. Citer l'ordonnance 88-144 comme
regle en vigueur serait pire qu'une absence de reponse.

**Le statut n'est pas dans l'index, et c'est deliberé.** Reclasser un
texte comme abroge est une decision juridique qui peut tomber a tout
moment ; l'inscrire dans les vecteurs obligerait a replonger des milliers
de chunks a chaque changement. Il est donc resolu au moment de la
recherche, a partir d'un registre declaratif que l'on edite a la main.

Le registre porte sur le `document_id`, l'unite dont on dispose. Un
recueil peut contenir des articles abroges isolement : la mention
explicite d'abrogation dans le texte d'un passage est donc detectee en
plus, et signalee.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

CHEMIN_PAR_DEFAUT = Path("data/statuts.json")


class Statut(str, Enum):
    """Etat juridique d'un texte."""

    EN_VIGUEUR = "en_vigueur"
    ABROGE = "abroge"
    # Le defaut : on ne sait pas, et on ne fait pas semblant de savoir.
    INCONNU = "inconnu"

    @property
    def libelle(self) -> str:
        return {
            Statut.EN_VIGUEUR: "en vigueur",
            Statut.ABROGE: "ABROGÉ",
            Statut.INCONNU: "statut non renseigné",
        }[self]

    @property
    def citable_comme_droit_applicable(self) -> bool:
        return self is Statut.EN_VIGUEUR


@dataclass(frozen=True, slots=True)
class FicheStatut:
    """Ce que le registre dit d'un document."""

    document_id: str
    statut: Statut = Statut.INCONNU
    note: str = ""
    remplace_par: str = ""

    def mention(self) -> str:
        """Mention courte a accoler a une citation."""
        if self.statut is Statut.EN_VIGUEUR:
            return ""
        if self.statut is Statut.ABROGE:
            suite = f", remplacé par {self.remplace_par}" if self.remplace_par else ""
            return f" [ABROGÉ{suite}]"
        return " [statut non vérifié]"


# Une abrogation mentionnee dans le corps d'un texte : "l'article 12 est
# abroge", "sont abrogees toutes dispositions contraires". Le passage
# n'est pas forcement abroge lui-meme, mais il merite un signalement.
_MENTION_ABROGATION = re.compile(
    r"(?i)\b(abrog[ée]e?s?|abrogation|cess[ée]e?\s+de\s+produire\s+effet"
    r"|n'?est\s+plus\s+en\s+vigueur)\b"
)


class RegistreStatuts:
    """Registre document_id -> statut, charge depuis un fichier JSON.

    Un document absent du registre est `INCONNU`, jamais `EN_VIGUEUR` par
    defaut : sur du droit, l'absence d'information n'est pas une garantie.
    """

    def __init__(self, fiches: dict[str, FicheStatut] | None = None) -> None:
        self._fiches = fiches or {}

    # -- chargement --------------------------------------------------------

    @classmethod
    def charger(cls, chemin: Path | None = None) -> RegistreStatuts:
        """Lit le registre. Un fichier absent donne un registre vide."""
        chemin = Path(chemin) if chemin else CHEMIN_PAR_DEFAUT

        if not chemin.is_file():
            logger.info(
                "Aucun registre de statuts (%s) : tous les textes seront marques "
                "« statut non vérifié ».",
                chemin,
            )
            return cls()

        try:
            brut = json.loads(chemin.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # Un registre illisible ne doit pas empecher de repondre, mais
            # il ne doit pas non plus passer inapercu.
            logger.error("Registre de statuts illisible (%s) : %s", chemin, exc)
            return cls()

        fiches: dict[str, FicheStatut] = {}
        for document_id, valeur in (brut.get("documents") or {}).items():
            if isinstance(valeur, str):
                valeur = {"statut": valeur}
            fiches[document_id] = FicheStatut(
                document_id=document_id,
                statut=_lire_statut(valeur.get("statut"), document_id),
                note=valeur.get("note", ""),
                remplace_par=valeur.get("remplace_par", ""),
            )

        compte: dict[str, int] = {}
        for fiche in fiches.values():
            compte[fiche.statut.value] = compte.get(fiche.statut.value, 0) + 1
        logger.info("Registre de statuts : %d document(s) %s", len(fiches), compte)
        return cls(fiches)

    # -- consultation ------------------------------------------------------

    def fiche(self, document_id: str) -> FicheStatut:
        return self._fiches.get(document_id, FicheStatut(document_id=document_id))

    def statut(self, document_id: str) -> Statut:
        return self.fiche(document_id).statut

    def __len__(self) -> int:
        return len(self._fiches)

    @property
    def abroges(self) -> list[str]:
        return sorted(f.document_id for f in self._fiches.values() if f.statut is Statut.ABROGE)


def _lire_statut(valeur: object, document_id: str) -> Statut:
    try:
        return Statut(str(valeur))
    except ValueError:
        logger.warning(
            "Statut inconnu %r pour %s : traite comme non verifie. "
            "Valeurs acceptees : %s",
            valeur,
            document_id,
            ", ".join(s.value for s in Statut),
        )
        return Statut.INCONNU


def mentionne_une_abrogation(texte: str) -> bool:
    """Le passage evoque-t-il une abrogation ?

    Sert a signaler qu'un extrait peut concerner du droit qui n'est plus
    applicable, meme quand le document dans son ensemble est en vigueur.
    """
    return bool(_MENTION_ABROGATION.search(texte))
