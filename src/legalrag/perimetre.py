"""Restriction du corpus interrogeable a un sous-ensemble de documents.

Le corpus indexe couvre tout le droit mauritanien — 43 documents, 12 000
chunks. Pour travailler sur un domaine precis, la peche par exemple, il
est utile de n'interroger que les textes qui s'y rapportent : le reste ne
fait qu'ajouter du bruit au vectoriel et diluer le budget de reranking.

**Rien n'est supprime.** La restriction s'applique *a la recherche*, comme
le registre des statuts : un filtre Qdrant, pose au moment d'interroger.
Revenir au corpus complet ne demande ni reindexation, ni reembedding, ni
ecriture dans la collection — seulement de remettre `actif` a `false`. Les
embeddings sont payes une fois, ils restent.

Le perimetre vit dans un fichier JSON plutot que dans une variable
d'environnement : une liste d'identifiants longs y est lisible, et chacun
peut porter la raison de sa presence. Sur un corpus juridique, savoir
*pourquoi* un texte est dans le champ compte autant que la liste.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class PerimetreError(RuntimeError):
    """Fichier de perimetre illisible ou mal forme."""


@dataclass(frozen=True, slots=True)
class Perimetre:
    """Les documents que la recherche a le droit de ramener."""

    actif: bool = False
    documents: frozenset[str] = frozenset()
    libelle: str = ""

    def autorise(self, document_id: str) -> bool:
        return not self.actif or document_id in self.documents

    def __bool__(self) -> bool:
        """Vrai s'il restreint effectivement quelque chose."""
        return self.actif and bool(self.documents)

    @classmethod
    def charger(cls, chemin: Path | None) -> Perimetre:
        """Lit le fichier de perimetre. Absent, le corpus reste entier.

        Un fichier absent n'est pas une erreur : c'est le cas normal d'une
        installation qui interroge tout le corpus.
        """
        if chemin is None or not chemin.is_file():
            return cls()

        try:
            brut = json.loads(chemin.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PerimetreError(f"Perimetre illisible ({chemin}) : {exc}") from exc

        if not isinstance(brut, dict):
            raise PerimetreError(
                f"Perimetre mal forme ({chemin}) : un objet JSON est attendu."
            )

        actif = bool(brut.get("actif", False))
        entrees = brut.get("documents", {})

        # Deux ecritures acceptees : une liste d'identifiants, ou un objet
        # associant chaque identifiant a la raison de sa presence.
        if isinstance(entrees, dict):
            documents = frozenset(entrees)
        elif isinstance(entrees, list):
            documents = frozenset(str(x) for x in entrees)
        else:
            raise PerimetreError(
                f"Perimetre mal forme ({chemin}) : 'documents' doit etre une "
                "liste ou un objet."
            )

        perimetre = cls(
            actif=actif,
            documents=documents,
            libelle=str(brut.get("libelle", "")),
        )

        if actif and not documents:
            # Un perimetre actif et vide ne ramenerait jamais rien : c'est
            # presque surement une erreur de saisie, pas une intention.
            logger.warning(
                "Perimetre actif mais aucun document liste (%s) : "
                "le corpus entier reste interrogeable.",
                chemin,
            )
            return cls()

        if perimetre:
            logger.info(
                "Perimetre restreint a %d document(s)%s.",
                len(documents),
                f" — {perimetre.libelle}" if perimetre.libelle else "",
            )

        return perimetre
