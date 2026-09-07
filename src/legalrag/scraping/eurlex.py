"""Collecte de textes europeens sur EUR-Lex.

EUR-Lex publie un `robots.txt` qui autorise la collecte, avec un
`Crawl-delay: 10` respecte ici, et n'interdit que le format DOC. Les
textes sont adressables par leur numero **CELEX**, ce qui evite d'avoir
a passer par la recherche : on sait quels reglements on veut.

Ce module ne couvre pas le socle legislatif mauritanien, deja present
dans les recueils officiels du Secretariat General du Gouvernement et
plus a jour que ce qu'on trouverait ailleurs. Il comble le manque reel :
la couche europeenne et internationale, absente du corpus.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

logger = logging.getLogger(__name__)

BASE_URL = "https://eur-lex.europa.eu/legal-content"

# Impose par le robots.txt d'EUR-Lex. Le respecter n'est pas optionnel.
CRAWL_DELAY = 10.0

USER_AGENT = "legalrag/0.1 (collecte de textes de loi publics ; contact via le depot)"


@dataclass(frozen=True, slots=True)
class TexteEuropeen:
    """Un texte identifie par son numero CELEX."""

    celex: str
    titre: str
    # Nom de fichier lisible, qui deviendra le titre du document apres
    # ingestion. Le nom de fichier prime sur les metadonnees PDF.
    fichier: str


# Textes europeens pertinents pour la peche maritime mauritanienne, absents
# du corpus. Les numeros CELEX sont stables : 3 = droit derive, l'annee,
# R = reglement, D = decision, puis le numero.
PECHE_MARITIME: tuple[TexteEuropeen, ...] = (
    TexteEuropeen(
        "32008R1005",
        "Reglement (CE) 1005/2008 etablissant un systeme communautaire "
        "destine a prevenir, decourager et eradiquer la peche INN",
        "UE-reglement-1005-2008-peche-INN",
    ),
    TexteEuropeen(
        "32009R1224",
        "Reglement (CE) 1224/2009 instituant un regime de controle de l'Union",
        "UE-reglement-1224-2009-controle",
    ),
    TexteEuropeen(
        "32011R0404",
        "Reglement d'execution (UE) 404/2011 portant modalites d'application "
        "du reglement controle",
        "UE-reglement-404-2011-modalites-controle",
    ),
    TexteEuropeen(
        "32017R2403",
        "Reglement (UE) 2017/2403 relatif a la gestion durable des flottes "
        "de peche externes",
        "UE-reglement-2403-2017-flottes-externes",
    ),
)


class EurlexError(RuntimeError):
    """Echec de recuperation sur EUR-Lex."""


class EurlexCollector:
    """Telecharge des textes EUR-Lex par numero CELEX."""

    def __init__(self, langue: str = "FR", delai: float = CRAWL_DELAY) -> None:
        self._langue = langue.upper()
        self._delai = delai
        self._derniere = 0.0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})

    def _patienter(self) -> None:
        ecoule = time.monotonic() - self._derniere
        if ecoule < self._delai:
            time.sleep(self._delai - ecoule)
        self._derniere = time.monotonic()

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        stop=stop_after_attempt(4),
        wait=wait_exponential_jitter(initial=5, max=60),
        reraise=True,
    )
    def _telecharger(self, url: str) -> bytes:
        self._patienter()
        reponse = self._session.get(url, timeout=120)
        reponse.raise_for_status()
        return reponse.content

    def url_pdf(self, celex: str) -> str:
        """URL du PDF. Le format DOC est interdit par le robots.txt, pas le PDF."""
        return f"{BASE_URL}/{self._langue}/TXT/PDF/?uri=CELEX:{celex}"

    def recuperer(self, texte: TexteEuropeen, destination: Path) -> Path | None:
        """Telecharge un texte, sauf s'il est deja present.

        Renvoie le chemin du PDF, ou None si la recuperation a echoue :
        un texte manquant ne doit pas interrompre la collecte du reste.
        """
        destination.mkdir(parents=True, exist_ok=True)
        cible = destination / f"{texte.fichier}.pdf"

        if cible.is_file() and cible.stat().st_size > 0:
            logger.debug("deja present : %s", cible.name)
            return cible

        try:
            contenu = self._telecharger(self.url_pdf(texte.celex))
        except requests.RequestException as exc:
            logger.error("%s (%s) : %s", texte.titre[:60], texte.celex, exc)
            return None

        # EUR-Lex renvoie parfois une page d'erreur HTML avec un code 200
        # quand le format demande n'existe pas pour ce texte.
        if not contenu.startswith(b"%PDF"):
            logger.error(
                "%s (%s) : la reponse n'est pas un PDF (%d octets). "
                "Le texte n'existe peut-etre pas dans cette langue.",
                texte.titre[:60],
                texte.celex,
                len(contenu),
            )
            return None

        temporaire = cible.with_suffix(".pdf.part")
        temporaire.write_bytes(contenu)
        temporaire.replace(cible)
        logger.info("%s (%d Ko)", cible.name, len(contenu) // 1024)
        return cible

    def collecter(
        self, textes: tuple[TexteEuropeen, ...], destination: Path
    ) -> tuple[list[Path], list[TexteEuropeen]]:
        """Telecharge une serie de textes. Renvoie (recuperes, echoues)."""
        recuperes: list[Path] = []
        echoues: list[TexteEuropeen] = []

        for texte in textes:
            chemin = self.recuperer(texte, destination)
            if chemin is None:
                echoues.append(texte)
            else:
                recuperes.append(chemin)

        logger.info(
            "%d texte(s) recupere(s), %d en echec (delai respecte : %.0fs entre requetes)",
            len(recuperes),
            len(echoues),
            self._delai,
        )
        return recuperes, echoues
