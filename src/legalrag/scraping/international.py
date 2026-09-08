"""Collecte de textes internationaux depuis leurs sources officielles.

Contrairement a EUR-Lex, ces textes n'ont pas de schema d'URL commun :
chaque organisation publie le sien a sa facon. La liste est donc tenue a
la main, chaque URL ayant ete verifiee.

Les sources retenues autorisent la collecte : le TIDM publie un
`robots.txt` en `Allow: /`, l'ICCAT n'en publie pas, et `faolex.fao.org`
sert ses PDF normalement — c'est seulement `www.fao.org/faolex` qui
refuse les robots, et il n'est pas sollicite ici.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

logger = logging.getLogger(__name__)

USER_AGENT = "legalrag/0.1 (collecte de textes de loi publics ; contact via le depot)"

# Delai entre deux requetes vers un meme hote. Aucune de ces sources
# n'impose de Crawl-delay, mais on ne martele pas un service public.
DELAI_PAR_HOTE = 5.0


@dataclass(frozen=True, slots=True)
class TexteInternational:
    """Un texte et son URL officielle, verifiee."""

    url: str
    titre: str
    fichier: str

    @property
    def hote(self) -> str:
        return urlsplit(self.url).netloc


PECHE_INTERNATIONAL: tuple[TexteInternational, ...] = (
    TexteInternational(
        "https://faolex.fao.org/treaty/docs/tre000003F.pdf",
        "Accord FAO de 2009 relatif aux mesures du ressort de l'Etat du port (PSMA)",
        "International-accord-FAO-2009-mesures-Etat-du-port",
    ),
    TexteInternational(
        "https://www.itlos.org/fileadmin/itlos/documents/cases/case_no.21"
        "/advisory_opinion_published/2015_21-advop-F.pdf",
        "TIDM, avis consultatif du 2 avril 2015, affaire n°21, demande de la CSRP",
        "International-TIDM-2015-avis-consultatif-CSRP-peche-INN",
    ),
    TexteInternational(
        "https://www.iccat.int/Documents/Commission/BasicTexts.pdf",
        "ICCAT, textes fondamentaux dont la convention de Rio de 1966",
        "International-ICCAT-textes-fondamentaux",
    ),
)


class CollecteurInternational:
    """Telecharge des textes a partir d'URL officielles verifiees."""

    def __init__(self, delai: float = DELAI_PAR_HOTE) -> None:
        self._delai = delai
        self._derniere_par_hote: dict[str, float] = {}
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})

    def _patienter(self, hote: str) -> None:
        """Temporise par hote : deux organisations differentes n'ont pas a
        s'attendre l'une l'autre."""
        derniere = self._derniere_par_hote.get(hote, 0.0)
        ecoule = time.monotonic() - derniere
        if ecoule < self._delai:
            time.sleep(self._delai - ecoule)
        self._derniere_par_hote[hote] = time.monotonic()

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        stop=stop_after_attempt(4),
        wait=wait_exponential_jitter(initial=5, max=60),
        reraise=True,
    )
    def _telecharger(self, texte: TexteInternational) -> bytes:
        self._patienter(texte.hote)
        reponse = self._session.get(texte.url, timeout=120, allow_redirects=True)
        reponse.raise_for_status()
        return reponse.content

    def recuperer(self, texte: TexteInternational, destination: Path) -> Path | None:
        destination.mkdir(parents=True, exist_ok=True)
        cible = destination / f"{texte.fichier}.pdf"

        if cible.is_file() and cible.stat().st_size > 0:
            logger.debug("deja present : %s", cible.name)
            return cible

        try:
            contenu = self._telecharger(texte)
        except requests.RequestException as exc:
            logger.error("%s : %s", texte.titre[:60], exc)
            return None

        if not contenu.startswith(b"%PDF"):
            logger.error(
                "%s : la reponse n'est pas un PDF (%d octets). L'URL a peut-etre change.",
                texte.titre[:60],
                len(contenu),
            )
            return None

        temporaire = cible.with_suffix(".pdf.part")
        temporaire.write_bytes(contenu)
        temporaire.replace(cible)
        logger.info("%s (%d Ko)", cible.name, len(contenu) // 1024)
        return cible

    def collecter(
        self, textes: tuple[TexteInternational, ...], destination: Path
    ) -> tuple[list[Path], list[TexteInternational]]:
        recuperes, echoues = [], []
        for texte in textes:
            chemin = self.recuperer(texte, destination)
            (echoues if chemin is None else recuperes).append(
                texte if chemin is None else chemin
            )
        logger.info("%d texte(s) recupere(s), %d en echec", len(recuperes), len(echoues))
        return recuperes, echoues
