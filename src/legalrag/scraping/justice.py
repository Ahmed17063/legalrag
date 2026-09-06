"""Collecte des textes de loi sur archive.justice.gov.mr.

Le site tourne sous SPIP : les pages sont adressees par
`spip.php?article<N>` et les fichiers joints vivent sous `IMG/pdf/`. Le
plan du site (`spip.php?page=plan`) recense les articles, mais pas les
fichiers : il faut donc visiter les pages pour decouvrir les documents.

Le crawler est volontairement lent. Aucun `robots.txt` n'est publie, donc
rien n'interdit la collecte, mais il s'agit d'un service public : une
pause entre chaque requete et un `User-Agent` identifiable valent mieux
qu'une rafale anonyme.

Le corpus est majoritairement en **arabe**. Les documents en francais
sont minoritaires et se reconnaissent au script de leur texte, pas a leur
nom de fichier : `code_des_mines.pdf` est en francais alors que rien dans
son nom ne l'indique. La detection se fait donc sur le contenu.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

logger = logging.getLogger(__name__)

BASE_URL = "https://archive.justice.gov.mr/"

# Rubriques « textes et lois » du site, telles qu'exposees par le menu.
RUBRIQUES = {
    355: "textes civils",
    356: "textes commerciaux",
    357: "textes penaux",
    358: "textes sociaux et commerciaux",
    935: "constitution",
}

# Politesse : le site est un service public, sans robots.txt. Une seconde
# entre deux requetes reste indolore pour lui et suffisante pour nous.
DELAI_ENTRE_REQUETES = 1.0

USER_AGENT = "legalrag/0.1 (collecte de textes de loi publics ; contact via le depot)"

_ARTICLE = re.compile(r"spip\.php\?article(\d+)")
_FICHIER = re.compile(r"""["'](IMG/(?:pdf|doc)/[^"'<>\s]+\.(?:pdf|doc|docx))["']""", re.I)
_TITRE = re.compile(r"<title>(.*?)</title>", re.S | re.I)


class ScrapingError(RuntimeError):
    """Echec de collecte."""


@dataclass(frozen=True, slots=True)
class Document:
    """Un fichier decouvert sur le site."""

    url: str
    nom_fichier: str
    article: int
    titre_page: str = ""

    @property
    def extension(self) -> str:
        return Path(self.nom_fichier).suffix.lower()


@dataclass
class Inventaire:
    """Ce qu'une exploration a trouve."""

    articles_visites: int = 0
    articles_en_echec: list[int] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)

    def resume(self) -> str:
        par_ext = Counter(d.extension for d in self.documents)
        return "\n".join(
            [
                "----- Inventaire archive.justice.gov.mr -----",
                f"Articles visites        : {self.articles_visites}",
                f"Articles en echec       : {len(self.articles_en_echec)}",
                f"Documents distincts     : {len(self.documents)}",
                f"  par extension         : {dict(par_ext)}",
            ]
        )


def detecter_script(texte: str) -> str:
    """Script dominant d'un texte : 'arabe', 'latin', ou 'indetermine'.

    Le nom de fichier ne dit rien de fiable : la detection porte sur les
    lettres reellement presentes.
    """
    compte: Counter[str] = Counter()
    for caractere in texte:
        if not caractere.isalpha():
            continue
        nom = unicodedata.name(caractere, "")
        if "ARABIC" in nom:
            compte["arabe"] += 1
        elif "LATIN" in nom:
            compte["latin"] += 1

    total = sum(compte.values())
    if total < 50:
        return "indetermine"
    dominant, occurrences = compte.most_common(1)[0]
    return dominant if occurrences / total > 0.6 else "indetermine"


class JusticeScraper:
    """Explore le site et telecharge les documents."""

    def __init__(self, base_url: str = BASE_URL, delai: float = DELAI_ENTRE_REQUETES) -> None:
        self._base = base_url
        self._delai = delai
        self._derniere_requete = 0.0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})

    def _patienter(self) -> None:
        ecoule = time.monotonic() - self._derniere_requete
        if ecoule < self._delai:
            time.sleep(self._delai - ecoule)
        self._derniere_requete = time.monotonic()

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=2, max=20),
        reraise=True,
    )
    def _get(self, url: str) -> requests.Response:
        self._patienter()
        reponse = self._session.get(url, timeout=60)
        reponse.raise_for_status()
        return reponse

    # -- decouverte --------------------------------------------------------

    def articles_du_plan(self) -> list[int]:
        """Identifiants d'articles recenses par le plan du site et le flux RSS."""
        trouves: set[int] = set()
        for page in ("spip.php?page=plan", "spip.php?page=backend"):
            try:
                contenu = self._get(urljoin(self._base, page)).text
            except requests.RequestException as exc:
                logger.warning("%s illisible : %s", page, exc)
                continue
            trouves.update(int(n) for n in _ARTICLE.findall(contenu))

        logger.info("%d article(s) recense(s)", len(trouves))
        return sorted(trouves)

    def documents_d_un_article(self, article: int) -> list[Document]:
        """Fichiers joints a une page d'article."""
        url = urljoin(self._base, f"spip.php?article{article}")
        contenu = self._get(url).text

        titre_trouve = _TITRE.search(contenu)
        titre = " ".join(titre_trouve.group(1).split()) if titre_trouve else ""

        vus: set[str] = set()
        documents = []
        for chemin in _FICHIER.findall(contenu):
            if chemin in vus:
                continue
            vus.add(chemin)
            documents.append(
                Document(
                    url=urljoin(self._base, chemin),
                    nom_fichier=Path(chemin).name,
                    article=article,
                    titre_page=titre,
                )
            )
        return documents

    def explorer(self, articles: list[int] | None = None, limite: int | None = None) -> Inventaire:
        """Parcourt les articles et recense leurs fichiers joints."""
        cibles = articles if articles is not None else self.articles_du_plan()
        if limite is not None:
            cibles = cibles[:limite]

        inventaire = Inventaire()
        deja_vus: set[str] = set()

        for position, article in enumerate(cibles, 1):
            try:
                for document in self.documents_d_un_article(article):
                    if document.url in deja_vus:
                        continue
                    deja_vus.add(document.url)
                    inventaire.documents.append(document)
            except requests.RequestException as exc:
                logger.warning("article%d illisible : %s", article, exc)
                inventaire.articles_en_echec.append(article)
            inventaire.articles_visites += 1

            if position % 50 == 0:
                logger.info(
                    "%d/%d articles, %d document(s) trouve(s)",
                    position,
                    len(cibles),
                    len(inventaire.documents),
                )

        return inventaire

    # -- telechargement ----------------------------------------------------

    def telecharger(self, document: Document, destination: Path) -> Path:
        """Recupere un fichier, sauf s'il est deja present."""
        destination.mkdir(parents=True, exist_ok=True)
        cible = destination / document.nom_fichier
        if cible.is_file() and cible.stat().st_size > 0:
            logger.debug("deja telecharge : %s", document.nom_fichier)
            return cible

        contenu = self._get(document.url).content
        # Ecriture atomique : un telechargement interrompu ne laisse pas un
        # fichier tronque qui serait pris pour valide au tour suivant.
        temporaire = cible.with_suffix(cible.suffix + ".part")
        temporaire.write_bytes(contenu)
        temporaire.replace(cible)
        logger.info("%s (%d Ko)", document.nom_fichier, len(contenu) // 1024)
        return cible
