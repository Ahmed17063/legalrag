"""Recherche dans l'index : question -> passages cites.

La recherche vectorielle seule ramene des chunks de ~2500 caracteres qui
contiennent en moyenne **six articles**. Le bon article s'y trouve, mais
noye : on ne peut ni le citer precisement, ni le donner tel quel a un
modele de generation sans diluer le contexte utile.

La recherche se fait donc en trois temps :

1. **Vectoriel** — Qdrant ramene les `top_k` chunks les plus proches de la
   question. C'est rapide et rappelle largement.
2. **Decoupage par article** — chaque chunk est redecoupe sur ses en-tetes
   d'article. La granularite passe du bloc de six articles a l'article.
3. **Reranking** — un modele de reranking classe ces passages face a la
   question. Contrairement a l'embedding, il voit la question et le passage
   ensemble, ce qui le rend bien meilleur pour trancher entre des passages
   thematiquement proches.

Le decoupage doit distinguer un **en-tete** d'article d'un **renvoi**
("conformement a l'article 3 ci-dessus"). Sur ce corpus, 2187 occurrences
de "Article N" pour 348 chunks : couper sur toutes en ferait de la
charpie.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient, models

from .config import Settings
from .perimetre import Perimetre
from .statuts import FicheStatut, RegistreStatuts, Statut, mentionne_une_abrogation
from .voyage_http import VoyageTransport

logger = logging.getLogger(__name__)

# Un en-tete d'article : "Article" capitalise, un numero, puis un
# separateur. Les renvois du corpus sont en minuscules ("article 3
# ci-dessus"), ce qui donne un signal simple et fiable.
_ARTICLE_HEADING = re.compile(
    r"(?P<label>ARTICLE|Article)\s+"
    r"(?P<num>\d+(?:\s*[-–]\s*\d+)?)"
    r"\s*(?P<sep>[-–.:]{1,3})",
)

# Meme capitalise, un "Article N" suivi de ces mots est un renvoi.
_RENVOI = re.compile(
    r"^\s*(ci-dessus|ci-dessous|ci-apres|precite|précité|susvise|susvisé"
    r"|de la presente|de la présente|du present|du présent|des presentes|et suivants)",
    re.IGNORECASE,
)

# Estimation grossiere du nombre de tokens d'un texte francais. Sert a ne
# pas soumettre au reranking plus que le debit autorise ne le permet. On
# vise volontairement bas : un lot refuse coute bien plus cher qu'un lot
# un peu trop prudent.
CHARS_PER_TOKEN = 3.5

# Marge sous le plafond de debit, pour absorber l'ecart entre cette
# estimation et le comptage reel du serveur.
RERANK_BUDGET_RATIO = 0.8


def estimate_tokens(text: str) -> int:
    """Nombre de tokens approximatif d'un texte."""
    return max(1, int(len(text) / CHARS_PER_TOKEN))


# Longueur minimale du *corps* d'un article, en-tete exclu. En dessous, ce
# n'est pas un article mais un en-tete orphelin en fin de chunk, qu'on
# rattache au passage precedent.
#
# Le seuil porte sur le corps et non sur le passage entier : en droit, un
# article d'une seule phrase est courant. "Article 326- Aucune personne
# acquittee legalement ne peut plus etre reprise ou accusee a raison des
# memes faits" fait 152 caracteres et doit rester un passage a part
# entiere.
MIN_BODY_CHARS = 30


@dataclass(frozen=True, slots=True)
class Passage:
    """Un extrait citable, issu du decoupage d'un chunk."""

    text: str
    document: str
    document_id: str
    page_start: int
    page_end: int
    pages: tuple[int, ...]
    timestamp: str
    chunk_id: str
    article: str | None = None
    vector_score: float = 0.0
    rerank_score: float | None = None
    # Resolu a la recherche depuis le registre, jamais depuis l'index :
    # une abrogation ne doit pas obliger a replonger le corpus.
    statut: Statut = Statut.INCONNU
    mention_statut: str = ""
    evoque_une_abrogation: bool = False

    @property
    def score(self) -> float:
        """Score retenu : celui du reranking s'il a eu lieu."""
        return self.rerank_score if self.rerank_score is not None else self.vector_score

    def citation(self) -> str:
        """Reference lisible, du type 'Code X, art. 326 (p. 127-128)'."""
        pages = (
            f"p. {self.page_start}"
            if self.page_start == self.page_end
            else f"p. {self.page_start}-{self.page_end}"
        )
        article = f", art. {self.article}" if self.article else ""
        return f"{self.document}{article} ({pages}){self.mention_statut}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "citation": self.citation(),
            "document": self.document,
            "document_id": self.document_id,
            "article": self.article,
            "pages": list(self.pages),
            "timestamp": self.timestamp,
            "chunk_id": self.chunk_id,
            "statut": self.statut.value,
            "mention_statut": self.mention_statut,
            "evoque_une_abrogation": self.evoque_une_abrogation,
            "score": round(self.score, 4),
            "vector_score": round(self.vector_score, 4),
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 4),
            "text": self.text,
        }


@dataclass
class SearchResult:
    """Ce qu'une recherche a produit, et comment."""

    query: str
    passages: list[Passage] = field(default_factory=list)
    chunks_examined: int = 0
    passages_examined: int = 0
    reranked: bool = False
    tokens_billed: int = 0
    # Requete reellement envoyee au moteur vectoriel, si elle differe de la
    # question. Rendue a l'appelant : une recherche doit rester explicable.
    query_reformulee: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "query_reformulee": self.query_reformulee,
            "reranked": self.reranked,
            "chunks_examined": self.chunks_examined,
            "passages_examined": self.passages_examined,
            "tokens_billed": self.tokens_billed,
            "passages": [p.as_dict() for p in self.passages],
        }


# ---------------------------------------------------------------------------
# Decoupage par article
# ---------------------------------------------------------------------------


def _is_heading(text: str, match: re.Match[str]) -> bool:
    """Un en-tete d'article, ou un simple renvoi ?"""
    if match.group("label")[0].islower():
        return False
    return not _RENVOI.match(text[match.end() : match.end() + 40])


def _body_length(numero: str | None, morceau: str) -> int:
    """Longueur du texte reel, en-tete d'article exclu."""
    if numero is None:
        return len(morceau)
    return len(_ARTICLE_HEADING.sub("", morceau, count=1).strip())


def split_articles(text: str, min_chars: int = MIN_BODY_CHARS) -> list[tuple[str | None, str]]:
    """Decoupe un chunk sur ses en-tetes d'article.

    Renvoie une liste de `(numero_d_article, texte)`. Le numero vaut None
    pour ce qui precede le premier en-tete, ou quand le chunk n'en contient
    aucun : dans ce cas le chunk est renvoye entier, ce qui garantit qu'on
    ne perd jamais de contenu.
    """
    coupures = [m for m in _ARTICLE_HEADING.finditer(text) if _is_heading(text, m)]
    if not coupures:
        nettoye = text.strip()
        return [(None, nettoye)] if nettoye else []

    bornes: list[tuple[str | None, int, int]] = []
    if coupures[0].start() > 0:
        bornes.append((None, 0, coupures[0].start()))
    for i, m in enumerate(coupures):
        fin = coupures[i + 1].start() if i + 1 < len(coupures) else len(text)
        numero = re.sub(r"\s+", "", m.group("num"))
        bornes.append((numero, m.start(), fin))

    passages: list[tuple[str | None, str]] = []
    for numero, debut, fin in bornes:
        morceau = text[debut:fin].strip()
        if not morceau:
            continue
        # Une bribe sans corps est rattachee au passage precedent : un
        # en-tete orphelin en fin de chunk n'a aucune valeur seul.
        if _body_length(numero, morceau) < min_chars and passages:
            precedent_num, precedent_txt = passages[-1]
            passages[-1] = (precedent_num, f"{precedent_txt}\n{morceau}")
        else:
            passages.append((numero, morceau))

    return passages


# ---------------------------------------------------------------------------
# Recherche
# ---------------------------------------------------------------------------


class Retriever:
    """Recherche dans la collection Qdrant, avec reranking optionnel."""

    def __init__(
        self,
        settings: Settings,
        reformulateur: Callable[[str], str] | None = None,
    ) -> None:
        # Le reformulateur est *injecte* plutot que construit ici : il
        # s'appuie sur `generation`, qui importe deja ce module. Le
        # construire ici creerait un cycle d'imports.
        self._reformulateur = reformulateur
        self._settings = settings
        self._transport = VoyageTransport(
            api_key=settings.voyage_api_key,
            base_url=settings.voyage_base_url,
            max_requests_per_minute=settings.max_requests_per_minute,
            max_tokens_per_minute=settings.max_tokens_per_minute,
        )
        self._client = _make_client(settings)
        self._registre = RegistreStatuts.charger(settings.statuts_file)
        self._perimetre = Perimetre.charger(settings.perimetre_file)

    # -- cycle de vie ------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Retriever:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- appels Voyage -----------------------------------------------------

    def embed_query(self, query: str) -> tuple[list[float], int]:
        """Plonge la question.

        `input_type="query"` n'est pas un detail : le modele place les
        questions et les documents differemment dans l'espace. Utiliser
        "document" pour une question degraderait la recherche.
        """
        payload = {
            "inputs": [[query]],
            "model": self._settings.embedding_model,
            "input_type": "query",
            "output_dimension": self._settings.embedding_dimension,
        }
        self._transport.limiter.acquire(estimate_tokens(query))
        data = self._transport.post("contextualizedembeddings", payload)
        vecteur = data["data"][0]["data"][0]["embedding"]
        return vecteur, int(data.get("usage", {}).get("total_tokens", 0))

    def _fit_budget(self, passages: list[Passage]) -> list[Passage]:
        """Ne garde que ce qu'une requete de reranking peut transporter.

        Sur un compte plafonne a 10 000 tokens par minute, soumettre les
        150 passages issus de 25 chunks represente environ 16 000 tokens :
        la requete serait refusee a chaque tentative, sans qu'aucun reessai
        puisse aboutir. On tronque donc la liste des candidats, en gardant
        ceux dont le chunk d'origine etait le mieux classe.
        """
        budget = int(self._settings.max_tokens_per_minute * RERANK_BUDGET_RATIO)

        retenus: list[Passage] = []
        total = 0
        for passage in passages:
            cout = estimate_tokens(passage.text)
            if retenus and total + cout > budget:
                break
            retenus.append(passage)
            total += cout

        if len(retenus) < len(passages):
            logger.warning(
                "Reranking limite a %d passage(s) sur %d (~%d tokens, budget %d). "
                "Baisser RETRIEVAL_TOP_K, ou relever MAX_TOKENS_PER_MINUTE si le "
                "compte le permet.",
                len(retenus),
                len(passages),
                total,
                budget,
            )
        return retenus

    def rerank(self, query: str, passages: list[Passage], top_n: int) -> tuple[list[Passage], int]:
        """Reclasse les passages face a la question."""
        if not passages:
            return [], 0

        passages = self._fit_budget(passages)
        self._transport.limiter.acquire(sum(estimate_tokens(p.text) for p in passages))

        payload = {
            "query": query,
            "documents": [p.text for p in passages],
            "model": self._settings.rerank_model,
            "top_k": min(top_n, len(passages)),
            # Les passages sont courts, mais un chunk sans en-tete d'article
            # peut rester long : on laisse le serveur tronquer plutot que
            # de faire echouer l'appel.
            "truncation": True,
        }
        data = self._transport.post("rerank", payload)

        classes = [
            Passage(
                **{
                    **{
                        champ: getattr(passages[item["index"]], champ)
                        for champ in (
                            "text", "document", "document_id", "page_start", "page_end",
                            "pages", "timestamp", "chunk_id", "article", "vector_score",
                        )
                    },
                    "rerank_score": float(item["relevance_score"]),
                }
            )
            for item in sorted(data["data"], key=lambda x: -x["relevance_score"])
        ]
        return classes, int(data.get("usage", {}).get("total_tokens", 0))

    def documents(self) -> list[dict[str, Any]]:
        """Documents presents dans la collection, avec leur nombre de chunks.

        Sert a alimenter le filtre de l'interface. La collection tient en
        quelques centaines de points : un parcours complet est moins couteux
        qu'une agregation, et fonctionne sur toutes les versions de Qdrant.
        """
        compte: dict[str, dict[str, Any]] = {}
        offset = None
        while True:
            points, offset = self._client.scroll(
                self._settings.qdrant_collection,
                limit=1000,
                offset=offset,
                with_payload=["document", "document_id"],
                with_vectors=False,
            )
            for point in points:
                charge = point.payload or {}
                cle = charge.get("document_id")
                if not cle:
                    continue
                entree = compte.setdefault(
                    cle, {"document_id": cle, "document": charge.get("document", cle), "chunks": 0}
                )
                entree["chunks"] += 1
            if offset is None:
                break

        # Le perimetre s'applique aussi au catalogue : proposer dans
        # l'interface un document que la recherche refusera d'ouvrir ne
        # menerait qu'a une erreur.
        retenus = [d for d in compte.values() if self._perimetre.autorise(d["document_id"])]
        return sorted(retenus, key=lambda d: -d["chunks"])

    def _filtre(self, document_id: str | None) -> models.Filter | None:
        """Filtre Qdrant combinant le perimetre et le document demande.

        Les deux se cumulent : un document hors perimetre ne devient pas
        interrogeable parce qu'on le nomme explicitement. L'appelant est
        prevenu plutot que de recevoir zero resultat sans explication.
        """
        conditions: list[models.FieldCondition] = []

        if document_id:
            if not self._perimetre.autorise(document_id):
                raise ValueError(
                    f"Le document {document_id!r} est hors du perimetre courant "
                    f"({self._perimetre.libelle or 'restreint'}). "
                    f"Voir {self._settings.perimetre_file}."
                )
            conditions.append(
                models.FieldCondition(
                    key="document_id", match=models.MatchValue(value=document_id)
                )
            )
        elif self._perimetre:
            conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=sorted(self._perimetre.documents)),
                )
            )

        return models.Filter(must=conditions) if conditions else None

    # -- recherche ---------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
        top_n: int | None = None,
        document_id: str | None = None,
        rerank: bool | None = None,
        reformuler: bool | None = None,
    ) -> SearchResult:
        """Cherche les passages qui repondent a la question."""
        top_k = top_k or self._settings.retrieval_top_k
        top_n = top_n or self._settings.retrieval_top_n
        rerank = self._settings.rerank_enabled if rerank is None else rerank
        if reformuler is None:
            reformuler = self._settings.reformulation_enabled

        result = SearchResult(query=query)

        # La recherche vectorielle part de la reformulation — elle cherche du
        # vocabulaire ; le reranking, lui, gardera la question d'origine.
        requete = query
        if reformuler and self._reformulateur is not None:
            reecrite = self._reformulateur(query)
            if reecrite and reecrite != query:
                requete = reecrite
                result.query_reformulee = reecrite

        vecteur, tokens = self.embed_query(requete)
        result.tokens_billed += tokens

        filtre = self._filtre(document_id)

        hits = self._client.query_points(
            self._settings.qdrant_collection,
            query=vecteur,
            limit=top_k,
            query_filter=filtre,
            with_payload=True,
        ).points
        result.chunks_examined = len(hits)

        passages = _explode(hits, self._registre)
        result.passages_examined = len(passages)
        logger.info(
            "%d chunk(s) -> %d passage(s) au niveau de l'article", len(hits), len(passages)
        )

        if rerank and passages:
            classes, tokens = self.rerank(query, passages, top_n)
            result.tokens_billed += tokens
            result.passages = classes
            result.reranked = True
        else:
            # Sans reranking, on garde l'ordre vectoriel des chunks, en ne
            # conservant que le premier passage de chacun : les suivants
            # n'ont pas ete classes face a la question.
            result.passages = passages[:top_n]

        return result


def _explode(hits: list[Any], registre: RegistreStatuts | None = None) -> list[Passage]:
    """Transforme les chunks retournes par Qdrant en passages par article.

    Le statut est appose ici, a partir du registre : c'est le dernier
    moment ou il peut l'etre sans figer une decision juridique dans les
    vecteurs.
    """
    registre = registre or RegistreStatuts()
    passages: list[Passage] = []
    for hit in hits:
        charge = hit.payload
        fiche = registre.fiche(charge["document_id"])
        for numero, texte in split_articles(charge["text"]):
            passages.append(
                Passage(
                    text=texte,
                    document=charge["document"],
                    document_id=charge["document_id"],
                    page_start=charge["page_start"],
                    page_end=charge["page_end"],
                    pages=tuple(charge["pages"]),
                    timestamp=charge.get("timestamp", ""),
                    chunk_id=charge["chunk_id"],
                    article=numero,
                    vector_score=float(hit.score),
                    statut=fiche.statut,
                    mention_statut=fiche.mention(),
                    evoque_une_abrogation=mentionne_une_abrogation(texte),
                )
            )
    return passages


def _make_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=60)
    return QdrantClient(path=str(settings.qdrant_path))

