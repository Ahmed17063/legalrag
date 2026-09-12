"""Interface web du chatbot juridique.

Une page unique, servie par FastAPI, qui interroge la meme chaine que la
ligne de commande : recherche vectorielle, decoupage par article,
reranking, puis redaction sourcee.

Deux contraintes ont dicte la conception :

* **Une question prend une a deux minutes.** Le compte Voyage est plafonne
  a deux requetes par minute, et une recherche en consomme deux. L'interface
  doit donc afficher une progression credible plutot qu'un sablier muet.
* **Les traitements sont bloquants.** Les points d'entree sont declares en
  `def` et non `async def` : FastAPI les execute alors dans un fil separe,
  sans bloquer la boucle d'evenements. Un verrou serialise les recherches,
  ce qui correspond de toute facon au debit autorise.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from ..config import ConfigError, Settings, load_settings
from ..generation import GenerationError, Generator
from ..logging_setup import setup_logging
from ..retrieval import Retriever

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# Les recherches sont serialisees : le debit autorise ne permet de toute
# facon pas de paralleliser, et cela evite que deux requetes simultanees
# se disputent le limiteur.
_verrou = threading.Lock()


class Question(BaseModel):
    """Ce que le navigateur envoie."""

    question: str = Field(min_length=3, max_length=1000)
    document_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    top_n: int | None = Field(default=None, ge=1, le=10)
    rerank: bool = True
    # Cle d'un modele configure (`groq`, `claude`). Absente, le defaut du
    # serveur s'applique : un navigateur ancien continue de fonctionner.
    modele: str | None = Field(default=None, max_length=40)


class Etat:
    """Ressources partagees, construites une fois au demarrage."""

    def __init__(self) -> None:
        self.settings: Settings | None = None
        self.retriever: Retriever | None = None
        self.generator: Generator | None = None
        self.erreur: str | None = None
        self.documents: list[dict[str, Any]] = []

    def demarrer(self) -> None:
        try:
            self.settings = load_settings()
        except ConfigError as exc:
            self.erreur = str(exc)
            logger.error("Configuration invalide : %s", exc)
            return

        try:
            self.retriever = Retriever(self.settings)
            self.documents = self.retriever.documents()
            logger.info(
                "%d document(s) dans la collection %r",
                len(self.documents),
                self.settings.qdrant_collection,
            )
        except Exception as exc:  # noqa: BLE001 - on veut afficher la cause
            self.erreur = f"Connexion a Qdrant impossible : {exc}"
            logger.error(self.erreur)
            return

        try:
            self.generator = Generator(self.settings)
        except GenerationError as exc:
            # La recherche reste utilisable sans generation : on n'echoue pas.
            logger.warning("Generation indisponible : %s", exc)

    def arreter(self) -> None:
        if self.retriever is not None:
            self.retriever.close()


etat = Etat()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    etat.demarrer()
    yield
    etat.arreter()


app = FastAPI(
    title="Chatbot juridique mauritanien",
    description="Questions-reponses sourcees sur les codes et textes de loi",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Etat du service, et ce qui manque le cas echeant."""
    modeles = etat.generator.modeles() if etat.generator is not None else []
    return {
        "pret": etat.retriever is not None,
        "generation": etat.generator is not None,
        "erreur": etat.erreur,
        "documents": etat.documents,
        "modele_embedding": etat.settings.embedding_model if etat.settings else None,
        "modele_generation": etat.settings.generation_model if etat.settings else None,
        # Modeles proposables a l'utilisateur, le defaut en tete.
        "modeles": [m.as_dict() for m in modeles],
        "modele_defaut": etat.generator.defaut if etat.generator is not None else None,
    }


@app.post("/api/ask")
def ask(payload: Question) -> JSONResponse:
    """Recherche puis redaction. C'est l'appel long."""
    if etat.retriever is None or etat.settings is None:
        raise HTTPException(503, etat.erreur or "Service non initialise")

    # Verifie avant la recherche : elle consomme du quota Voyage et dure une
    # minute, il serait absurde de la payer pour echouer ensuite.
    if payload.modele and etat.generator is not None:
        connus = {m.cle for m in etat.generator.modeles()}
        if payload.modele not in connus:
            raise HTTPException(
                400,
                f"Modele inconnu : {payload.modele!r}. "
                "Disponibles : " + ", ".join(sorted(connus)) + ".",
            )

    debut = time.monotonic()
    question = payload.question.strip()

    with _verrou:
        try:
            resultat = etat.retriever.search(
                question,
                top_k=payload.top_k,
                top_n=payload.top_n,
                document_id=payload.document_id or None,
                rerank=payload.rerank,
            )
        except Exception as exc:  # noqa: BLE001 - remonte au navigateur
            logger.exception("Recherche en echec")
            raise HTTPException(502, f"Recherche impossible : {exc}") from exc

        if etat.generator is None:
            # Sans cle de generation, on rend quand meme les passages :
            # c'est deja une reponse exploitable, juste non redigee.
            return JSONResponse(
                {
                    "question": question,
                    "answer": None,
                    "raison": (
                        "Generation indisponible : ni GROQ_API_KEY ni "
                        "ANTHROPIC_API_KEY ne sont renseignees."
                    ),
                    "grounded": False,
                    "sources": [],
                    "passages": [p.as_dict() for p in resultat.passages],
                    "duree": round(time.monotonic() - debut, 1),
                }
            )

        try:
            reponse = etat.generator.answer(
                question, resultat.passages, modele=payload.modele
            )
        except GenerationError as exc:
            logger.exception("Generation en echec")
            raise HTTPException(502, f"Redaction impossible : {exc}") from exc

    donnees = reponse.as_dict()
    donnees["duree"] = round(time.monotonic() - debut, 1)
    donnees["chunks_examined"] = resultat.chunks_examined
    donnees["passages_examined"] = resultat.passages_examined
    donnees["reranked"] = resultat.reranked
    return JSONResponse(donnees)
