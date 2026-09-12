"""Point d'entree en ligne de commande.

    python -m legalrag.cli index --dry-run
    python -m legalrag.cli index
    python -m legalrag.cli compact
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import ConfigError, load_settings
from .generation import GenerationError
from .logging_setup import setup_logging
from .veille import garder_eveille

# `pipeline`, `index_store` et `qdrant_store` entrainent pyarrow et les
# bibliotheques Google, qui ne servent qu'a l'indexation. Les importer ici
# les rendrait obligatoires pour lancer le serveur web, et l'image Docker
# ne pourrait pas s'en passer. Ils sont donc charges dans les branches qui
# les utilisent.

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="legalrag",
        description="Indexation d'un corpus juridique Drive vers un index d'embeddings.",
    )
    parser.add_argument("--env-file", type=Path, default=None, help="Fichier .env a charger")
    parser.add_argument("-v", "--verbose", action="store_true", help="Logs detailles")

    sub = parser.add_subparsers(dest="command", required=True)

    index_cmd = sub.add_parser("index", help="Telecharge, decoupe et plonge le corpus")
    index_cmd.add_argument(
        "--limit", type=int, default=None, help="Ne traiter que les N premieres pages (test)"
    )
    index_cmd.add_argument(
        "--dry-run",
        action="store_true",
        help="Tout preparer et estimer le cout, sans appeler l'API d'embeddings",
    )
    index_cmd.add_argument(
        "--prune",
        action="store_true",
        help="Supprimer de l'index les lignes qui ne correspondent plus au corpus",
    )
    index_cmd.add_argument(
        "--compact",
        action="store_true",
        help="Fusionner les fragments en un fichier unique une fois le job termine",
    )

    qdrant_cmd = sub.add_parser("qdrant", help="Charger l'index dans Qdrant")
    qdrant_cmd.add_argument(
        "--recreate",
        action="store_true",
        help="Supprimer et recreer la collection avant de charger",
    )
    qdrant_cmd.add_argument(
        "--prune",
        action="store_true",
        help="Supprimer de la collection les points absents de l'index",
    )

    search_cmd = sub.add_parser("search", help="Chercher des passages repondant a une question")
    search_cmd.add_argument("question", help="La question, entre guillemets")
    search_cmd.add_argument("--top-k", type=int, default=None, help="Chunks ramenes par le vectoriel")
    search_cmd.add_argument("--top-n", type=int, default=None, help="Passages renvoyes")
    search_cmd.add_argument("--document", default=None, help="Restreindre a un document_id")
    search_cmd.add_argument("--no-rerank", action="store_true", help="Sauter le reranking")
    search_cmd.add_argument("--json", action="store_true", help="Sortie JSON")

    ask_cmd = sub.add_parser("ask", help="Poser une question et obtenir une reponse redigee")
    ask_cmd.add_argument("question", help="La question, entre guillemets")
    ask_cmd.add_argument("--top-k", type=int, default=None, help="Chunks ramenes par le vectoriel")
    ask_cmd.add_argument("--top-n", type=int, default=None, help="Passages soumis au modele")
    ask_cmd.add_argument("--document", default=None, help="Restreindre a un document_id")
    ask_cmd.add_argument("--no-rerank", action="store_true", help="Sauter le reranking")
    ask_cmd.add_argument("--json", action="store_true", help="Sortie JSON")
    ask_cmd.add_argument(
        "--modele",
        default=None,
        help="Modele redacteur : 'groq' ou 'claude'. Par defaut GENERATION_DEFAULT",
    )
    ask_cmd.add_argument(
        "--show-passages",
        action="store_true",
        help="Afficher aussi le texte integral des extraits utilises",
    )

    ingest_cmd = sub.add_parser(
        "ingest", help="Convertir des PDF en pages .txt exploitables par le pipeline"
    )
    ingest_cmd.add_argument("source", type=Path, help="Un PDF, ou un repertoire de PDF")
    ingest_cmd.add_argument(
        "--out", type=Path, default=Path("./data/corpus"), help="Repertoire de sortie"
    )
    ingest_cmd.add_argument(
        "--document", default=None, help="Titre du document (deduit du PDF sinon)"
    )
    ingest_cmd.add_argument(
        "--no-lang-dirs",
        action="store_true",
        help="Ne pas ranger par langue dans des sous-repertoires fr/ et ar/",
    )

    serve_cmd = sub.add_parser("serve", help="Lancer l'interface web du chatbot")
    serve_cmd.add_argument("--host", default="127.0.0.1", help="Adresse d'ecoute")
    serve_cmd.add_argument("--port", type=int, default=8000, help="Port d'ecoute")
    serve_cmd.add_argument(
        "--reload", action="store_true", help="Recharger a chaque modification (developpement)"
    )

    compact_cmd = sub.add_parser("compact", help="Fusionner les fragments d'un index existant")
    compact_cmd.add_argument(
        "--remove-parts",
        action="store_true",
        help="Supprimer les fragments apres fusion",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(verbose=args.verbose)

    try:
        settings = load_settings(args.env_file)
    except ConfigError as exc:
        logger.error("Configuration invalide.\n%s", exc)
        return 2

    try:
        if args.command == "index":
            from .pipeline import run

            # Ce job dure de quelques minutes a plusieurs heures : une mise
            # en veille le suspendrait, et le diagnostic serait trompeur.
            with garder_eveille("l'indexation"):
                report = run(
                    settings, limit=args.limit, dry_run=args.dry_run, prune=args.prune
                )
            print(report.summary(), file=sys.stderr)

            if args.compact and not args.dry_run:
                _store(settings).compact()

        elif args.command == "search":
            from .retrieval import Retriever

            with Retriever(settings) as retriever:
                resultat = retriever.search(
                    args.question,
                    top_k=args.top_k,
                    top_n=args.top_n,
                    document_id=args.document,
                    rerank=not args.no_rerank,
                )
            _afficher_recherche(resultat, en_json=args.json)

        elif args.command == "ask":
            from .generation import Generator
            from .retrieval import Retriever

            with Retriever(settings) as retriever:
                resultat = retriever.search(
                    args.question,
                    top_k=args.top_k,
                    top_n=args.top_n,
                    document_id=args.document,
                    rerank=not args.no_rerank,
                )
            reponse = Generator(settings).answer(
                args.question, resultat.passages, modele=args.modele
            )
            _afficher_reponse(reponse, en_json=args.json, extraits=args.show_passages)

        elif args.command == "ingest":
            from .ingestion import ingerer_pdf, ingerer_repertoire

            par_langue = not args.no_lang_dirs
            if args.source.is_dir():
                rapports = ingerer_repertoire(args.source, args.out, par_langue=par_langue)
            else:
                rapports = [
                    ingerer_pdf(
                        args.source, args.out, document=args.document, par_langue=par_langue
                    )
                ]
            _afficher_ingestion(rapports, args.out)

        elif args.command == "serve":
            import uvicorn

            print(
                f"Interface disponible sur http://{args.host}:{args.port}",
                file=sys.stderr,
            )
            uvicorn.run(
                "legalrag.web.server:app",
                host=args.host,
                port=args.port,
                reload=args.reload,
                log_level="warning",
            )

        elif args.command == "qdrant":
            from .pipeline import push_to_qdrant

            with garder_eveille("le chargement dans Qdrant"):
                rapport = push_to_qdrant(settings, recreate=args.recreate, prune=args.prune)
            print(rapport.summary(), file=sys.stderr)

        elif args.command == "compact":
            _store(settings).compact(remove_parts=args.remove_parts)

    except _erreurs_indexation() as exc:
        logger.error("Traitement impossible.\n%s", exc)
        return 2
    except GenerationError as exc:
        logger.error("Generation impossible.\n%s", exc)
        return 2
    except ImportError as exc:
        # L'installation par defaut ne couvre que le serveur web. Une trace
        # brute laisserait croire a un bug ; c'est une dependance absente.
        message = _message_extra_manquant(exc, args.command)
        if message is None:
            logger.exception("Echec du job : %s", exc)
            return 1
        logger.error("%s", message)
        return 2
    except KeyboardInterrupt:
        # Les lots deja ecrits sont valides : la reprise repartira d'eux.
        logger.warning("Interrompu. Relancez la meme commande pour reprendre ou vous en etiez.")
        return 130
    except Exception as exc:  # noqa: BLE001 - dernier filet avant la sortie
        logger.exception("Echec du job : %s", exc)
        return 1

    return 0


def _afficher_recherche(resultat, en_json: bool = False) -> None:
    """Rend le resultat lisible en terminal, ou exploitable en JSON."""
    if en_json:
        print(json.dumps(resultat.as_dict(), ensure_ascii=False, indent=2))
        return

    entete = (
        f"{len(resultat.passages)} passage(s) | "
        f"{resultat.chunks_examined} chunks -> {resultat.passages_examined} passages | "
        f"{'reclasses' if resultat.reranked else 'ordre vectoriel'} | "
        f"{resultat.tokens_billed} tokens"
    )
    print(f"\n{resultat.query}\n{entete}\n" + "-" * 78)

    for rang, p in enumerate(resultat.passages, 1):
        texte = " ".join(p.text.split())
        print(f"\n{rang}. [{p.score:.3f}] {p.citation()}")
        print(f"   {texte}")


def _afficher_ingestion(rapports, destination) -> None:
    """Recapitule ce que l'ingestion a produit, et ce qui bloque."""
    par_langue: dict[str, int] = {}
    for rapport in rapports:
        if rapport.pages_ecrites:
            par_langue[rapport.langue] = par_langue.get(rapport.langue, 0) + rapport.pages_ecrites

    ocr = [r for r in rapports if r.ocr_necessaire]

    print("----- Ingestion -----", file=sys.stderr)
    print(f"PDF traites          : {len(rapports)}", file=sys.stderr)
    print(f"Pages ecrites        : {sum(par_langue.values())}", file=sys.stderr)
    for langue, nombre in sorted(par_langue.items()):
        print(f"  {langue:14} : {nombre}", file=sys.stderr)
    print(f"Destination          : {destination}", file=sys.stderr)

    if ocr:
        # Ces documents sont des images : sans OCR, ils resteraient des
        # pages vides dans l'index.
        print(f"\nPDF image, OCR necessaire ({len(ocr)}) :", file=sys.stderr)
        for rapport in ocr:
            print(f"  {rapport.document}", file=sys.stderr)

    print(
        "\nPour indexer : CORPUS_SOURCE=local et LOCAL_CORPUS_DIR="
        f"{destination}/fr dans .env, puis `legalrag index`.",
        file=sys.stderr,
    )


def _afficher_reponse(reponse, en_json: bool = False, extraits: bool = False) -> None:
    """Affiche la reponse, ses sources, et les reserves eventuelles."""
    if en_json:
        print(json.dumps(reponse.as_dict(), ensure_ascii=False, indent=2))
        return

    print(f"\n{reponse.question}\n" + "=" * 78)
    print(f"\n{reponse.text}\n")

    if reponse.sources():
        print("Sources :")
        for source in reponse.sources():
            print(f"  {source}")

    # Les reserves comptent autant que la reponse : sur du juridique, une
    # affirmation non rattachee a un texte ne vaut rien.
    if reponse.invalid_citations:
        print(
            f"\n/!\\ Le modele cite des extraits inexistants {list(reponse.invalid_citations)} "
            f"alors que {len(reponse.passages)} lui ont ete fournis. Reponse a verifier."
        )
    elif not reponse.cited and not reponse.declined:
        print("\n/!\\ Aucune citation : rien ne rattache cette reponse aux extraits.")

    if extraits:
        print("\n" + "-" * 78)
        for i, passage in enumerate(reponse.passages, 1):
            print(f"\n[{i}] {passage.citation()}")
            print("    " + " ".join(passage.text.split()))

    print(
        f"\n({reponse.model} | {reponse.prompt_tokens} tokens en entree, "
        f"{reponse.completion_tokens} en sortie)"
    )


# Distributions apportees par l'extra `indexation`, indexees par le module
# qu'elles installent : sert a nommer le paquet manquant dans le message.
_PAQUETS_INDEXATION = {
    "googleapiclient": "google-api-python-client",
    "google": "google-auth",
    "pyarrow": "pyarrow",
    "pypdf": "pypdf",
    "voyageai": "voyageai",
}


def _message_extra_manquant(exc: ImportError, commande: str | None) -> str | None:
    """Transforme un ImportError en consigne d'installation.

    Les commandes hors ligne (`index`, `qdrant`, `ingest`, `compact`) tirent
    des bibliotheques lourdes que l'image web n'embarque pas. Le message
    doit dire quoi installer, pas afficher une pile d'appels.

    Renvoie None si l'import rate pour une autre raison : l'appelant traite
    alors l'erreur comme un bug ordinaire, trace comprise.
    """
    racine = (exc.name or "").split(".")[0]
    paquet = _PAQUETS_INDEXATION.get(racine)
    if paquet is None:
        return None

    return (
        f"La commande '{commande}' demande {paquet}, absent de cette installation.\n"
        "Seules les dependances du serveur web sont installees par defaut.\n"
        "Ajoutez celles de l'indexation :\n"
        "    uv sync --extra indexation\n"
        "  ou : pip install 'legalrag[indexation]'"
    )


def _erreurs_indexation() -> tuple[type[Exception], ...]:
    """Erreurs des modules d'indexation, chargees seulement si presents.

    Une image qui ne sert que le web n'installe pas ces dependances :
    les importer ici ferait echouer le demarrage.
    """
    try:
        from .index_store import IndexError_
        from .qdrant_store import QdrantStoreError
    except ImportError:
        return ()
    return (IndexError_, QdrantStoreError)


def _store(settings):
    from .index_store import IndexStore, make_manifest

    return IndexStore(
        settings.index_dir,
        make_manifest(
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
            mode=settings.embedding_mode,
            chunk_tokens=settings.chunk_tokens,
            overlap=settings.chunk_overlap_tokens,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
