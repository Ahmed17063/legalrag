# Image de l'interface web du chatbot juridique.
#
# Le serveur interroge Qdrant Cloud, Voyage et Groq par le reseau : il n'a
# besoin ni de l'index Parquet, ni du corpus, ni du cache de telechargement.
# Seuls `data/statuts.json` et `data/perimetre.json` sont embarques : petits,
# et ils portent la qualification juridique des textes et le perimetre
# interrogeable.
#
# Construction en deux etapes : les dependances sont installees dans une
# image qui contient uv et les outils de compilation, puis seul le
# `.venv` resultant passe dans l'image finale. Ni uv ni les sources de
# construction ne s'y retrouvent.

# --------------------------------------------------------------------------
# Etape 1 : dependances
# --------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS build

# Copier les octets plutot que des liens : l'image finale n'aura pas le
# cache de uv, et un lien pointerait dans le vide.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Les manifestes d'abord, le code ensuite : tant que les dependances ne
# changent pas, Docker reutilise cette couche et l'installation est sautee.
COPY pyproject.toml uv.lock ./

# --no-install-project : le projet lui-meme viendra apres, pour ne pas
# invalider le cache des dependances a chaque modification du code.
#
# Aucun --extra : l'extra `indexation` (pyarrow, pypdf, google-*, voyageai)
# ne sert qu'au job hors ligne et n'a rien a faire ici. Ne pas l'ajouter :
# le serveur n'en importe rien, et tests/test_dependances.py le verifie.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
COPY README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# --------------------------------------------------------------------------
# Etape 2 : execution
# --------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

# UTF-8 partout : la typographie francaise (espace fine insecable, guillemets)
# fait echouer un print sur une sortie en ASCII.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    LANG=C.UTF-8 \
    PATH="/app/.venv/bin:$PATH"

# curl sert au HEALTHCHECK ci-dessous, et a rien d'autre.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Un compte sans privilege : le service n'a aucune raison d'etre root.
RUN useradd --create-home --uid 10001 legalrag

WORKDIR /app

COPY --from=build --chown=legalrag:legalrag /app/.venv /app/.venv
COPY --chown=legalrag:legalrag src/ ./src/
# Registre des statuts juridiques : petit, et necessaire pour signaler les
# textes abroges. Montez-le en volume pour le modifier sans reconstruire.
COPY --chown=legalrag:legalrag data/statuts.json ./data/statuts.json
# Perimetre d'interrogation. Montez-le en volume pour l'elargir sans
# reconstruire l'image.
COPY --chown=legalrag:legalrag data/perimetre.json ./data/perimetre.json

USER legalrag

EXPOSE 8000

# Le conteneur repond a l'exterieur : 127.0.0.1, le defaut de la commande,
# ne serait joignable que depuis le conteneur lui-meme.
ENV LEGALRAG_HOST=0.0.0.0 \
    LEGALRAG_PORT=8000

# La sonde interroge le point de sante, qui verifie la connexion a Qdrant.
# start-period genereux : l'ouverture de la collection prend quelques
# secondes au demarrage.
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${LEGALRAG_PORT}/api/health" \
        | grep -q '"pret": *true' || exit 1

# Aucun secret n'est embarque : VOYAGE_API_KEY, GROQ_API_KEY, QDRANT_URL et
# QDRANT_API_KEY sont fournis a l'execution, par --env-file ou -e.
CMD ["sh", "-c", "legalrag serve --host \"$LEGALRAG_HOST\" --port \"$LEGALRAG_PORT\""]
