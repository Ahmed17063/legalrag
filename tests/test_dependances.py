"""Les dependances declarees correspondent-elles au code livre ?

Ce projet a remplace une implementation precedente, archivee sous `old/`.
Rien de ses dependances ne doit reapparaitre : la liste de `pyproject.toml`
doit se deduire strictement des imports du code actuel.

Deux derives sont surveillees, symetriques :

* une dependance declaree que personne n'importe alourdit l'image Docker
  et la surface a maintenir ;
* un module importe sans etre declare fonctionne tant qu'une autre
  dependance l'entraine, puis casse le jour ou celle-ci change. C'est
  arrive avec `pydantic`, tire par FastAPI.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[1]
SOURCES = RACINE / "src"

# Nom de distribution sur PyPI -> nom du module a importer.
EQUIVALENCES = {
    "google-api-python-client": "googleapiclient",
    "google-auth": "google",
    "python-dotenv": "dotenv",
    "qdrant-client": "qdrant_client",
}


def modules_importes() -> set[str]:
    """Modules tiers effectivement importes par le code livre."""
    trouves: set[str] = set()
    for fichier in SOURCES.rglob("*.py"):
        arbre = ast.parse(fichier.read_text(encoding="utf-8"))
        for noeud in ast.walk(arbre):
            if isinstance(noeud, ast.Import):
                trouves |= {alias.name.split(".")[0] for alias in noeud.names}
            elif isinstance(noeud, ast.ImportFrom) and noeud.level == 0 and noeud.module:
                trouves.add(noeud.module.split(".")[0])

    return trouves - set(sys.stdlib_module_names) - {"legalrag"}


def dependances_declarees() -> set[str]:
    manifeste = tomllib.loads((RACINE / "pyproject.toml").read_text(encoding="utf-8"))
    return {
        re.split(r"[><=\[]", ligne)[0].strip()
        for ligne in manifeste["project"]["dependencies"]
    }


def modules_attendus() -> set[str]:
    return {
        EQUIVALENCES.get(nom, nom.replace("-", "_")) for nom in dependances_declarees()
    }


def test_aucune_dependance_inutilisee():
    """Une dependance que personne n'importe alourdit l'image pour rien."""
    inutilisees = modules_attendus() - modules_importes()
    assert not inutilisees, (
        f"declarees mais jamais importees : {sorted(inutilisees)}. "
        "Les retirer de pyproject.toml, ou verifier la table EQUIVALENCES."
    )


def test_aucune_dependance_implicite():
    """S'appuyer sur une transitive casse le jour ou elle change."""
    implicites = modules_importes() - modules_attendus()
    assert not implicites, (
        f"importees sans etre declarees : {sorted(implicites)}. "
        "Les ajouter a pyproject.toml plutot que compter sur une transitive."
    )


def test_le_dockerfile_ne_copie_que_le_necessaire():
    """L'archive `old/` et les donnees generees restent hors de l'image."""
    dockerfile = (RACINE / "Dockerfile").read_text(encoding="utf-8")
    copies = [
        ligne.split(maxsplit=1)[1]
        for ligne in dockerfile.splitlines()
        if ligne.startswith("COPY")
    ]
    joint = " ".join(copies)

    archive = chr(111) + chr(108) + chr(100)  # nom du repertoire archive
    assert archive + "/" not in joint
    assert "Documents_juridiques" not in joint
    assert "secrets" not in joint
    # Le corpus et l'index ne servent pas au serveur web.
    assert "data/corpus" not in joint
    assert "data/index" not in joint


def test_le_contexte_de_build_exclut_l_archive_et_les_secrets():
    exclusions = {
        ligne.strip()
        for ligne in (RACINE / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if ligne.strip() and not ligne.startswith("#")
    }
    archive = chr(111) + chr(108) + chr(100) + "/"

    for attendu in (archive, ".env", "secrets/", "data/", "Documents_juridiques/", ".venv/"):
        assert attendu in exclusions, f"{attendu} devrait etre exclu du contexte de build"

    # Le registre des statuts, lui, doit rester : le serveur en a besoin.
    assert "!data/statuts.json" in exclusions


@pytest.mark.parametrize("interdit", ["groq", "openai", "langchain", "ollama", "streamlit"])
def test_aucune_dependance_heritee_de_l_ancienne_implementation(interdit):
    """L'ancienne version utilisait d'autres bibliotheques.

    La generation passe par Groq, mais en HTTP direct : aucun SDK n'est
    necessaire, et aucun ne doit reapparaitre par inadvertance.
    """
    assert not any(interdit in nom for nom in dependances_declarees())
