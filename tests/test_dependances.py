"""Les dependances declarees correspondent-elles au code livre ?

Ce projet a remplace une implementation precedente, archivee sous `old/`.
Rien de ses dependances ne doit reapparaitre : la liste de `pyproject.toml`
doit se deduire strictement des imports du code actuel.

Les dependances sont scindees en deux groupes, et la frontiere est ce que
ce fichier surveille :

* le groupe principal est celui qu'embarque l'image Docker. Il ne doit
  contenir que ce dont le serveur web a besoin pour repondre : il interroge
  Qdrant, Voyage et Groq par le reseau et ne lit ni Drive, ni PDF, ni index
  Parquet ;
* l'extra `indexation` porte les bibliotheques lourdes du job hors ligne
  (Drive, PDF, Parquet, SDK Voyage). Elles ne doivent jamais etre atteintes
  a l'import du serveur, sans quoi l'image devrait les embarquer.

Trois derives sont surveillees :

* une dependance declaree que personne n'importe alourdit l'image Docker
  et la surface a maintenir ;
* un module importe sans etre declare fonctionne tant qu'une autre
  dependance l'entraine, puis casse le jour ou celle-ci change. C'est
  arrive avec `pydantic`, tire par FastAPI ;
* une dependance d'indexation atteinte depuis le serveur ferait echouer
  le conteneur, ou obligerait a la reintegrer au groupe principal.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[1]
SOURCES = RACINE / "src"
PAQUET = SOURCES / "legalrag"

# Nom de distribution sur PyPI -> nom du module a importer.
EQUIVALENCES = {
    "google-api-python-client": "googleapiclient",
    "google-auth": "google",
    "python-dotenv": "dotenv",
    "qdrant-client": "qdrant_client",
}

# Ce que le conteneur execute : `legalrag serve` passe par la CLI, qui
# expose le serveur. Tout ce que ces deux modules chargent doit tenir dans
# le groupe principal.
ENTREES_WEB = ("legalrag.cli", "legalrag.web.server")

# Modules que l'extra `indexation` apporte, et qui ne doivent pas etre
# charges par le serveur. `google` seul serait ambigu : `google.protobuf`
# vient de qdrant-client, pas de google-auth.
MODULES_INDEXATION = (
    "googleapiclient",
    "google.auth",
    "google.oauth2",
    "pyarrow",
    "pypdf",
    "voyageai",
)


def _tiers(modules: set[str]) -> set[str]:
    return modules - set(sys.stdlib_module_names) - {"legalrag"}


def _imports_du_fichier(fichier: Path, *, niveau_module_seul: bool) -> set[str]:
    """Modules importes par un fichier.

    `niveau_module_seul` ne retient que les imports executes au chargement
    du module. Les imports differes dans un corps de fonction -- ceux que la
    CLI utilise pour ne pas tirer l'indexation -- sont alors ignores, ce qui
    est exactement ce que fait l'interpreteur.
    """
    arbre = ast.parse(fichier.read_text(encoding="utf-8"))
    noeuds = arbre.body if niveau_module_seul else list(ast.walk(arbre))

    trouves: set[str] = set()
    for noeud in noeuds:
        if isinstance(noeud, ast.Import):
            trouves |= {alias.name.split(".")[0] for alias in noeud.names}
        elif isinstance(noeud, ast.ImportFrom) and noeud.level == 0 and noeud.module:
            trouves.add(noeud.module.split(".")[0])
    return trouves


def modules_importes() -> set[str]:
    """Modules tiers importes par le code livre, imports differes compris."""
    trouves: set[str] = set()
    for fichier in SOURCES.rglob("*.py"):
        trouves |= _imports_du_fichier(fichier, niveau_module_seul=False)
    return _tiers(trouves)


def _fichier_du_module(nom: str) -> Path | None:
    """`legalrag.web.server` -> src/legalrag/web/server.py."""
    relatif = Path(*nom.split(".")[1:])
    candidats = (PAQUET / relatif.with_suffix(".py"), PAQUET / relatif / "__init__.py")
    for candidat in candidats:
        if candidat.is_file():
            return candidat
    return None


def _relatifs(fichier: Path, nom: str) -> set[str]:
    """Modules `legalrag.*` importes au chargement de `fichier`."""
    arbre = ast.parse(fichier.read_text(encoding="utf-8"))
    paquet_courant = nom if fichier.name == "__init__.py" else nom.rsplit(".", 1)[0]

    voisins: set[str] = set()
    for noeud in arbre.body:
        if not isinstance(noeud, ast.ImportFrom) or noeud.level == 0:
            continue
        base = paquet_courant.split(".")
        remontee = noeud.level - 1
        prefixe = base[: len(base) - remontee] if remontee else base
        if noeud.module:
            voisins.add(".".join([*prefixe, noeud.module]))
        else:
            # `from . import x` : chaque nom importe est un sous-module.
            voisins |= {".".join([*prefixe, alias.name]) for alias in noeud.names}
    return voisins


def fermeture_web() -> tuple[set[str], set[str]]:
    """Modules legalrag charges a l'import du serveur, et leurs dependances."""
    a_voir = list(ENTREES_WEB)
    vus: set[str] = set()
    tiers: set[str] = set()

    while a_voir:
        nom = a_voir.pop()
        if nom in vus:
            continue
        vus.add(nom)
        fichier = _fichier_du_module(nom)
        if fichier is None:
            continue
        tiers |= _imports_du_fichier(fichier, niveau_module_seul=True)
        a_voir.extend(_relatifs(fichier, nom))

    return vus, _tiers(tiers)


def _manifeste() -> dict:
    return tomllib.loads((RACINE / "pyproject.toml").read_text(encoding="utf-8"))


def dependances_declarees(extra: str | None = None) -> set[str]:
    projet = _manifeste()["project"]
    lignes = (
        projet["dependencies"]
        if extra is None
        else projet["optional-dependencies"][extra]
    )
    return {re.split(r"[><=\[]", ligne)[0].strip() for ligne in lignes}


def toutes_les_dependances() -> set[str]:
    return dependances_declarees() | dependances_declarees("indexation")


def modules_attendus(noms: set[str]) -> set[str]:
    return {EQUIVALENCES.get(nom, nom.replace("-", "_")) for nom in noms}


def test_aucune_dependance_inutilisee():
    """Une dependance que personne n'importe alourdit l'image pour rien."""
    inutilisees = modules_attendus(toutes_les_dependances()) - modules_importes()
    assert not inutilisees, (
        f"declarees mais jamais importees : {sorted(inutilisees)}. "
        "Les retirer de pyproject.toml, ou verifier la table EQUIVALENCES."
    )


def test_aucune_dependance_implicite():
    """S'appuyer sur une transitive casse le jour ou elle change."""
    implicites = modules_importes() - modules_attendus(toutes_les_dependances())
    assert not implicites, (
        f"importees sans etre declarees : {sorted(implicites)}. "
        "Les ajouter a pyproject.toml plutot que compter sur une transitive."
    )


def test_le_groupe_principal_couvre_le_serveur():
    """Le serveur doit demarrer sans l'extra `indexation`.

    C'est ce que fait l'image Docker : elle installe le groupe principal
    seul. Un import du serveur qui n'y figurerait pas casserait le
    conteneur au demarrage.
    """
    _, requis = fermeture_web()

    # `dotenv` est charge paresseusement par config._load_dotenv : il ne
    # figure pas dans la fermeture statique, mais le serveur en depend.
    requis.add("dotenv")

    manquants = requis - modules_attendus(dependances_declarees())
    assert not manquants, (
        f"charges par le serveur mais absents du groupe principal : {sorted(manquants)}. "
        "Les deplacer de l'extra 'indexation' vers project.dependencies."
    )


def test_l_indexation_reste_hors_du_chemin_du_serveur():
    """Aucune bibliotheque lourde ne doit etre atteinte a l'import.

    Verification dynamique plutot que statique : c'est l'interpreteur qui
    tranche. `google` seul ne suffirait pas comme sonde, `google.protobuf`
    arrivant par qdrant-client.
    """
    programme = "\n".join(
        (
            "import sys, importlib",
            f"for nom in {ENTREES_WEB!r}:",
            "    importlib.import_module(nom)",
            f"charges = [m for m in {MODULES_INDEXATION!r} if m in sys.modules]",
            "print(','.join(charges))",
        )
    )
    resultat = subprocess.run(
        [sys.executable, "-c", programme],
        capture_output=True,
        text=True,
        cwd=RACINE,
        check=True,
    )
    charges = [nom for nom in resultat.stdout.strip().split(",") if nom]
    assert not charges, (
        f"l'import du serveur charge des dependances d'indexation : {charges}. "
        "Les importer dans le corps de la fonction qui les utilise."
    )


def test_les_deux_groupes_sont_disjoints():
    """Une dependance presente dans les deux listes serait installee deux fois."""
    commun = dependances_declarees() & dependances_declarees("indexation")
    assert not commun, f"declarees dans les deux groupes : {sorted(commun)}"


def test_le_groupe_de_developpement_tire_l_indexation():
    """Les tests couvrent aussi le job hors ligne : il leur faut l'extra."""
    dev = _manifeste()["dependency-groups"]["dev"]
    assert any("legalrag[indexation]" in str(ligne) for ligne in dev), (
        "le groupe dev doit inclure legalrag[indexation], sinon les tests "
        "du pipeline d'indexation ne peuvent pas s'executer."
    )


def test_le_dockerfile_n_installe_pas_l_indexation():
    """L'image ne sert que le web : l'extra n'a rien a y faire."""
    dockerfile = (RACINE / "Dockerfile").read_text(encoding="utf-8")
    for ligne in dockerfile.splitlines():
        if "uv sync" in ligne:
            assert "--extra" not in ligne and "--all-extras" not in ligne, (
                f"l'image installerait l'extra indexation : {ligne.strip()}"
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
    assert not any(interdit in nom for nom in toutes_les_dependances())
