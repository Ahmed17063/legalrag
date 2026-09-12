"""Tests du menu de choix du modele redacteur.

`remplirModeles` decide ce que voit l'utilisateur : quels modeles sont
proposes, lequel est preselectionne, et si le choix survit au
rechargement. La logique est courte mais porte plusieurs regles, et une
erreur y passerait inapercue — elle vit dans le gestionnaire de
`/api/health`, ou une exception ne fait qu'interrompre silencieusement le
remplissage de la page.

Comme pour le rendu Markdown, le JavaScript est extrait de la page et
execute par Node : tester une reecriture en Python ne prouverait rien sur
ce qui tourne dans le navigateur.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[1] / "src" / "legalrag" / "web" / "static" / "index.html"

# Le DOM et `localStorage` sont absents sous Node : on leur substitue le
# minimum que la fonction touche reellement.
STUBS = """
function faireSelect() {
  return { options: [], value: "", title: "", disabled: false, onchange: null,
           appendChild(o) { this.options.push(o); } };
}
let stock = {};
const selectMod = faireSelect();
const blocMod = { hidden: true };
function lireMemoire() { return stock.v ?? null; }
function ecrireMemoire(c) { stock.v = c; }
const document = { createElement: () => ({ value: "", textContent: "", title: "" }) };
"""

MODELES = [
    {"cle": "groq", "libelle": "Groq — gpt-oss", "description": "Rapide."},
    {"cle": "claude", "libelle": "Claude — sonnet-5", "description": "Raisonne."},
]


def extraire_javascript() -> str:
    """Recupere la fonction telle qu'elle vit dans la page."""
    html = PAGE.read_text(encoding="utf-8")
    return html[html.index("function remplirModeles") : html.index("// --- Saisie")]


@pytest.fixture(scope="module")
def executer(tmp_path_factory):
    """Renvoie un appelable qui joue un scenario et rend l'etat final."""
    if shutil.which("node") is None:
        pytest.skip("Node n'est pas disponible : selecteur non testable")

    dossier = tmp_path_factory.mktemp("selecteur")
    script = dossier / "selecteur.mjs"
    script.write_text(
        STUBS
        + extraire_javascript()
        + """
const scenario = JSON.parse(process.argv[2]);
for (const etape of scenario) {
  if (etape.remplir) {
    selectMod.options.length = 0;
    remplirModeles(etape.remplir.modeles, etape.remplir.defaut);
  }
  if (etape.choisir) { selectMod.value = etape.choisir; selectMod.onchange(); }
  if (etape.oublier) { stock = {}; }
}
process.stdout.write(JSON.stringify({
  options: selectMod.options.map(o => o.value),
  libelles: selectMod.options.map(o => o.textContent),
  valeur: selectMod.value,
  titre: selectMod.title,
  desactive: selectMod.disabled,
  cache: blocMod.hidden,
  memoire: stock.v ?? null,
}));
""",
        encoding="utf-8",
    )

    def appeler(scenario: list[dict]) -> dict:
        resultat = subprocess.run(
            ["node", str(script), json.dumps(scenario)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        assert resultat.returncode == 0, resultat.stderr
        return json.loads(resultat.stdout)

    return appeler


def test_les_modeles_sont_proposes(executer):
    etat = executer([{"remplir": {"modeles": MODELES, "defaut": "groq"}}])

    assert etat["options"] == ["groq", "claude"]
    assert etat["libelles"] == ["Groq — gpt-oss", "Claude — sonnet-5"]
    assert etat["cache"] is False


def test_le_defaut_du_serveur_est_preselectionne(executer):
    etat = executer([{"remplir": {"modeles": MODELES, "defaut": "claude"}}])
    assert etat["valeur"] == "claude"
    assert etat["titre"] == "Raisonne."


def test_le_choix_est_memorise(executer):
    etat = executer(
        [{"remplir": {"modeles": MODELES, "defaut": "groq"}}, {"choisir": "claude"}]
    )
    assert etat["memoire"] == "claude"
    assert etat["titre"] == "Raisonne."


def test_le_choix_memorise_prime_au_rechargement(executer):
    """Rechoisir son modele a chaque visite serait agacant."""
    etat = executer(
        [
            {"remplir": {"modeles": MODELES, "defaut": "groq"}},
            {"choisir": "claude"},
            {"remplir": {"modeles": MODELES, "defaut": "groq"}},
        ]
    )
    assert etat["valeur"] == "claude"


def test_un_choix_memorise_devenu_invalide_est_ignore(executer):
    """La cle Anthropic a pu etre retiree depuis la derniere visite."""
    etat = executer(
        [
            {"remplir": {"modeles": MODELES, "defaut": "groq"}},
            {"choisir": "claude"},
            {"remplir": {"modeles": [MODELES[0]], "defaut": "groq"}},
        ]
    )
    assert etat["valeur"] == "groq"


def test_un_seul_modele_desactive_le_menu(executer):
    """Rien a choisir, mais savoir qui redige reste une information."""
    etat = executer([{"remplir": {"modeles": [MODELES[0]], "defaut": "groq"}}])

    assert etat["desactive"] is True
    assert etat["cache"] is False


def test_aucun_modele_laisse_le_menu_cache(executer):
    """Sans generation, le menu n'aurait aucun sens."""
    etat = executer([{"remplir": {"modeles": [], "defaut": None}}])

    assert etat["options"] == []
    assert etat["cache"] is True
