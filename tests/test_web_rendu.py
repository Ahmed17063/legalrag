"""Tests du rendu Markdown de la page web.

Le modele de generation repond en Markdown : gras, puces, listes
numerotees. La page doit le rendre, et le faire sans jamais laisser
passer de balise HTML — l'echappement precede le formatage, et cet ordre
est la seule chose qui empeche une injection.

Le code teste est du JavaScript embarque dans la page. Il est extrait et
execute par Node, plutot que reecrit en Python : tester une traduction ne
prouverait rien sur ce qui tourne reellement dans le navigateur.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[1] / "src" / "legalrag" / "web" / "static" / "index.html"

# `echapper` s'appuie sur le DOM, absent sous Node : on lui substitue un
# equivalent strict pour les tests.
ECHAPPER_NODE = """function echapper(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}"""


def extraire_javascript() -> str:
    """Recupere les fonctions de rendu telles qu'elles vivent dans la page."""
    html = PAGE.read_text(encoding="utf-8")
    debut = html.index("function echapper")
    fin = html.index("function ajouterQuestion")
    code = html[debut:fin]
    return re.sub(r"function echapper.*?\n}", ECHAPPER_NODE, code, flags=re.S)


@pytest.fixture(scope="module")
def rendre(tmp_path_factory):
    """Renvoie un appelable qui rend du Markdown via le vrai code de la page."""
    if shutil.which("node") is None:
        pytest.skip("Node n'est pas disponible : rendu JavaScript non testable")

    dossier = tmp_path_factory.mktemp("rendu")
    (dossier / "rendu.mjs").write_text(
        extraire_javascript() + "\nexport { rendreMarkdown };", encoding="utf-8"
    )
    (dossier / "executer.mjs").write_text(
        "import { rendreMarkdown } from './rendu.mjs';\n"
        "process.stdout.write(rendreMarkdown(JSON.parse(process.argv[2])));\n",
        encoding="utf-8",
    )

    def appeler(texte: str) -> str:
        resultat = subprocess.run(
            ["node", str(dossier / "executer.mjs"), json.dumps(texte)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        assert resultat.returncode == 0, resultat.stderr
        return resultat.stdout

    return appeler


# ---------------------------------------------------------------------------
# Le cas signale par l'utilisateur
# ---------------------------------------------------------------------------


def test_puces_et_gras_rendus(rendre):
    """Les asterisques apparaissaient tels quels dans la page."""
    texte = (
        "* **Vol repondant aux conditions de l'article 351** : amputation [1].\n"
        "\n"
        "* **Vol en lieux publics** : peine de travaux forces [2]."
    )
    rendu = rendre(texte)

    assert "<ul>" in rendu and rendu.count("<li>") == 2
    assert "<strong>Vol repondant" in rendu
    assert "**" not in rendu


def test_marqueurs_de_citation_conserves(rendre):
    rendu = rendre("Selon l'article 351 [1] et l'article 360 [2].")
    assert '<span class="renvoi">1</span>' in rendu
    assert '<span class="renvoi">2</span>' in rendu


# ---------------------------------------------------------------------------
# Blocs
# ---------------------------------------------------------------------------


def test_liste_numerotee(rendre):
    rendu = rendre("1. Premier point\n2. Second point")
    assert rendu.startswith("<ol>")
    assert rendu.count("<li>") == 2


def test_paragraphes_separes(rendre):
    assert rendre("Un paragraphe.\n\nUn autre.").count("<p>") == 2


def test_liste_fermee_avant_le_paragraphe_suivant(rendre):
    rendu = rendre("Introduction :\n\n* un\n* deux\n\nConclusion.")
    assert "</ul>" in rendu
    assert rendu.index("</ul>") < rendu.index("Conclusion")


def test_italique(rendre):
    assert "<em>important</em>" in rendre("Le mot *important* ici.")


def test_texte_simple_reste_lisible(rendre):
    rendu = rendre("Une reponse sans aucune mise en forme.")
    assert rendu == "<p>Une reponse sans aucune mise en forme.</p>"


# ---------------------------------------------------------------------------
# Tableaux
# ---------------------------------------------------------------------------

TABLEAU = (
    "**Sanctions encourues**\n"
    "\n"
    "| Infraction | Sanction prevue |\n"
    "|------------|-----------------|\n"
    "| **Infraction tres grave** (Art. 84) | - 100 000 a 800 000 Ouguiya <br> - Confiscation [1] |\n"
    "| Navire etranger | - 1 000 000 Ouguiya |\n"
    "\n"
    "Suite du texte."
)


def test_tableau_rendu(rendre):
    """Les barres verticales apparaissaient telles quelles dans la page."""
    rendu = rendre(TABLEAU)
    assert "<thead><tr><th>Infraction</th><th>Sanction prevue</th></tr></thead>" in rendu
    assert rendu.count("<tr>") == 3
    assert "<td><strong>Infraction tres grave</strong> (Art. 84)</td>" in rendu
    assert "|" not in rendu
    assert "---" not in rendu
    assert "<p>Suite du texte.</p>" in rendu


def test_saut_de_ligne_dans_une_cellule(rendre):
    rendu = rendre(TABLEAU)
    assert "800 000 Ouguiya <br> - Confiscation" in rendu
    assert "&lt;br" not in rendu
    assert '<span class="renvoi">1</span>' in rendu


def test_tableau_sans_ligne_de_tirets(rendre):
    rendu = rendre("| a | b |\n| c | d |")
    assert "<thead>" not in rendu
    assert rendu.count("<td>") == 4


def test_balise_dans_une_cellule_echappee(rendre):
    rendu = rendre("| a |\n|---|\n| <img src=x onerror=alert(1)> <br onload=x> |")
    assert "<img" not in rendu
    assert "<br onload" not in rendu


# ---------------------------------------------------------------------------
# Securite : le modele ne doit jamais pouvoir injecter de balise
# ---------------------------------------------------------------------------


def test_html_du_modele_echappe(rendre):
    rendu = rendre("<img src=x onerror=alert(1)> et **gras**")
    assert "<img" not in rendu
    assert "&lt;img" in rendu
    # Le formatage legitime continue de fonctionner.
    assert "<strong>gras</strong>" in rendu


def test_script_neutralise(rendre):
    rendu = rendre("* <script>alert(1)</script>")
    assert "<script>" not in rendu
    assert "&lt;script&gt;" in rendu


def test_attribut_injecte_neutralise(rendre):
    rendu = rendre('**gras** <a href="javascript:alert(1)">lien</a>')
    assert "<a href" not in rendu
    assert "&lt;a href=" in rendu
