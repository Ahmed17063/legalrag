"""Comportement de la ligne de commande hors execution reelle des jobs.

L'installation par defaut ne contient que les dependances du serveur web :
c'est ce que l'image Docker embarque. Les commandes hors ligne (`index`,
`qdrant`, `ingest`, `compact`) tirent alors des bibliotheques absentes, et
l'utilisateur doit lire une consigne d'installation, pas une pile d'appels.
"""

from __future__ import annotations

import pytest

from legalrag.cli import _message_extra_manquant


@pytest.mark.parametrize(
    ("module", "paquet"),
    [
        ("google.oauth2", "google-auth"),
        ("googleapiclient.discovery", "google-api-python-client"),
        ("pyarrow", "pyarrow"),
        ("pypdf", "pypdf"),
        ("voyageai", "voyageai"),
    ],
)
def test_un_module_d_indexation_absent_donne_la_commande_d_installation(module, paquet):
    message = _message_extra_manquant(ModuleNotFoundError(name=module), "index")

    assert message is not None
    assert paquet in message
    assert "uv sync --extra indexation" in message
    # La commande fautive est nommee : l'utilisateur sait quoi relancer.
    assert "'index'" in message


def test_un_import_sans_rapport_reste_un_bug():
    """Ne pas masquer une vraie erreur derriere un message d'installation."""
    assert _message_extra_manquant(ModuleNotFoundError(name="fastapi"), "serve") is None


def test_un_import_error_sans_nom_de_module_reste_un_bug():
    """`ImportError` nu (symbole introuvable) n'a pas d'attribut name utile."""
    assert _message_extra_manquant(ImportError("cannot import name X"), "index") is None


def test_les_erreurs_d_indexation_sont_ignorees_si_les_modules_manquent():
    """`_erreurs_indexation` sert de garde : elle ne doit jamais lever.

    Dans l'image web, `index_store` et `qdrant_store` sont importables mais
    leurs dependances non : la fonction renvoie alors un tuple vide, et
    `except ()` n'attrape rien — ce qui est le comportement voulu.
    """
    from legalrag.cli import _erreurs_indexation

    erreurs = _erreurs_indexation()
    assert isinstance(erreurs, tuple)
    assert all(issubclass(e, Exception) for e in erreurs)
