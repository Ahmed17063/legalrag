"""Tests de la resolution de l'endpoint Voyage.

Voyage AI appartient a MongoDB : selon l'endroit ou la cle a ete creee,
l'API ne repond pas sur le meme domaine. Se tromper d'endpoint produit un
echec d'authentification difficile a diagnostiquer.
"""

from __future__ import annotations

import pytest

from legalrag.config import (
    VOYAGE_ATLAS_EU_URL,
    VOYAGE_ATLAS_URL,
    VOYAGE_ATLAS_US_URL,
    VOYAGE_LEGACY_URL,
    resolve_base_url,
)


def test_cle_ancienne_plateforme_voyage():
    assert resolve_base_url("pa-abcdef123456") == VOYAGE_LEGACY_URL


def test_cle_atlas_sans_restriction():
    """Une cle Atlas non limitee n'a pas de prefixe distinctif."""
    assert resolve_base_url("abcdef123456") == VOYAGE_ATLAS_URL


@pytest.mark.parametrize(
    ("cle", "attendu"),
    [
        ("al-eu-abcdef", VOYAGE_ATLAS_EU_URL),
        ("al-us-abcdef", VOYAGE_ATLAS_US_URL),
    ],
)
def test_cle_limitee_a_une_geographie(cle, attendu):
    assert resolve_base_url(cle) == attendu


def test_override_explicite_prioritaire():
    assert resolve_base_url("pa-abc", "https://exemple.test/v1") == "https://exemple.test/v1"


def test_override_sans_slash_final():
    assert resolve_base_url("pa-abc", "https://exemple.test/v1/") == "https://exemple.test/v1"


def test_override_vide_ignore():
    assert resolve_base_url("pa-abc", "") == VOYAGE_LEGACY_URL
