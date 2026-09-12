"""Tests de la resolution de l'endpoint Voyage.

Voyage AI appartient a MongoDB : selon l'endroit ou la cle a ete creee,
l'API ne repond pas sur le meme domaine. Se tromper d'endpoint produit un
echec d'authentification difficile a diagnostiquer.
"""

from __future__ import annotations

from pathlib import Path

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


# ---------------------------------------------------------------------------
# Les acces au corpus ne sont exiges qu'a l'indexation
# ---------------------------------------------------------------------------


def test_configuration_chargee_sans_acces_drive(monkeypatch, tmp_path):
    """Servir le chatbot ne demande ni Drive ni corpus local.

    Les exiger au chargement faisait echouer `serve`, `search` et `ask`,
    qui interrogent Qdrant par le reseau — et empechait le conteneur
    Docker de demarrer, faute de fichier de credentials.
    """
    from legalrag.config import load_settings

    for nom, valeur in {
        "VOYAGE_API_KEY": "cle",
        "GOOGLE_APPLICATION_CREDENTIALS": "",
        "DRIVE_FOLDER_ID": "",
        "CORPUS_SOURCE": "drive",
    }.items():
        monkeypatch.setenv(nom, valeur)

    settings = load_settings(tmp_path / "absent.env")
    assert settings.corpus_source == "drive"


def test_indexation_refusee_sans_acces_drive(settings):
    """La verification a lieu au moment de construire le corpus."""
    from dataclasses import replace

    from legalrag.config import ConfigError, verifier_acces_corpus

    sans_drive = replace(settings, drive_folder_id="", google_credentials_file=Path())
    with pytest.raises(ConfigError, match="DRIVE_FOLDER_ID"):
        verifier_acces_corpus(sans_drive)


def test_indexation_refusee_si_credentials_introuvables(settings, tmp_path):
    from dataclasses import replace

    from legalrag.config import ConfigError, verifier_acces_corpus

    mauvais = replace(settings, google_credentials_file=tmp_path / "rien.json")
    with pytest.raises(ConfigError, match="introuvable"):
        verifier_acces_corpus(mauvais)


def test_corpus_local_verifie_le_repertoire(settings, tmp_path):
    from dataclasses import replace

    from legalrag.config import ConfigError, verifier_acces_corpus

    absent = replace(settings, corpus_source="local", local_corpus_dir=tmp_path / "rien")
    with pytest.raises(ConfigError, match="Repertoire de corpus introuvable"):
        verifier_acces_corpus(absent)

    present = replace(settings, corpus_source="local", local_corpus_dir=tmp_path)
    verifier_acces_corpus(present)  # ne doit pas lever


# ---------------------------------------------------------------------------
# Deuxieme fournisseur de generation
# ---------------------------------------------------------------------------


def environnement_minimal(monkeypatch):
    """De quoi charger la configuration sans acces au corpus."""
    for nom, valeur in {
        "VOYAGE_API_KEY": "cle",
        "GOOGLE_APPLICATION_CREDENTIALS": "",
        "DRIVE_FOLDER_ID": "",
        "CORPUS_SOURCE": "drive",
    }.items():
        monkeypatch.setenv(nom, valeur)


def test_anthropic_absent_par_defaut(monkeypatch, tmp_path):
    """Une installation existante continue de fonctionner avec Groq seul."""
    from legalrag.config import load_settings

    environnement_minimal(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    settings = load_settings(tmp_path / "absent.env")
    assert settings.anthropic_api_key is None
    assert settings.anthropic_model == "claude-sonnet-5"
    assert settings.generation_default == "groq"


def test_anthropic_lu_depuis_l_environnement(monkeypatch, tmp_path):
    from legalrag.config import load_settings

    environnement_minimal(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-de-test")
    monkeypatch.setenv("ANTHROPIC_EFFORT", "medium")
    monkeypatch.setenv("GENERATION_DEFAULT", "claude")

    settings = load_settings(tmp_path / "absent.env")
    assert settings.anthropic_api_key == "sk-ant-de-test"
    assert settings.anthropic_effort == "medium"
    assert settings.generation_default == "claude"


def test_effort_inconnu_refuse(monkeypatch, tmp_path):
    """L'API rejette une valeur inconnue : autant le dire au chargement."""
    from legalrag.config import ConfigError, load_settings

    environnement_minimal(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_EFFORT", "maximal")

    with pytest.raises(ConfigError, match="ANTHROPIC_EFFORT"):
        load_settings(tmp_path / "absent.env")


def test_temperature_par_defaut(monkeypatch, tmp_path):
    """Bridee, mais pas nulle."""
    from legalrag.config import load_settings

    environnement_minimal(monkeypatch)
    monkeypatch.delenv("GENERATION_TEMPERATURE", raising=False)

    assert load_settings(tmp_path / "absent.env").generation_temperature == 0.3


def test_temperature_lue_depuis_l_environnement(monkeypatch, tmp_path):
    from legalrag.config import load_settings

    environnement_minimal(monkeypatch)
    monkeypatch.setenv("GENERATION_TEMPERATURE", "0.7")

    assert load_settings(tmp_path / "absent.env").generation_temperature == 0.7


@pytest.mark.parametrize("valeur", ["-0.1", "2.5"])
def test_temperature_hors_bornes_refusee(monkeypatch, tmp_path, valeur):
    from legalrag.config import ConfigError, load_settings

    environnement_minimal(monkeypatch)
    monkeypatch.setenv("GENERATION_TEMPERATURE", valeur)

    with pytest.raises(ConfigError, match="GENERATION_TEMPERATURE"):
        load_settings(tmp_path / "absent.env")


def test_temperature_non_numerique_refusee(monkeypatch, tmp_path):
    from legalrag.config import ConfigError, load_settings

    environnement_minimal(monkeypatch)
    monkeypatch.setenv("GENERATION_TEMPERATURE", "tiede")

    with pytest.raises(ConfigError, match="GENERATION_TEMPERATURE"):
        load_settings(tmp_path / "absent.env")
