"""Tests de la suspension de veille pendant les traitements longs."""

from __future__ import annotations

import sys

import pytest

from legalrag import veille


def test_etat_retabli_a_la_sortie(monkeypatch):
    """Sans le rappel final, la veille resterait suspendue."""
    appels = []
    monkeypatch.setattr(veille, "_appliquer", lambda d: appels.append(d) or True)

    with veille.garder_eveille("un test") as actif:
        assert actif

    assert len(appels) == 2
    # Premier appel : on demande a rester eveille. Second : on rend la main.
    assert appels[0] == veille._ES_CONTINUOUS | veille._ES_SYSTEM_REQUIRED
    assert appels[1] == veille._ES_CONTINUOUS


def test_etat_retabli_meme_en_cas_d_exception(monkeypatch):
    appels = []
    monkeypatch.setattr(veille, "_appliquer", lambda d: appels.append(d) or True)

    with pytest.raises(RuntimeError):
        with veille.garder_eveille("un test"):
            raise RuntimeError("le job a echoue")

    assert appels[-1] == veille._ES_CONTINUOUS


def test_echec_n_empeche_pas_le_traitement(monkeypatch):
    """Si l'API refuse, le job doit se derouler quand meme."""
    monkeypatch.setattr(veille, "_appliquer", lambda d: False)

    with veille.garder_eveille("un test") as actif:
        assert actif is False


def test_echec_signale_sur_windows(monkeypatch, caplog):
    import logging

    monkeypatch.setattr(veille, "_appliquer", lambda d: False)
    monkeypatch.setattr(veille.sys, "platform", "win32")

    with caplog.at_level(logging.WARNING):
        with veille.garder_eveille("l'indexation"):
            pass

    assert "veille" in caplog.text.lower()


def test_silencieux_hors_windows(monkeypatch, caplog):
    """Sur Linux ou macOS, ce n'est pas une anomalie."""
    import logging

    monkeypatch.setattr(veille, "_appliquer", lambda d: False)
    monkeypatch.setattr(veille.sys, "platform", "linux")

    with caplog.at_level(logging.WARNING):
        with veille.garder_eveille("un test"):
            pass

    assert caplog.text == ""


def test_ecran_non_maintenu_allume():
    """Garder l'ecran allume n'apporte rien a un job de fond."""
    source = (veille.__file__)
    assert not hasattr(veille, "_ES_DISPLAY_REQUIRED")


@pytest.mark.skipif(not sys.platform.startswith("win"), reason="API Windows")
def test_appel_reel_sur_windows():
    """L'API repond bien sur la machine qui execute les tests."""
    with veille.garder_eveille("un test") as actif:
        assert actif is True
