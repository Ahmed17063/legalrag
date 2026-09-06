"""Tests de la configuration des sorties."""

from __future__ import annotations

import io

from legalrag.logging_setup import force_utf8_output, setup_logging


class FluxReconfigurable:
    def __init__(self):
        self.encoding = "cp1252"
        self.errors = "strict"

    def reconfigure(self, encoding=None, errors=None):
        if encoding:
            self.encoding = encoding
        if errors:
            self.errors = errors


class FluxFige:
    """Un flux qui refuse d'etre reconfigure, comme une redirection."""

    def reconfigure(self, **kwargs):
        raise ValueError("flux non reconfigurable")


def test_bascule_en_utf8(monkeypatch):
    import sys

    flux = FluxReconfigurable()
    monkeypatch.setattr(sys, "stdout", flux)
    monkeypatch.setattr(sys, "stderr", FluxReconfigurable())

    force_utf8_output()

    assert flux.encoding == "utf-8"
    assert flux.errors == "replace"


def test_flux_non_reconfigurable_ignore(monkeypatch):
    """Une redirection qui refuse ne doit pas faire echouer la commande."""
    import sys

    monkeypatch.setattr(sys, "stdout", FluxFige())
    monkeypatch.setattr(sys, "stderr", FluxFige())

    force_utf8_output()  # ne doit pas lever


def test_flux_sans_reconfigure_ignore(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    force_utf8_output()  # ne doit pas lever


def test_typographie_francaise_encodable_apres_bascule():
    """Le cas reel : espace fine insecable devant les deux-points."""
    texte = "Article 326\u202f: aucune personne acquitt\u00e9e \u2014 m\u00eame sous une autre qualification."
    # Sans la bascule, cp1252 refuse ces caracteres.
    assert texte.encode("utf-8").decode("utf-8") == texte
    tampon = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="replace")
    tampon.write(texte)  # ne doit pas lever


def test_setup_logging_bascule_les_sorties(monkeypatch):
    import sys

    flux = FluxReconfigurable()
    monkeypatch.setattr(sys, "stdout", flux)
    monkeypatch.setattr(sys, "stderr", FluxReconfigurable())

    setup_logging()

    assert flux.encoding == "utf-8"
