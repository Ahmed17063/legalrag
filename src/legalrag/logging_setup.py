"""Configuration des logs pour un job batch."""

from __future__ import annotations

import logging
import sys


def force_utf8_output() -> None:
    """Bascule stdout et stderr en UTF-8.

    La console Windows est en cp1252 par defaut. Elle ne sait encoder ni
    les espaces fines insecables de la typographie francaise (U+202F,
    devant « : » et « ? »), ni certains guillemets et tirets : un simple
    `print` de texte juridique francais leve alors UnicodeEncodeError et
    interrompt la commande.

    `errors="replace"` garantit qu'un caractere exotique degrade
    l'affichage au lieu de faire echouer le programme.
    """
    for flux in (sys.stdout, sys.stderr):
        reconfigure = getattr(flux, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # Flux redirige vers quelque chose qui ne se reconfigure pas :
            # sans importance, on continue.
            pass


def setup_logging(verbose: bool = False) -> None:
    """Logs horodates sur stderr, avec le nom du module a l'origine du message."""
    force_utf8_output()

    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s %(name)-24s %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Ces bibliotheques sont tres bavardes en DEBUG et noieraient nos messages.
    for noisy in ("googleapiclient", "google_auth_httplib2", "urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
