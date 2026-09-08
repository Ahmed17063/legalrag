"""Empecher la mise en veille pendant un traitement long.

Un job d'indexation dure de quelques minutes a plusieurs heures. Si la
machine s'endort entre-temps, le traitement est suspendu : il paraitra
bloque, puis reprendra au reveil, parfois des heures plus tard. C'est
arrive plusieurs fois sur ce projet, et le diagnostic est trompeur —
rien dans les journaux ne distingue une veille d'un blocage.

Windows expose `SetThreadExecutionState` pour declarer qu'un traitement
est en cours. La demande est **volatile** : elle disparait a la fin du
processus, meme s'il est tue. Rien n'est donc modifie durablement dans
les reglages de l'utilisateur, contrairement a `powercfg`.

Sur les autres systemes, le gestionnaire ne fait rien et le dit une fois.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from collections.abc import Iterator

logger = logging.getLogger(__name__)

# Constantes de l'API Windows.
_ES_CONTINUOUS = 0x80000000  # l'etat vaut jusqu'a nouvel ordre
_ES_SYSTEM_REQUIRED = 0x00000001  # la machine ne doit pas s'endormir
# ES_DISPLAY_REQUIRED est volontairement omis : garder l'ecran allume
# n'apporte rien a un job de fond et consomme de la batterie.


def _appliquer(drapeaux: int) -> bool:
    """Transmet l'etat souhaite a Windows. Renvoie False si impossible."""
    if not sys.platform.startswith("win"):
        return False
    try:
        import ctypes

        resultat = ctypes.windll.kernel32.SetThreadExecutionState(drapeaux)
    except (AttributeError, OSError) as exc:  # pragma: no cover - dependant de l'OS
        logger.debug("SetThreadExecutionState indisponible : %s", exc)
        return False

    # L'API renvoie 0 en cas d'echec, l'etat precedent sinon.
    return resultat != 0


@contextmanager
def garder_eveille(motif: str = "traitement en cours") -> Iterator[bool]:
    """Empeche la veille tant que le bloc s'execute.

    Yields:
        True si la demande a ete acceptee, False sinon — auquel cas le
        traitement se deroule normalement, mais la machine peut s'endormir.

    L'etat est retabli a la sortie, y compris si une exception survient.
    """
    actif = _appliquer(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)

    if actif:
        logger.info("Veille suspendue pendant %s.", motif)
    elif sys.platform.startswith("win"):
        logger.warning(
            "Impossible de suspendre la veille : si la machine s'endort, %s "
            "sera interrompu puis reprendra au reveil.",
            motif,
        )
    else:
        logger.debug("Suspension de la veille non geree sur %s", sys.platform)

    try:
        yield actif
    finally:
        if actif:
            # Rendre la main au systeme : sans ce rappel, l'etat resterait
            # actif jusqu'a la fin du processus.
            _appliquer(_ES_CONTINUOUS)
            logger.debug("Veille retablie.")
