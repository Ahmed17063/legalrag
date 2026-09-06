"""Interface web du chatbot juridique.

Le module s'appelle `server` et non `app` : `legalrag.web.app` aurait
designe tantot le module, tantot l'objet FastAPI qu'il exporte, ce qui
casse aussi bien les imports que les chemins passes a uvicorn.
"""

from .server import app

__all__ = ["app"]
