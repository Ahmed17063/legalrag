"""Collecte de textes de loi sur les sources publiques qui l'autorisent."""

from .eurlex import PECHE_MARITIME, EurlexCollector, EurlexError, TexteEuropeen
from .justice import JusticeScraper, ScrapingError

__all__ = [
    "EurlexCollector",
    "EurlexError",
    "JusticeScraper",
    "PECHE_MARITIME",
    "ScrapingError",
    "TexteEuropeen",
]
