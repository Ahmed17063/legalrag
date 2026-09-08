"""Collecte de textes de loi sur les sources publiques qui l'autorisent."""

from .eurlex import PECHE_MARITIME, EurlexCollector, EurlexError, TexteEuropeen
from .international import (
    PECHE_INTERNATIONAL,
    CollecteurInternational,
    TexteInternational,
)
from .justice import JusticeScraper, ScrapingError

__all__ = [
    "CollecteurInternational",
    "EurlexCollector",
    "EurlexError",
    "JusticeScraper",
    "PECHE_INTERNATIONAL",
    "PECHE_MARITIME",
    "ScrapingError",
    "TexteEuropeen",
    "TexteInternational",
]
