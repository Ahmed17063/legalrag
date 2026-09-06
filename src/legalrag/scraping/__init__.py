"""Collecte de textes de loi sur les sites publics mauritaniens."""

from .justice import JusticeScraper, ScrapingError

__all__ = ["JusticeScraper", "ScrapingError"]
