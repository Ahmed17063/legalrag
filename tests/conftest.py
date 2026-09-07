"""Fixtures partagees par les tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from legalrag.config import Settings

DIMENSION_TEST = 8


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Configuration complete, pointant sur des chemins temporaires."""
    credentials = tmp_path / "sa.json"
    credentials.write_text("{}", encoding="utf-8")
    return Settings(
        voyage_api_key="cle-de-test",
        voyage_base_url="https://ai.mongodb.com/v1",
        google_credentials_file=credentials,
        drive_folder_id="dossier",
        drive_recursive=True,
        corpus_source="drive",
        local_corpus_dir=tmp_path / "corpus",
        embedding_model="voyage-context-4",
        embedding_mode="contextualized",
        embedding_dimension=DIMENSION_TEST,
        model_max_tokens=32_000,
        chunk_tokens=60,
        chunk_overlap_tokens=20,
        strip_boilerplate=True,
        max_requests_per_minute=1000,
        max_tokens_per_minute=10_000_000,
        index_dir=tmp_path / "index",
        cache_dir=tmp_path / "cache",
        qdrant_collection="test",
        qdrant_url=None,
        qdrant_api_key=None,
        qdrant_path=tmp_path / "qdrant",
        qdrant_batch=64,
        rerank_model="rerank-3",
        rerank_enabled=True,
        retrieval_top_k=25,
        retrieval_top_n=5,
        groq_api_key="cle-groq-de-test",
        generation_model="openai/gpt-oss-120b",
        generation_max_tokens=2048,
    )
