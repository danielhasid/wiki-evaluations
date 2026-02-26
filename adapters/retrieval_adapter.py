"""
Adapter that replicates the retrieval / vector-DB interface used throughout
the original AppServer codebase.

Interfaces replicated:
    from AppServer.src.AppServer.Databases.wisery_milvus import MilvusConnector
    from AppServer.src.AppServer.RAG.retrieval import (
        RetrievalType,
        RETRIEVAL_RANKING_FUNCTIONS,
        retrieve_documents_and_scores_with_llm_keywords,
        retrieve_documents_and_scores_with_bm25_keywords,
        retrieve_documents_and_scores_with_vectordb_semantic,
        filter_docs_by_scores,
        retrieve_relevant_docs,
    )
    from AppServer.src.AppServer.Utils.evaluation.common import (
        call_retrieval_function,
        store_to_mongo_aggregated_results,
        evaluate_single_db,
        EvaluationType,
    )
    from AppServer.src.AppServer.Common.SystemConfig.system_config import SystemConfig

Wire this to real Milvus / retrieval backends by implementing the stubs below.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

import config


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RetrievalType(Enum):
    LLM_KEYWORDS = "llm_keywords"
    VECTOR_DB_SEMANTIC = "vectordb_semantic"
    BM25_KEYWORDS = "bm25_keywords"
    LOCATION_KEYWORDS = "location_keywords"
    GEO_SEMANTIC = "geo_semantic"
    RETRIEVE_RELEVANT_DOCS = "retrieve_relevant_docs"
    TEMPORAL_RETRIEVAL = "temporal_retrieval"


class RETRIEVAL_RANKING_FUNCTIONS(Enum):
    RRF = "reciprocal_rank_fusion"
    DEFAULT = "default"


class EvaluationType(Enum):
    CORRECTNESS = "correctness"


# ---------------------------------------------------------------------------
# MilvusConnector stub
# ---------------------------------------------------------------------------

class MilvusConnector:
    """
    Stub for AppServer's MilvusConnector.

    TODO: Replace with a real pymilvus / langchain-milvus implementation
    pointing at config.MILVUS_HOST / config.MILVUS_PORT.
    """

    def __init__(self, collection_name: str):
        self.collection_name = collection_name

    def get_milvus_client(self) -> "MilvusConnector":
        # TODO: return a real Milvus / LangChain VectorStore client
        return self


# ---------------------------------------------------------------------------
# Retrieval function stubs
# ---------------------------------------------------------------------------

DocAndScore = Tuple[Document, float]


def retrieve_documents_and_scores_with_llm_keywords(
    query: str,
    agent_name: str,
    k: int = 20,
    vector_db: Any = None,
    **kwargs,
) -> List[DocAndScore]:
    """TODO: implement LLM-keyword-based retrieval."""
    raise NotImplementedError("LLM-keyword retrieval not yet wired up")


def retrieve_documents_and_scores_with_bm25_keywords(
    query: str,
    agent_name: str,
    k: int = 20,
    vector_db: Any = None,
    **kwargs,
) -> List[DocAndScore]:
    """TODO: implement BM25 keyword retrieval."""
    raise NotImplementedError("BM25 retrieval not yet wired up")


def retrieve_documents_and_scores_with_vectordb_semantic(
    query: str,
    vector_db: Any,
    agent_name: str,
    k: int = 20,
    **kwargs,
) -> List[DocAndScore]:
    """TODO: implement vector-DB semantic retrieval."""
    raise NotImplementedError("Semantic retrieval not yet wired up")


def filter_docs_by_scores(
    docs_and_scores: List[DocAndScore],
    threshold: float = 0.5,
) -> List[DocAndScore]:
    """Filter retrieved documents by score threshold."""
    return [(doc, score) for doc, score in docs_and_scores if score <= threshold]


def retrieve_relevant_docs(
    query: str,
    vector_db: Any,
    agent_name: str,
    k: int = 5,
    **kwargs,
) -> List[DocAndScore]:
    """TODO: implement hybrid relevance retrieval."""
    raise NotImplementedError("Hybrid retrieval not yet wired up")


# ---------------------------------------------------------------------------
# Evaluation common stubs
# ---------------------------------------------------------------------------

def call_retrieval_function(
    retrievers_names: List[str],
    agent_name: str,
    user_query: str,
    k: int,
) -> List[DocAndScore]:
    """TODO: dispatch to the appropriate retriever(s) and return results."""
    raise NotImplementedError("call_retrieval_function not yet wired up")


def store_to_mongo_aggregated_results(
    db_name: Optional[str],
    all_retrievers: List[str],
) -> Dict[str, Any]:
    """TODO: aggregate retrieval evaluation results from MongoDB."""
    raise NotImplementedError("store_to_mongo_aggregated_results not yet wired up")


def evaluate_single_db(
    metric_dict: Dict[str, Any],
    db_name: str,
    N: int,
    db_filter: Dict[str, Any],
    all_retrievers: List[str],
    evaluation_id: str,
    k: int,
) -> None:
    """TODO: run evaluation loop over a single database."""
    raise NotImplementedError("evaluate_single_db not yet wired up")


# ---------------------------------------------------------------------------
# SystemConfig stub
# ---------------------------------------------------------------------------

class SystemConfig:
    """
    Minimal stub for AppServer's SystemConfig.
    Override get_config / set_config to read from environment or a config file.
    """

    _store: Dict[str, Any] = {
        "correctness_tests_db_limit": "10",
        "ranking_function_name": RETRIEVAL_RANKING_FUNCTIONS.RRF.value,
    }

    def get_config(self, config: str) -> Any:
        return self._store.get(config)

    def set_config(self, config: str, new_value: Any) -> None:
        self._store[config] = new_value
