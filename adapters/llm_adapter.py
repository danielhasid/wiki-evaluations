"""
Adapter that replicates the LLM-invocation interface used throughout the
original AppServer codebase:

    from AppServer.src.AppServer.LLM.llm_utils import execute_prompt
    from AppServer.src.AppServer.prompts.prompt_types import PromptType
    from AppServer.src.AppServer.LLM.invokers.llm_invoker import LLMInvoker
    from AppServer.src.AppServer.LLM.config.model_config import ModelConfig

Wire this to a real LLM by setting OPENAI_API_KEY (and optionally LLM_MODEL)
in your environment (see config.py).
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

import tiktoken
from openai import OpenAI

import config

_client: Optional[OpenAI] = None


def _get_openai() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


# ---------------------------------------------------------------------------
# PromptType – mirrors the enum in AppServer; add values as needed
# ---------------------------------------------------------------------------

class PromptType(str, Enum):
    FRAMES_REPHRASE = "frames_rephrase"
    RAGAS_WO_NV_ACCURACY = "ragas_wo_nv_accuracy"
    EVAL_TEXT2SQL_ALIGNMENT = "eval_text2sql_alignment"
    LLM_KEYWORDS = "llm_keywords"


# ---------------------------------------------------------------------------
# Prompt templates (minimal stubs – fill in the real prompts as needed)
# ---------------------------------------------------------------------------

_PROMPTS: dict[str, str] = {
    PromptType.FRAMES_REPHRASE: (
        "Rephrase the following question in a different but semantically equivalent way.\n"
        "Question: {query}\nRephrased:"
    ),
    PromptType.RAGAS_WO_NV_ACCURACY: (
        "Given the question, expected answer, and actual answer, rate how accurate the actual answer is.\n"
        "Question: {query}\nExpected: {sentence_true}\nActual: {sentence_inference}\n"
        "Output format: {output_format}\nRating:"
    ),
    PromptType.EVAL_TEXT2SQL_ALIGNMENT: (
        "Compare the two SQL queries for semantic alignment.\n"
        "Question: {user_query}\nGround truth SQL: {ground_truth_sql}\n"
        "Created SQL: {created_sql}\nOutput format: {output_format}\nScore:"
    ),
    PromptType.LLM_KEYWORDS: (
        "Extract search keywords from the following query.\nQuery: {query}\nKeywords:"
    ),
}


def execute_prompt(prompt_type: PromptType, **kwargs) -> str:
    """
    Render the prompt template for *prompt_type*, call the configured LLM,
    and return the raw text response.
    """
    template = _PROMPTS.get(prompt_type, "")
    if not template:
        raise ValueError(f"No prompt template registered for {prompt_type!r}")

    prompt_text = template.format(**kwargs)

    response = _get_openai().chat.completions.create(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        messages=[{"role": "user", "content": prompt_text}],
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# LLMInvoker – mirrors the batch-processing helper in AppServer
# ---------------------------------------------------------------------------

class ModelConfig:
    """Mirrors AppServer's ModelConfig constants."""
    PROVIDER_MODEL_MISTRAL_LARGE_2407 = "mistral-large-2407"
    PROVIDER_MODEL_MISTRAL_SMALL_2409 = "mistral-small-2409"
    PROVIDER_MODEL_GPT4O = "gpt-4o"


class LLMInvoker:
    """Minimal replica of AppServer's LLMInvoker batch helpers."""

    @staticmethod
    def get_number_of_tokens(model_id: str, prompts: str) -> int:
        """Estimate token count using tiktoken (falls back to GPT-4 encoding)."""
        try:
            enc = tiktoken.encoding_for_model(model_id)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(prompts))

    @staticmethod
    def get_prompts_batches(
        model_id: str,
        prompts: List[str],
        max_output_tokens: int = 8192,
        context_window: int = 127500,
    ) -> List[List[str]]:
        """
        Greedily group *prompts* into batches that fit within *context_window*
        tokens (reserving *max_output_tokens* for the response).
        """
        try:
            enc = tiktoken.encoding_for_model(model_id)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")

        token_budget = context_window - max_output_tokens
        batches: List[List[str]] = []
        current_batch: List[str] = []
        current_tokens = 0

        for prompt in prompts:
            n = len(enc.encode(prompt))
            if current_batch and current_tokens + n > token_budget:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append(prompt)
            current_tokens += n

        if current_batch:
            batches.append(current_batch)

        return batches
