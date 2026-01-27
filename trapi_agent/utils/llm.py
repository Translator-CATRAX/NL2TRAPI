#!/usr/bin/env python3
"""
llm.py

Utilities for interacting with a HuggingFace text-generation pipeline.

Responsibilities:
  - Lazy-load a singleton HF pipeline (with optional accelerate support)
  - Expose a simple `run_llm` wrapper for text generation
  - Support optional stop-sequence truncation in post-processing
"""

from functools import lru_cache
import logging
import os
from typing import Dict, List, Optional

import torch
from transformers import AutoConfig, Pipeline, pipeline

from ..config import settings

logger = logging.getLogger(__name__)


def _env_true(name: str) -> bool:
    value = (os.getenv(name) or "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _accelerate_available() -> bool:
    """
    Detect if the `accelerate` library is installed for optimized multi-GPU/mixed-precision.

    Returns:
        True if `accelerate` can be imported, False otherwise.
    """
    try:
        import accelerate  # noqa: F401
        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def get_llm() -> Pipeline:
    """
    Returns a singleton HuggingFace text-generation pipeline.

    Automatically applies `device_map` and `torch_dtype` if `accelerate` is available.
    """
    model_config = AutoConfig.from_pretrained(
        settings.LLM_NAME,
        trust_remote_code=True,
    )
    is_enc_dec = bool(getattr(model_config, "is_encoder_decoder", False))

    task = "text2text-generation" if is_enc_dec else "text-generation"

    kwargs: Dict = {
        "task": task,
        "model": settings.LLM_NAME,
        "trust_remote_code": True,
    }
    if task == "text-generation":
        # ensure return format for post-processing
        kwargs["return_full_text"] = False

    # if _accelerate_available():
    if _accelerate_available() or torch.cuda.is_available():
        kwargs.update(device_map="auto", torch_dtype="auto")

    return pipeline(**kwargs)


def run_llm(
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    stop: Optional[List[str]] = None,
    min_new_tokens: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    *,
    allow_empty: bool = False,
) -> str:
    """
    Generate text from the shared text-generation pipeline.

    Args:
        prompt: The input prompt string.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (<=0 for greedy).
        top_p: Nucleus sampling cutoff (only if sampling).
        top_k: Top-k sampling cutoff (only if sampling).
        stop: Optional list of substrings; if any appear in the output,
              the result is truncated before the first occurrence.
        allow_empty: When NO_LLM=1, return empty output instead of raising.

    Returns:
        The generated text, truncated at the first stop sequence if provided.
    """
    if _env_true("NO_LLM"):
        if not (allow_empty or _env_true("ALLOW_EMPTY_LLM")):
            raise RuntimeError(
                "NO_LLM=1 is set, but run_llm() was called. "
                "Disable NO_LLM, or call run_llm(..., allow_empty=True), "
                "or set ALLOW_EMPTY_LLM=1 to permit empty outputs."
            )
        logger.warning("NO_LLM=1 -> skipping LLM call; returning empty output")
        return ""

    llm = get_llm()
    greedy = temperature <= 0.0

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": not greedy,
    }
    if not greedy:
        gen_kwargs.update(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    if min_new_tokens is not None:
        gen_kwargs["min_new_tokens"] = int(min_new_tokens)
    if repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = float(repetition_penalty)

    # perform generation
    outputs = llm(prompt, **gen_kwargs)
    raw = outputs[0].get("generated_text", "")

    # apply stop-sequence truncation
    if stop:
        for seq in stop:
            idx = raw.find(seq)
            if idx != -1:
                raw = raw[:idx]
                break

    return raw
