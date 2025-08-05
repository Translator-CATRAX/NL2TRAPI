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
from typing import List, Optional

from transformers import pipeline, Pipeline
import torch
from ..config import settings


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
    kwargs: Dict = {
        "task": "text-generation",
        "model": settings.LLM_NAME,
        # ensure return format for post-processing
        "return_full_text": False,
    }

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

    Returns:
        The generated text, truncated at the first stop sequence if provided.
    """
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