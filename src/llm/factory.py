"""
LLM factory: default local Ollama, optional OpenAI.
Used by Layer C optional LangChain analysis.
"""
from __future__ import annotations

from typing import Any, Optional

from src.config import get_config


def get_llm_for_layer_c():
    """
    Return a LangChain chat model for Layer C analysis.
    Default: Ollama (local). Set LLM_BACKEND=openai to use OpenAI.
    """
    cfg = get_config()
    backend = (cfg.LLM_BACKEND or "ollama").strip().lower()
    if backend == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=cfg.LLM_TEMPERATURE,
            openai_api_key=cfg.OPENAI_API_KEY or None,
            request_timeout=cfg.LLM_TIMEOUT,
        )
    # default: ollama
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise ImportError(
            "LLM_BACKEND=ollama requires langchain-ollama. Install: pip install langchain-ollama"
        )
    base_url = (cfg.LLM_OLLAMA_PRIMARY or "http://127.0.0.1:11434").strip()
    model = (cfg.LLM_MODEL or "llama3.2:latest").strip()
    return ChatOllama(
        base_url=base_url,
        model=model,
        temperature=cfg.LLM_TEMPERATURE,
        timeout=cfg.LLM_TIMEOUT,
    )
