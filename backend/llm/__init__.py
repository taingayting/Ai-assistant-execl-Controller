"""LLM routing package for Sara.

Provides a modular, production-ready router that dispatches requests
between Groq (cloud), Ollama (local), and a rule-based heuristic,
with support for both blocking and streaming responses.

Quick start::

    from llm.route_llm import LLMRouter

    router = LLMRouter()
    action = router.route("create excel with headers name and score")

    # Streaming (yields partial text chunks):
    for chunk in router.route_stream("add row Tom 90"):
        print(chunk, end="", flush=True)
"""

from llm.route_llm import LLMRouter, route, route_stream
from llm.ollama_client import OllamaClient, RECOMMENDED_MODELS

__all__ = [
    "LLMRouter",
    "route",
    "route_stream",
    "OllamaClient",
    "RECOMMENDED_MODELS",
]
