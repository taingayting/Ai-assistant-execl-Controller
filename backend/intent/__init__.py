"""Intent parsing package for Sara.

Provides :class:`IntentParser` — a hybrid intent parser that uses LLM
reasoning as the primary strategy with rule-based regex as fallback,
converting natural-language commands into structured JSON actions.

Quick start::

    from intent.intent_parser import IntentParser

    parser = IntentParser()
    result = parser.parse("Create a new sheet called Sales")
    # -> {"action": "create_sheet", "sheet_name": "Sales"}

    result = parser.parse("Write hello to cell A1")
    # -> {"action": "write_cell", "cell": "A1", "value": "hello"}
"""

from intent.intent_parser import IntentParser

__all__ = ["IntentParser"]
