"""Hybrid Intent Parser for Sara.

Converts natural-language commands into structured JSON actions using a
two-tier strategy:

1. **LLM reasoning** — sends the command through the LLM router
   (Groq → Ollama → heuristic chain) and maps the output to the
   canonical intent schema.
2. **Rule-based fallback** — regex patterns that catch common commands
   when the LLM is unavailable or returns an unusable result.

Supported actions
-----------------
* ``create_sheet``  — create a new worksheet
* ``write_cell``    — write a value to a specific cell
* ``read_cell``     — read the value of a specific cell
* ``create_chart``  — create a chart from a data range
* ``sum_column``    — sum all values in a column
* ``filter_table``  — filter rows by a column condition

Output schema
-------------
Every result is a ``dict`` with at least an ``"action"`` key.  Action-specific
fields vary per type — see :pydata:`SUPPORTED_ACTIONS` for the full list.

Configuration
-------------
The parser inherits all LLM configuration from the LLM router (env vars
``GROQ_API_KEY``, ``OLLAMA_MODEL``, etc.).
"""
from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any

# ── Resolve sibling backend modules ──────────────────────────────
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

log = logging.getLogger("intent.parser")

# ───────────────────── Public constants ──────────────────────────

SUPPORTED_ACTIONS: list[str] = [
    "create_sheet",
    "write_cell",
    "read_cell",
    "create_chart",
    "sum_column",
    "filter_table",
]

# Mapping from existing LLM/executor action types to the user-facing
# intent actions defined in this module.
_LLM_TYPE_TO_INTENT: dict[str, str] = {
    "create_sheet": "create_sheet",
    "create_excel": "create_sheet",
    "write_cell": "write_cell",
    "create_chart": "create_chart",
    "calculate_average": "sum_column",
    "calculate_row_sum": "sum_column",
}


# ───────────────────── Intent Parser ─────────────────────────────

class IntentParser:
    """Hybrid intent parser — LLM first, regex fallback.

    Parameters
    ----------
    use_llm:
        If True (default), attempt LLM parsing before falling back to
        regex rules.  Set to False for pure rule-based parsing.
    """

    def __init__(self, *, use_llm: bool = True) -> None:
        self.use_llm = use_llm

    # ── Public API ───────────────────────────────────────────────

    def parse(self, text: str) -> dict[str, Any]:
        """Parse *text* and return a structured intent dict.

        Returns a dict with at least ``{"action": "<action_name>"}`` and
        action-specific fields.  Falls back to ``{"action": "unknown", "raw": text}``
        when intent cannot be determined.
        """
        if not text or not isinstance(text, str):
            return {"action": "unknown", "raw": ""}

        text = text.strip()

        # ── Tier 1: LLM reasoning ──
        if self.use_llm:
            result = self._try_llm(text)
            if result is not None:
                return result

        # ── Tier 2: Rule-based fallback ──
        result = self._rule_based(text)
        if result is not None:
            return result

        return {"action": "unknown", "raw": text}

    # ── Tier 1: LLM ─────────────────────────────────────────────

    def _try_llm(self, text: str) -> dict[str, Any] | None:
        """Attempt to parse *text* via the LLM router and normalise the
        output to the intent schema."""
        try:
            from llm_router import route_llm  # lazy import to avoid circular deps
            raw = route_llm(text)
        except Exception as exc:
            log.warning("[IntentParser] LLM routing failed: %s", exc)
            return None

        if not isinstance(raw, dict):
            return None

        return self._normalise_llm_result(raw, text)

    def _normalise_llm_result(
        self, raw: dict[str, Any], text: str,
    ) -> dict[str, Any] | None:
        """Map an LLM router result (``{"type": ...}``) to the intent
        schema (``{"action": ...}``)."""

        # Handle multi-step plans — take the first meaningful step.
        steps = raw.get("steps")
        if isinstance(steps, list) and steps:
            for step in steps:
                mapped = self._map_single(step, text)
                if mapped is not None:
                    return mapped
            return None

        return self._map_single(raw, text)

    def _map_single(
        self, action: dict[str, Any], text: str,
    ) -> dict[str, Any] | None:
        """Convert a single LLM action dict to the intent schema."""
        atype = action.get("type", "")

        # Direct match in our supported set.
        intent_name = _LLM_TYPE_TO_INTENT.get(atype)
        if intent_name:
            return self._build_intent(intent_name, action, text)

        # Infer from LLM action fields even if type name differs.
        if atype == "write_formula":
            return self._build_intent("write_cell", action, text)

        if atype == "unknown":
            return None

        return None

    def _build_intent(
        self, intent_name: str, action: dict[str, Any], text: str,
    ) -> dict[str, Any]:
        """Build a canonical intent dict from *intent_name* + raw action."""
        if intent_name == "create_sheet":
            sheet_name = (
                action.get("sheet_name")
                or _extract_sheet_name(text)
                or "Sheet1"
            )
            result: dict[str, Any] = {
                "action": "create_sheet",
                "sheet_name": sheet_name,
            }
            headers = action.get("headers")
            if isinstance(headers, list) and headers:
                result["headers"] = headers
            return result

        if intent_name == "write_cell":
            cell = action.get("cell", "")
            value = action.get("value", action.get("formula", ""))
            return {"action": "write_cell", "cell": cell, "value": value}

        if intent_name == "create_chart":
            result = {"action": "create_chart"}
            if action.get("chart_type"):
                result["chart_type"] = action["chart_type"]
            if action.get("data_range"):
                result["data_range"] = action["data_range"]
            if action.get("title"):
                result["title"] = action["title"]
            return result

        if intent_name == "sum_column":
            result = {"action": "sum_column"}
            col = action.get("column")
            if col:
                result["column"] = col
            idx = action.get("index")
            if idx is not None:
                result["index"] = idx
            return result

        return {"action": intent_name}

    # ── Tier 2: Rule-based ───────────────────────────────────────

    def _rule_based(self, text: str) -> dict[str, Any] | None:
        """Apply regex rules to extract intent from *text*."""
        lower = text.lower().strip()

        # ── create_sheet ─────────────────────────────────────────
        m = re.search(
            r"\b(?:create|make|add|new)\s+(?:a\s+)?(?:new\s+)?"
            r"(?:sheet|worksheet|tab)\s+"
            r"(?:called|named|titled)?\s*['\"]?([A-Za-z0-9_\s]+?)['\"]?\s*$",
            text, re.IGNORECASE,
        )
        if m:
            return {"action": "create_sheet", "sheet_name": m.group(1).strip()}

        # Shorter form: "create sheet Sales"
        m = re.search(
            r"\b(?:create|make|add|new)\s+(?:a\s+)?(?:new\s+)?"
            r"(?:sheet|worksheet|tab)\s+([A-Za-z0-9_]+)",
            text, re.IGNORECASE,
        )
        if m:
            name = m.group(1).strip()
            if name.lower() not in ("called", "named", "titled", "for", "with"):
                return {"action": "create_sheet", "sheet_name": name}

        # ── write_cell ───────────────────────────────────────────
        m = re.search(
            r"\b(?:write|put|set|enter|type)\s+(.+?)\s+"
            r"(?:to|in|into|at)\s+(?:cell\s+)?([A-Za-z]+\d+)",
            text, re.IGNORECASE,
        )
        if m:
            raw_val = m.group(1).strip().strip("'\"")
            cell = m.group(2).upper()
            return {"action": "write_cell", "cell": cell, "value": _coerce_value(raw_val)}

        # Reverse: "write cell A1 value 100" / "set A1 to hello"
        m = re.search(
            r"\b(?:write|put|set|enter)\s+(?:cell\s+)?([A-Za-z]+\d+)\s+"
            r"(?:to|=|value)?\s*(.+)",
            text, re.IGNORECASE,
        )
        if m:
            cell = m.group(1).upper()
            raw_val = m.group(2).strip().strip("'\"")
            return {"action": "write_cell", "cell": cell, "value": _coerce_value(raw_val)}

        # ── read_cell ────────────────────────────────────────────
        m = re.search(
            r"\b(?:read|get|show|what(?:'s|\s+is)?(?:\s+the)?(?:\s+value)?(?:\s+of)?|"
            r"fetch|retrieve)\s+(?:cell\s+)?([A-Za-z]+\d+)",
            text, re.IGNORECASE,
        )
        if m:
            return {"action": "read_cell", "cell": m.group(1).upper()}

        # "what is in A1" / "value of B2"
        m = re.search(
            r"\b(?:what(?:'s|\s+is)?\s+in|value\s+(?:of|in|at))\s+(?:cell\s+)?([A-Za-z]+\d+)",
            text, re.IGNORECASE,
        )
        if m:
            return {"action": "read_cell", "cell": m.group(1).upper()}

        # ── create_chart ─────────────────────────────────────────
        m = re.search(
            r"\b(?:create|make|build|generate|draw|add)\s+(?:a\s+)?"
            r"(?:(bar|line|pie|scatter)\s+)?chart",
            lower,
        )
        if m:
            result: dict[str, Any] = {"action": "create_chart"}
            if m.group(1):
                result["chart_type"] = m.group(1)
            range_m = re.search(r"([A-Z]\d+:[A-Z]+\d+)", text, re.IGNORECASE)
            if range_m:
                result["data_range"] = range_m.group(1).upper()
            title_m = re.search(
                r"(?:titled?|called|named)\s+['\"]?([^'\"]+?)['\"]?\s*$",
                text, re.IGNORECASE,
            )
            if title_m:
                result["title"] = title_m.group(1).strip()
            return result

        # ── sum_column ───────────────────────────────────────────
        m = re.search(
            r"\b(?:sum|total|add\s+up)\s+(?:of\s+|all\s+|values?\s+(?:in|of)\s+)?"
            r"(?:column\s+)?([A-Za-z])\b",
            text, re.IGNORECASE,
        )
        if m:
            return {"action": "sum_column", "column": m.group(1).upper()}

        # "calculate sum of column B" / "sum column B"
        m = re.search(
            r"\b(?:calculate|compute|find|get)\s+(?:the\s+)?(?:sum|total)\s+"
            r"(?:of\s+)?(?:column\s+)?([A-Za-z])\b",
            text, re.IGNORECASE,
        )
        if m:
            return {"action": "sum_column", "column": m.group(1).upper()}

        # ── filter_table ─────────────────────────────────────────
        m = re.search(
            r"\b(?:filter|show\s+(?:only\s+)?(?:rows?\s+)?(?:where)?|"
            r"find\s+(?:rows?\s+)?(?:where)?)\s+"
            r"(?:by\s+)?(?:column\s+)?([A-Za-z]+)\s*"
            r"(>=|<=|!=|<>|==|=|>|<|(?:is|equals?|contains?|greater\s+than|less\s+than))\s*"
            r"['\"]?(.+?)['\"]?\s*$",
            text, re.IGNORECASE,
        )
        if m:
            column = m.group(1).strip()
            op_raw = m.group(2).strip().lower()
            value = m.group(3).strip()
            operator = _normalise_operator(op_raw)
            return {
                "action": "filter_table",
                "column": column,
                "operator": operator,
                "value": _coerce_value(value),
            }

        # Simpler: "filter by column Status = Active"
        m = re.search(
            r"\bfilter\s+(?:by\s+)?(?:column\s+)?([A-Za-z]+)\s*"
            r"(>=|<=|!=|<>|==|=|>|<)\s*['\"]?(.+?)['\"]?\s*$",
            text, re.IGNORECASE,
        )
        if m:
            return {
                "action": "filter_table",
                "column": m.group(1).strip(),
                "operator": _normalise_operator(m.group(2).strip()),
                "value": _coerce_value(m.group(3).strip()),
            }

        return None


# ───────────────────── Helpers ───────────────────────────────────

def _extract_sheet_name(text: str) -> str | None:
    """Try to find a sheet name from text like "called X" / "named X"."""
    m = re.search(
        r"(?:called|named|titled)\s+['\"]?([A-Za-z0-9_\s]+?)['\"]?\s*$",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    # Try to get the last capitalized word after "sheet"
    m = re.search(r"\bsheet\s+([A-Z][A-Za-z0-9_]*)", text)
    if m:
        return m.group(1)
    return None


def _coerce_value(raw: str) -> int | float | str:
    """Convert a string to int/float if it looks numeric, else keep as str."""
    raw = raw.strip()
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _normalise_operator(op: str) -> str:
    """Normalise human-friendly operators to symbols."""
    op = op.lower().strip()
    mapping = {
        "=": "=",
        "==": "=",
        "is": "=",
        "equals": "=",
        "equal": "=",
        "!=": "!=",
        "<>": "!=",
        ">": ">",
        "greater than": ">",
        "<": "<",
        "less than": "<",
        ">=": ">=",
        "<=": "<=",
        "contains": "contains",
        "contain": "contains",
    }
    return mapping.get(op, op)
