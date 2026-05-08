"""LLM-based parser that converts free text to structured actions.

This module prefers the Groq inference API (LLaMA 3) when configured via
environment variables. If Groq is not available or the call fails, it falls
back to returning an unknown-action dict.

The Groq call attempts to force the model to output pure JSON describing a
single Excel action. The code contains flexible response parsing to handle
different Groq response shapes; adapt `GROQ_API_URL`/parsing as needed for
your account/SDK.
"""
import os
import json
import re
import logging
import requests

log = logging.getLogger('llm_parser')

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
GROQ_API_URL = os.environ.get('GROQ_API_URL', 'https://api.groq.com/openai/v1/chat/completions')
# llama3-8b-8192 is the recommended fast model for Groq.
# If you see 403 errors, the model set in GROQ_MODEL in your .env is likely
# deprecated (e.g. llama3-70b-8192).  Switch to one of:
#   llama3-8b-8192  |  llama-3.3-70b-versatile  |  gemma2-9b-it
GROQ_MODEL = os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile')
# Per-request timeout for the Groq HTTP call.  Keep this short so that a
# slow or unavailable Groq API fails fast and lets the router fall through to
# Ollama quickly.  8 s is enough for normal responses; raise it if you are
# sending very long prompts.
GROQ_TIMEOUT = int(os.environ.get('GROQ_TIMEOUT', '8'))

# HTTP status codes that indicate the model is deprecated or unauthorised.
_GROQ_AUTH_CODES = {401, 403}


def _extract_json_from_text(s: str):
    """Try to pull a JSON object from free text. Return Python object or raise."""
    if not s or not isinstance(s, str):
        raise ValueError('no text')
    s = s.strip()
    # If the entire string is JSON
    try:
        return json.loads(s)
    except Exception:
        pass
    # otherwise look for the first { ... } block
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # Try for bracketed arrays (unlikely for our schema but safe)
    m = re.search(r"\[[\s\S]*\]", s)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    raise ValueError('no json found')


class RobustJsonExtractor:
    """
    Extracts and parses JSON from potentially messy LLM responses.
    """
    _fence_re = re.compile(r"```(?:json)?\n([\s\S]*?)```", re.IGNORECASE)
    _brace_re = re.compile(r"(\{[\s\S]*\})")
    _bracket_re = re.compile(r"(\[[\s\S]*\])")

    @staticmethod
    def _clean_common_noise(s: str) -> str:
        # Remove leading labels like "json" or "response:" and surrounding markdown
        s = s.strip()
        # Strip common prefixes
        s = re.sub(r"^response:\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^json:\s*", "", s, flags=re.IGNORECASE)
        return s.strip()

    @staticmethod
    def _try_fixes(s: str) -> str:
        # Remove trailing commas before closing braces/brackets
        s = re.sub(r",\s*([}\]])", r"\1", s)
        # Replace Python booleans/null with JSON literals
        s = re.sub(r"\bTrue\b", "true", s)
        s = re.sub(r"\bFalse\b", "false", s)
        s = re.sub(r"\bNone\b", "null", s)
        return s

    @classmethod
    def extract(cls, raw_text: str):
        if not raw_text or not isinstance(raw_text, str):
            raise ValueError('no text')

        # 1) Try fenced code blocks first
        m = cls._fence_re.search(raw_text)
        if m:
            candidate = m.group(1).strip()
            candidate = cls._clean_common_noise(candidate)
            try:
                return json.loads(candidate)
            except Exception:
                # attempt simple fixes
                try:
                    return json.loads(cls._try_fixes(candidate))
                except Exception:
                    pass

        # 2) Try to find the first JSON object or array in the text
        for regex in (cls._brace_re, cls._bracket_re):
            m = regex.search(raw_text)
            if m:
                candidate = m.group(1)
                candidate = cls._clean_common_noise(candidate)
                try:
                    return json.loads(candidate)
                except Exception:
                    try:
                        return json.loads(cls._try_fixes(candidate))
                    except Exception:
                        pass

        # 3) Try the whole text directly
        candidate = cls._clean_common_noise(raw_text)
        try:
            return json.loads(candidate)
        except Exception:
            try:
                return json.loads(cls._try_fixes(candidate))
            except Exception:
                pass

        # 4) As a last resort try to coerce single quotes to double quotes (best-effort)
        coerced = re.sub(r"'([^\"]*?)'", lambda m: '"' + m.group(1).replace('"', '\\"') + '"', raw_text)
        coerced = cls._try_fixes(coerced)
        try:
            return json.loads(coerced)
        except Exception:
            pass

        log.warning('RobustJsonExtractor: all strategies failed — raw LLM output: %r', raw_text[:500])
        raise ValueError('no json found')


def parse_with_llm(text: str) -> dict:
    """Parse user text through the hybrid LLM router (Groq → Ollama fallback).

    Delegates to ``llm_router.route_llm`` so that Groq is tried first and
    Ollama serves as an automatic fallback.  Import is lazy to avoid circular
    dependency (llm_router imports helpers from this module).
    """
    from llm_router import route_llm  # lazy to break circular import
    return route_llm(text)


def _parse_groq_direct(text: str) -> dict:
    """Call the Groq cloud API directly (LLaMA 3).

    Returns a structured action dict.  Falls back to heuristic parse on failure.
    This is the low-level Groq backend used by llm_router.try_groq().
    """
    if not GROQ_API_KEY:
        log.warning('_parse_groq_direct: no GROQ_API_KEY configured')
        return {"type": "unknown", "raw": text}

    system = """CRITICAL RULE: Return ONLY valid JSON. No explanations. No markdown. No code blocks. No text before or after the JSON object.
If you cannot parse the command, return exactly: {"type":"unknown"}

You are an expert office assistant. Convert spoken Excel commands into structured JSON actions.
Infer intent intelligently — do NOT be a strict keyword parser.

========================
OUTPUT CONTRACT (STRICT)
========================
Return ONLY valid JSON. No markdown. No explanations.
You MUST choose from the allowed actions only. Do NOT invent or use tool names that are not listed below.

WRONG (never do this):
  Here is the JSON: {"type":"add_row",...}
  {"row": [{"name": "john", "score": 90}]}
  ```json {"type":"add_row"} ```

CORRECT (do exactly this):
  {"type":"add_row","values":["John",90]}

Preferred shapes:
1) Single action:
    {"type":"add_row","values":["Tom",90]}

2) Multi-step plan when command has multiple operations:
    {"steps":[
      {"type":"create_excel","headers":["Name","Score"]},
      {"type":"add_row","values":["Tom",90]},
      {"type":"add_row","values":["Alex",85]}
    ]}

Use "steps" whenever there are 2+ operations in one sentence.

========================
ALLOWED ACTION TYPES
========================
- create_excel:       fields -> headers (required), filename (optional)
- add_row:            fields -> values (required)
- delete_row:         fields -> index (required integer, 1-based)
- delete_column:      fields -> column (required, letter e.g. "B")
- add_column:         fields -> name (required)
- write_cell:         fields -> cell (required e.g. "A1"), value (required)
- write_formula:      fields -> cell (required e.g. "B10"), formula (required e.g. "=SUM(B2:B9)")
- calculate_row_sum:  fields -> index (required integer, 1-based)
- calculate_average:  fields -> column (optional, letter e.g. "B"; omit to auto-detect numeric column)
- open_file:          no extra fields (opens the active workbook in Excel)
- create_chart:       fields -> chart_type (optional: bar/line/pie/scatter, default bar), data_range (required e.g. "A1:C10"), title (optional)
- format_cells:       fields -> range (required e.g. "A1:C10"), bold/italic/font_size/font_color/fill_color (all optional)
- read_cell:          fields -> cell (required e.g. "B3")
- create_sheet:       fields -> sheet_name (required)
- switch_sheet:       fields -> sheet_name (required)
- undo:               no extra fields (reverts the last change)
- save:               no extra fields

========================
DOMAIN HEADER INFERENCE TABLE
========================
When the user mentions one of the following topics WITHOUT giving explicit column names,
automatically use the corresponding headers:

  Topic keywords                       | Inferred headers
  -------------------------------------|-------------------------------------------
  score, grades, marks, result, exam   | ["Name", "Score"]
  student score, grade sheet           | ["Name", "Score"]
  attendance, present, absent, roll    | ["Name", "Date", "Status"]
  inventory, stock, items, products    | ["Item", "Quantity"]
  expense, expenses, cost, costs,      | ["Item", "Cost"]
    payment                            |
  budget, finance, spending            | ["Category", "Budget", "Actual"]
  sales, revenue, orders               | ["Product", "Quantity", "Revenue"]
  employee, staff, hr, payroll         | ["Name", "Department", "Salary"]
  project, task, todo, milestone       | ["Task", "Owner", "Status", "Due Date"]
  meeting, discussion, agenda          | ["Topic", "Date", "Participants"]

If the user provides explicit column names, always use those instead.

========================
INTENT INFERENCE RULES
========================
1) Identify the domain from keywords and apply the header table above when no explicit headers are given.
2) If user says create/make/build + add/insert in one request, always produce a "steps" plan.
3) If multiple data records are mentioned in one command, output one add_row per record.
4) Convert word numbers to integers when obvious ("ninety" -> 90, "ten" -> 10, "twenty-five" -> 25).
5) Keep person/item names as strings with natural capitalization.
6) For attendance: map "present"/"yes"/"here" -> "Present", "absent"/"no" -> "Absent" in the Status field.
7) For inventory/expense amounts: treat the numeric value as Quantity or Cost respectively.
8) Prefer a concrete guess over returning unknown - if intent is at all inferable, produce a valid action.
9) Use {"type":"unknown","raw":"..."} ONLY when there is truly zero actionable intent.
10) For project/task: extract task name and status keyword (pending/done/in-progress/completed); Owner and Due Date default to "".
11) For meeting: extract topic text after the add/insert trigger word; Date and Participants default to "".
12) For budget: extract category name and numeric Budget amount; Actual defaults to "".

========================
FEW-SHOT EXAMPLES
========================

--- STUDENT SCORES ---
Input: "Create an excel sheet for student score and add Tom with 90 and Alex with 85"
Output: {"steps":[{"type":"create_excel","headers":["Name","Score"]},{"type":"add_row","values":["Tom",90]},{"type":"add_row","values":["Alex",85]}]}

Input: "make a score table and insert Tom 90 and Alex 85"
Output: {"steps":[{"type":"create_excel","headers":["Name","Score"]},{"type":"add_row","values":["Tom",90]},{"type":"add_row","values":["Alex",85]}]}

Input: "Create a student score sheet and add Tom 90"
Output: {"steps":[{"type":"create_excel","headers":["Name","Score"]},{"type":"add_row","values":["Tom",90]}]}

Input: "build a spreadsheet for grades and put Alex 85"
Output: {"steps":[{"type":"create_excel","headers":["Name","Score"]},{"type":"add_row","values":["Alex",85]}]}

Input: "make a score sheet"
Output: {"type":"create_excel","headers":["Name","Score"]}

--- ATTENDANCE ---
Input: "Make an attendance table and add Alice present"
Output: {"steps":[{"type":"create_excel","headers":["Name","Date","Status"]},{"type":"add_row","values":["Alice","","Present"]}]}

Input: "create an attendance sheet and mark Bob absent and Carol present"
Output: {"steps":[{"type":"create_excel","headers":["Name","Date","Status"]},{"type":"add_row","values":["Bob","","Absent"]},{"type":"add_row","values":["Carol","","Present"]}]}

Input: "build a table for attendance"
Output: {"type":"create_excel","headers":["Name","Date","Status"]}

--- INVENTORY ---
Input: "Create an inventory list and add keyboard 10"
Output: {"steps":[{"type":"create_excel","headers":["Item","Quantity"]},{"type":"add_row","values":["Keyboard",10]}]}

Input: "make a stock sheet and insert mouse 5 and monitor 2"
Output: {"steps":[{"type":"create_excel","headers":["Item","Quantity"]},{"type":"add_row","values":["Mouse",5]},{"type":"add_row","values":["Monitor",2]}]}

Input: "create an inventory spreadsheet"
Output: {"type":"create_excel","headers":["Item","Quantity"]}

--- EXPENSES ---
Input: "Make an expense table and add lunch 25"
Output: {"steps":[{"type":"create_excel","headers":["Item","Cost"]},{"type":"add_row","values":["Lunch",25]}]}

Input: "create a spending sheet and add taxi 15 and dinner 40"
Output: {"steps":[{"type":"create_excel","headers":["Item","Cost"]},{"type":"add_row","values":["Taxi",15]},{"type":"add_row","values":["Dinner",40]}]}

Input: "make an expense tracker"
Output: {"type":"create_excel","headers":["Item","Cost"]}

--- PROJECT MANAGEMENT ---
Input: "Create a project task list and add build API pending"
Output: {"steps":[{"type":"create_excel","headers":["Task","Owner","Status","Due Date"]},{"type":"add_row","values":["Build Api","","Pending",""]}]}

Input: "make a task sheet and add design UI in progress and write tests pending"
Output: {"steps":[{"type":"create_excel","headers":["Task","Owner","Status","Due Date"]},{"type":"add_row","values":["Design Ui","","In Progress",""]},{"type":"add_row","values":["Write Tests","","Pending",""]}]}

Input: "create a project tracker"
Output: {"type":"create_excel","headers":["Task","Owner","Status","Due Date"]}

--- MEETING NOTES ---
Input: "Create a meeting log and add AI project discussion"
Output: {"steps":[{"type":"create_excel","headers":["Topic","Date","Participants"]},{"type":"add_row","values":["AI Project Discussion","",""]}]}

Input: "make a meeting notes sheet and add weekly standup"
Output: {"steps":[{"type":"create_excel","headers":["Topic","Date","Participants"]},{"type":"add_row","values":["Weekly Standup","",""]}]}

Input: "create an agenda table"
Output: {"type":"create_excel","headers":["Topic","Date","Participants"]}

--- BUDGET TRACKING ---
Input: "Create a budget sheet and add marketing 5000"
Output: {"steps":[{"type":"create_excel","headers":["Category","Budget","Actual"]},{"type":"add_row","values":["Marketing",5000,""]}]}

Input: "make a finance tracker and add IT 3000 and HR 2000"
Output: {"steps":[{"type":"create_excel","headers":["Category","Budget","Actual"]},{"type":"add_row","values":["IT",3000,""]},{"type":"add_row","values":["HR",2000,""]}]}

Input: "create a budget spreadsheet"
Output: {"type":"create_excel","headers":["Category","Budget","Actual"]}

--- OTHER ---
Input: "add row Tom 90"
Output: {"type":"add_row","values":["Tom",90]}

Input: "create file with headers name, age"
Output: {"type":"create_excel","headers":["Name","Age"]}

Input: "delete the 2nd row"
Output: {"type":"delete_row","index":2}

Input: "write 100 to B3"
Output: {"type":"write_cell","cell":"B3","value":100}

Input: "write formula SUM(A1:A10) in cell A11"
Output: {"type":"write_formula","cell":"A11","formula":"=SUM(A1:A10)"}

Input: "calculate total using sum formula in B10 for B2 to B9"
Output: {"type":"write_formula","cell":"B10","formula":"=SUM(B2:B9)"}

Input: "创建成绩表并添加 Tom 90 和 Alex 85"
Output: {"steps":[{"type":"create_excel","headers":["Name","Score"]},{"type":"add_row","values":["Tom",90]},{"type":"add_row","values":["Alex",85]}]}

Input: "what is in cell B3"
Output: {"type":"read_cell","cell":"B3"}

Input: "switch to Sheet2"
Output: {"type":"switch_sheet","sheet_name":"Sheet2"}

Input: "create a new sheet called Sales"
Output: {"type":"create_sheet","sheet_name":"Sales"}

Input: "undo"
Output: {"type":"undo"}
"""

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        "max_tokens": 512,
        "temperature": 0.0,
    }

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    def _get_generated_text(body, raw_text):
        # Handle common Groq response shapes and return a best-effort generated string
        if isinstance(body, dict):
            # outputs -> content -> text
            outs = body.get('outputs') or body.get('output')
            if isinstance(outs, list):
                for o in outs:
                    if isinstance(o, dict):
                        content = o.get('content')
                        if isinstance(content, list):
                            for c in content:
                                if isinstance(c, dict) and 'text' in c:
                                    return c.get('text')
                                if isinstance(c, str):
                                    return c
                        if 'text' in o and isinstance(o.get('text'), str):
                            return o.get('text')
                        if 'content' in o and isinstance(o.get('content'), str):
                            return o.get('content')
            # top-level keys
            for k in ('text', 'result', 'response'):
                if k in body and isinstance(body[k], str):
                    return body[k]

        if isinstance(body, str):
            return body
        if isinstance(body, list) and body:
            try:
                return '\n'.join([str(x) for x in body])
            except Exception:
                pass

        return raw_text or ''

    try:
        log.info('_parse_groq_direct: calling Groq model=%s url=%s prompt_len=%d',
                 GROQ_MODEL, GROQ_API_URL, len(text))
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=GROQ_TIMEOUT)
        if resp.status_code in _GROQ_AUTH_CODES:
            log.error(
                '_parse_groq_direct: Groq returned %d — the model "%s" may be deprecated '
                'or your API key is invalid.  Set GROQ_MODEL to a supported model '
                '(e.g. llama3-8b-8192) in backend/.env',
                resp.status_code, GROQ_MODEL,
            )
            return {"type": "unknown", "raw": text}
        resp.raise_for_status()
        log.debug('_parse_groq_direct: Groq status=%d', resp.status_code)
        try:
            body = resp.json()
        except Exception:
            body = None

        # OpenAI-compatible response: choices[0].message.content
        generated = ''
        if isinstance(body, dict):
            choices = body.get('choices')
            if isinstance(choices, list) and choices:
                msg = choices[0].get('message', {})
                generated = msg.get('content', '')
        if not generated:
            generated = _get_generated_text(body, resp.text if resp is not None else '')

        if generated:
            # Strip <think>...</think> reasoning blocks before JSON extraction.
            # Some Groq-served models (e.g. reasoning variants) emit these before
            # the JSON answer.  Leaving them in causes RobustJsonExtractor to
            # search through kilobytes of reasoning text before finding the JSON.
            _think_re = re.compile(r'<think>[\s\S]*?</think>', re.IGNORECASE)
            generated = _think_re.sub('', generated).strip()
            log.debug('_parse_groq_direct: generated text=%r', generated[:200])
            try:
                result = RobustJsonExtractor.extract(generated)
                validated = _validate_llm_result(result, text)
                planned = _plan_and_validate(validated, text)
                log.info('_parse_groq_direct: extracted action=%s', planned)
                return planned
            except Exception:
                log.warning('_parse_groq_direct: JSON extraction failed from generated text')
                # If the response body itself looks like an action dict, return it
                if isinstance(body, dict) and any(k in body for k in ('action', 'type', 'headers', 'values', 'index', 'name')):
                    return _plan_and_validate(_validate_llm_result(body, text), text)
    except Exception as exc:
        log.error('_parse_groq_direct: Groq request failed: %s', exc)

    # Do NOT fall back to heuristic here — let route_llm try Ollama first.
    # Heuristic fallback is the last resort in the router chain.
    log.warning('_parse_groq_direct: Groq LLM parsing failed, returning unknown')
    return {"type": "unknown", "raw": text}


_ALLOWED_TYPES = {
    # core data operations
    'create_excel',
    'create_sheet_and_add_row',
    'add_row',
    'delete_row',
    'insert_row_at',
    'insert_column_at',
    'delete_column',
    'delete_cell',
    'delete_range',
    'add_column',
    'write_cell',
    'insert_cell',
    'write_formula',
    'write_sum_formula',
    'write_average_formula',
    'calculate_row_sum',
    'calculate_average',
    'open_file',
    'open_excel_app',
    'close_excel_app',
    'save',
    # formatting / structure
    'format_cells',
    'create_chart',
    'freeze_panes',
    'unfreeze_panes',
    'merge_cells',
    'unmerge_cells',
    'set_column_width',
    # search / sort / filter
    'find_replace',
    'sort_range',
    'filter_table',
    'clear_filter',
    'read_range',
    # sheet management
    'create_sheet',
    'switch_sheet',
    'delete_sheet',
    'list_sheets',
    'rename_sheet',
    # file management
    'rename_file',
    # undo / restore
    'undo',
    'undo_last',
    # read operations
    'read_cell',
    # sentinel
    'unknown',
}


def _is_number_like(v):
    return isinstance(v, (int, float)) or (isinstance(v, str) and re.fullmatch(r'-?\d+(\.\d+)?', v.strip()))


def _validate_single_action(action: dict) -> dict | None:
    if not isinstance(action, dict):
        return None
    atype = action.get('type')
    if atype not in _ALLOWED_TYPES:
        log.warning('_validate_single_action: rejected unknown type %r', atype)
        return None
    if atype == 'unknown':
        return {'type': 'unknown', 'raw': action.get('raw', '')}
    if atype == 'create_excel':
        headers = action.get('headers')
        # Allow empty headers — user may want a blank spreadsheet.
        if not isinstance(headers, list):
            return None
    elif atype == 'create_sheet_and_add_row':
        headers = action.get('headers')
        values = action.get('values')
        if not isinstance(headers, list) or not headers or not isinstance(values, list) or not values:
            return None
    elif atype == 'add_row':
        if not isinstance(action.get('values'), list) or not action.get('values'):
            return None
    elif atype == 'delete_row':
        # Accept both 'index' and 'row' — LLM sometimes returns 'row' instead.
        idx = action.get('index')
        if idx is None:
            idx = action.get('row')
        # index must be an integer ≥ 2; index 1 is always the header row
        if not isinstance(idx, int) or idx < 2:
            log.warning('_validate_single_action: delete_row rejected index=%r '
                        '(must be integer >= 2 to protect the header row)', idx)
            return None
        # Normalise: downstream code (kwarg map, Pydantic schema) all use 'index'.
        if 'index' not in action:
            action = dict(action, index=idx)
    elif atype == 'insert_row_at':
        idx = action.get('index')
        if idx is None:
            idx = action.get('row')
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            return None
        if idx < 1:
            return None
        # idx == 1 is allowed here; the controller rejects it with a clear message.
        if 'index' not in action or action['index'] != idx:
            action = dict(action, index=idx)
    elif atype == 'insert_column_at':
        col = action.get('column') or action.get('col')
        if not col or not str(col).strip():
            return None
        if 'column' not in action:
            action = dict(action, column=col)
    elif atype == 'delete_column':
        col = action.get('column')
        if not isinstance(col, str) or not col.strip():
            return None
    elif atype == 'add_column':
        name = action.get('name')
        if not isinstance(name, str) or not name.strip():
            return None
    elif atype == 'write_cell':
        cell = action.get('cell')
        if not isinstance(cell, str) or not re.fullmatch(r'[A-Za-z]+\d+', cell.strip()):
            return None
        if 'value' not in action:
            return None
    elif atype == 'calculate_row_sum':
        idx = action.get('index')
        if not isinstance(idx, int):
            return None
    elif atype == 'write_formula':
        cell = action.get('cell')
        if not isinstance(cell, str) or not re.fullmatch(r'[A-Za-z]+\d+', cell.strip()):
            return None
        formula = action.get('formula')
        if not isinstance(formula, str) or not formula.strip():
            return None
        # Validate formula contains only whitelisted functions to prevent
        # dangerous calls like =EXEC(), =INDIRECT(), =HYPERLINK() etc.
        try:
            from pipeline_sanitizer import is_formula_safe
            if not is_formula_safe(formula):
                log.warning('_validate_single_action: write_formula rejected '
                            'due to non-whitelisted function: %r', formula[:80])
                return None
        except ImportError:
            pass  # sanitizer not yet available — fail open, log warning

    elif atype == 'write_sum_formula':
        # Requires a cell range (e.g. "B2:B10") and a target cell (e.g. "C2").
        range_ref = action.get('range_ref') or action.get('range')
        if not isinstance(range_ref, str) or not range_ref.strip():
            return None
        cell = action.get('cell')
        if not isinstance(cell, str) or not re.fullmatch(r'[A-Za-z]+\d+', cell.strip()):
            return None
        # Normalise: always store under 'range_ref' so the kwarg map hits cleanly.
        if 'range_ref' not in action:
            action = dict(action, range_ref=range_ref)

    elif atype == 'write_average_formula':
        # Requires a cell range (e.g. "B2:B10") and a target cell (e.g. "C2").
        range_ref = action.get('range_ref') or action.get('range')
        if not isinstance(range_ref, str) or not range_ref.strip():
            return None
        cell = action.get('cell')
        if not isinstance(cell, str) or not re.fullmatch(r'[A-Za-z]+\d+', cell.strip()):
            return None
        # Normalise: always store under 'range_ref' so the kwarg map hits cleanly.
        if 'range_ref' not in action:
            action = dict(action, range_ref=range_ref)

    elif atype == 'create_chart':
        # data_range is no longer required here — the controller auto-detects
        # from the sheet when omitted (supports "create a bar chart" voice commands).
        pass
    elif atype == 'format_cells':
        if not action.get('range') and not action.get('cell_range'):
            return None
    elif atype == 'find_replace':
        if action.get('find_value') is None and action.get('find') is None:
            return None
    elif atype == 'sort_range':
        if not action.get('column') and not action.get('col'):
            return None
    elif atype == 'filter_table':
        if not action.get('column'):
            return None
    elif atype in ('merge_cells', 'unmerge_cells', 'read_range'):
        if not action.get('range') and not action.get('cell_range'):
            return None
    elif atype == 'rename_sheet':
        if not action.get('old_name') or not action.get('new_name'):
            return None
    elif atype == 'rename_file':
        if not action.get('new_name'):
            return None
    elif atype in ('create_sheet', 'switch_sheet', 'delete_sheet'):
        if not action.get('sheet_name'):
            return None
    # Types with no required fields beyond 'type': open_file, save, list_sheets,
    # set_column_width, freeze_panes, insert_cell — pass through as-is.
    return action


def _validate_llm_result(result, raw_text: str):
    if not isinstance(result, dict):
        log.warning('[PARSE] _validate_llm_result: non-dict result type=%s — returning unknown',
                    type(result).__name__)
        return {'type': 'unknown', 'raw': raw_text}

    if 'steps' in result:
        steps = result.get('steps')
        if not isinstance(steps, list) or not steps:
            return {'type': 'unknown', 'raw': raw_text}
        validated_steps = []
        for step in steps:
            validated = _validate_single_action(step)
            if not validated or validated.get('type') == 'unknown':
                # A single bad step does not invalidate the whole plan.
                # Log and skip it so the remaining valid steps can execute.
                log.warning(
                    '_validate_llm_result: skipping invalid step %r in multi-step plan',
                    step,
                )
                continue
            validated_steps.append(validated)
        if not validated_steps:
            return {'type': 'unknown', 'raw': raw_text}
        # Run each validated step through Pydantic schema for type coercion.
        coerced_steps = [validate_command(s) for s in validated_steps]
        coerced_steps = [s for s in coerced_steps if s.get('type') != 'unknown']
        if not coerced_steps:
            log.warning('[PARSE] _validate_llm_result: all steps rejected after schema validation')
            return {'type': 'unknown', 'raw': raw_text}
        log.info('[PARSE] _validate_llm_result: multi-step plan OK  steps=%d', len(coerced_steps))
        return {'steps': coerced_steps}

    validated = _validate_single_action(result)
    if validated:
        # Run through Pydantic schema for type coercion + formal validation.
        # validate_command() is a no-op when pydantic is not installed.
        coerced = validate_command(validated)
        log.info('[PARSE] _validate_llm_result: single action OK  type=%s', coerced.get('type'))
        return coerced
    log.warning('[PARSE] _validate_llm_result: action type=%r failed validation — returning unknown',
                result.get('type'))
    return {'type': 'unknown', 'raw': raw_text}


_DOMAIN_NAME_BY_HEADERS = {
    ('Name', 'Score'): 'student_scores',
    ('Name', 'Date', 'Status'): 'attendance',
    ('Item', 'Quantity'): 'inventory',
    ('Item', 'Cost'): 'expenses',
    ('Category', 'Budget', 'Actual'): 'budget_tracking',
    ('Product', 'Quantity', 'Revenue'): 'sales',
    ('Name', 'Department', 'Salary'): 'employees',
    ('Task', 'Owner', 'Status', 'Due Date'): 'project_management',
    ('Topic', 'Date', 'Participants'): 'meeting_notes',
}

_HEADERS_BY_DOMAIN_NAME = {
    'student_scores': ['Name', 'Score'],
    'attendance': ['Name', 'Date', 'Status'],
    'inventory': ['Item', 'Quantity'],
    'expenses': ['Item', 'Cost'],
    'budget_tracking': ['Category', 'Budget', 'Actual'],
    'sales': ['Product', 'Quantity', 'Revenue'],
    'employees': ['Name', 'Department', 'Salary'],
    'project_management': ['Task', 'Owner', 'Status', 'Due Date'],
    'meeting_notes': ['Topic', 'Date', 'Participants'],
    'generic_table': ['Column1', 'Column2'],
}

_NUMERIC_HEADER_NAMES = {
    'score', 'quantity', 'cost', 'budget', 'actual', 'revenue', 'salary',
}


def _coerce_number(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        t = value.strip().replace(',', '')
        if re.fullmatch(r'-?\d+', t):
            return int(t)
        if re.fullmatch(r'-?\d+\.\d+', t):
            return float(t)
    return None


def _infer_domain_name(raw_text: str, headers=None):
    if isinstance(headers, list) and headers:
        mapped = _DOMAIN_NAME_BY_HEADERS.get(tuple(headers))
        if mapped:
            return mapped
    lower = (raw_text or '').lower()
    inferred_headers, domain_key = _infer_domain_headers(lower)
    if inferred_headers:
        mapped = _DOMAIN_NAME_BY_HEADERS.get(tuple(inferred_headers))
        if mapped:
            return mapped
    if domain_key == 'budget':
        return 'budget_tracking'
    if domain_key == 'project':
        return 'project_management'
    if domain_key == 'meeting':
        return 'meeting_notes'
    if domain_key:
        return domain_key
    return 'generic_table'


def _headers_for_domain(domain: str):
    return _HEADERS_BY_DOMAIN_NAME.get(domain, ['Column1', 'Column2'])


def _infer_headers_from_first_row(first_row):
    if not isinstance(first_row, list) or not first_row:
        return ['Column1', 'Column2']
    if len(first_row) >= 2 and _coerce_number(first_row[1]) is not None:
        headers = ['Name', 'Score']
        for i in range(2, len(first_row)):
            headers.append(f'Column{i + 1}')
        return headers
    return [f'Column{i + 1}' for i in range(len(first_row))]


def _normalize_row(values, headers):
    vals = list(values) if isinstance(values, list) else []
    target_len = len(headers)
    numeric_indexes = {
        i for i, h in enumerate(headers)
        if isinstance(h, str) and h.strip().lower() in _NUMERIC_HEADER_NAMES
    }

    # Trim overflow values so row shape remains aligned with headers.
    vals = vals[:target_len]

    # Fill missing values safely.
    while len(vals) < target_len:
        idx = len(vals)
        vals.append(0 if idx in numeric_indexes else '')

    # Import lazily to avoid circular dependency at module level
    try:
        from pipeline_sanitizer import sanitize_cell_value as _scv
    except ImportError:
        _scv = None

    normalized = []
    for i, val in enumerate(vals):
        if i in numeric_indexes:
            num = _coerce_number(val)
            normalized.append(0 if num is None else num)
        else:
            # Apply formula-injection sanitization to all string values
            safe_val = '' if val is None else val
            if _scv is not None:
                safe_val = _scv(safe_val)
            normalized.append(safe_val)
    return normalized


def build_intermediate_plan(action: dict, raw_text: str) -> dict:
    """Build a domain-aware intermediate plan from model output/action dict."""
    if not isinstance(action, dict):
        return {
            'intent': 'unknown',
            'domain': 'generic_table',
            'headers': [],
            'rows': [],
        }

    steps = action.get('steps') if isinstance(action.get('steps'), list) else []
    headers = []
    rows = []

    if steps:
        for step in steps:
            if not isinstance(step, dict):
                continue
            if step.get('type') == 'create_excel' and isinstance(step.get('headers'), list) and step.get('headers'):
                headers = list(step.get('headers'))
            elif step.get('type') == 'add_row' and isinstance(step.get('values'), list):
                rows.append(list(step.get('values')))
    else:
        atype = action.get('type')
        if atype == 'create_excel' and isinstance(action.get('headers'), list):
            headers = list(action.get('headers'))
        elif atype == 'add_row' and isinstance(action.get('values'), list):
            rows.append(list(action.get('values')))
        elif atype == 'create_sheet_and_add_row':
            if isinstance(action.get('headers'), list):
                headers = list(action.get('headers'))
            if isinstance(action.get('values'), list):
                rows.append(list(action.get('values')))

    domain = _infer_domain_name(raw_text, headers if headers else None)
    if not headers and rows:
        headers = _headers_for_domain(domain)
        if domain == 'generic_table':
            headers = _infer_headers_from_first_row(rows[0])

    if headers and rows:
        intent = f'create_{domain}_and_add_rows'
    elif headers:
        intent = f'create_{domain}_table'
    elif rows:
        intent = f'add_rows_to_{domain}_table'
    else:
        intent = 'unknown'

    return {
        'intent': intent,
        'domain': domain,
        'headers': headers,
        'rows': rows,
    }


def validate_execution_plan(plan: dict) -> dict:
    """Validate and auto-repair an execution plan into safe executable steps."""
    if not isinstance(plan, dict):
        return {'type': 'unknown', 'raw': ''}

    raw_steps = plan.get('steps') if isinstance(plan.get('steps'), list) else None
    domain = plan.get('domain') if isinstance(plan.get('domain'), str) else 'generic_table'
    headers = plan.get('headers') if isinstance(plan.get('headers'), list) else []
    rows = plan.get('rows') if isinstance(plan.get('rows'), list) else []

    if raw_steps is None:
        raw_steps = []
        if headers:
            raw_steps.append({'type': 'create_excel', 'headers': headers})
        for r in rows:
            if isinstance(r, list):
                raw_steps.append({'type': 'add_row', 'values': r})

    normalized_steps = []
    create_inserted = False
    effective_headers = list(headers) if headers else []

    # Detect if the original plan explicitly includes a create_excel step.
    # If not (e.g. standalone "add row"), skip auto-prepending create_excel
    # so that the executor can append to an existing file.
    has_explicit_create = any(
        isinstance(s, dict) and s.get('type') == 'create_excel'
        for s in raw_steps
    )

    for step in raw_steps:
        if not isinstance(step, dict):
            continue
        stype = step.get('type')

        if stype == 'create_excel':
            # Prevent duplicate file creation by keeping only the first create step.
            if create_inserted:
                continue
            candidate_headers = step.get('headers') if isinstance(step.get('headers'), list) else []
            if candidate_headers:
                effective_headers = candidate_headers
            elif not effective_headers and 'headers' not in step:
                # Only inject domain defaults when the step omitted the key entirely.
                # An explicit empty list (headers=[]) means "blank sheet" — honour it.
                effective_headers = _headers_for_domain(domain)
            create_step: dict = {'type': 'create_excel', 'headers': effective_headers}
            if step.get('filename'):
                create_step['filename'] = step['filename']
            normalized_steps.append(create_step)
            create_inserted = True
            continue

        if stype == 'add_row':
            values = step.get('values') if isinstance(step.get('values'), list) else []
            if not create_inserted and has_explicit_create:
                if not effective_headers:
                    if values:
                        effective_headers = _infer_headers_from_first_row(values)
                    else:
                        effective_headers = _headers_for_domain(domain)
                normalized_steps.append({'type': 'create_excel', 'headers': effective_headers})
                create_inserted = True

            normalized_values = _normalize_row(values, effective_headers) if effective_headers else list(values)
            normalized_steps.append({'type': 'add_row', 'values': normalized_values})
            continue

        normalized_steps.append(step)

    if not normalized_steps:
        return {'type': 'unknown', 'raw': plan.get('raw_text', '') if isinstance(plan.get('raw_text', ''), str) else ''}
    return {'steps': normalized_steps}


def _plan_and_validate(action: dict, raw_text: str) -> dict:
    if not isinstance(action, dict) or action.get('type') == 'unknown':
        return action
    plan = build_intermediate_plan(action, raw_text)
    if isinstance(action.get('steps'), list):
        plan['steps'] = action.get('steps')
    else:
        atype = action.get('type')
        if atype == 'create_excel':
            _cs: dict = {'type': 'create_excel', 'headers': action.get('headers', [])}
            if action.get('filename'):
                _cs['filename'] = action['filename']
            plan['steps'] = [_cs]
        elif atype == 'add_row':
            plan['steps'] = [{'type': 'add_row', 'values': action.get('values', [])}]
        elif atype == 'create_sheet_and_add_row':
            plan['steps'] = [
                {'type': 'create_excel', 'headers': action.get('headers', [])},
                {'type': 'add_row', 'values': action.get('values', [])},
            ]
        elif atype in ('insert_row_at', 'insert_column_at', 'rename_file'):
            # Single-op bypass: skip steps wrapping so _dispatch_single is used
            # directly.  For insert_row/column_at this surfaces positional
            # rejection messages.  For rename_file this also ensures:
            #   (a) _update_memory receives the real result dict (with new_name)
            #       rather than the generic steps_executed wrapper, so
            #       memory.set_active_file(new_name) fires correctly; and
            #   (b) error results (same_name, name_taken) reach the caller
            #       as clean rejections, not as transaction_rolled_back.
            return _validate_llm_result(action, raw_text)
        elif atype:
            plan['steps'] = [action]
    repaired = validate_execution_plan(plan)
    return _validate_llm_result(repaired, raw_text)


# Domain header inference table for heuristic fallback
_DOMAIN_HEADERS = [
    # (list of trigger keywords, inferred headers)
    (['score', 'scores', 'grade', 'grades', 'mark', 'marks', 'result', 'results', 'exam'],
     ['Name', 'Score']),
    (['attendance', 'present', 'absent', 'roll'],
     ['Name', 'Date', 'Status']),
    (['inventory', 'stock', 'item', 'items', 'product', 'products'],
     ['Item', 'Quantity']),
    (['expense', 'expenses', 'cost', 'costs', 'payment'],
     ['Item', 'Cost']),
    (['budget', 'finance', 'spending'],
     ['Category', 'Budget', 'Actual']),
    (['sales', 'revenue', 'order', 'orders'],
     ['Product', 'Quantity', 'Revenue']),
    (['employee', 'staff', 'hr', 'payroll'],
     ['Name', 'Department', 'Salary']),
    (['project', 'task', 'todo', 'tasks', 'milestone'],
     ['Task', 'Owner', 'Status', 'Due Date']),
    (['meeting', 'discussion', 'agenda'],
     ['Topic', 'Date', 'Participants']),
]

_ATTENDANCE_STOP_WORDS = {
    'attendance', 'sheet', 'table', 'excel', 'spreadsheet', 'create', 'make',
    'build', 'new', 'and', 'add', 'insert', 'mark', 'record', 'for', 'the', 'a',
}
_SCORE_STOP_WORDS = {
    'sheet', 'excel', 'score', 'scores', 'student', 'students', 'table', 'with',
    'and', 'add', 'insert', 'create', 'make', 'build', 'new', 'for', 'the', 'a',
    'grade', 'grades', 'spreadsheet',
}
_ITEM_STOP_WORDS = {
    'inventory', 'stock', 'expense', 'expenses', 'sheet', 'table', 'excel',
    'spreadsheet', 'create', 'make', 'build', 'new', 'and', 'add', 'insert',
    'list', 'for', 'the', 'a', 'an',
}
_PROJECT_STOP_WORDS = {
    'project', 'task', 'tasks', 'todo', 'milestone', 'sheet', 'table', 'excel',
    'spreadsheet', 'tracker', 'create', 'make', 'build', 'new', 'and', 'add',
    'insert', 'list', 'for', 'the', 'a', 'an',
}
_MEETING_STOP_WORDS = {
    'meeting', 'log', 'notes', 'sheet', 'table', 'excel', 'spreadsheet',
    'create', 'make', 'build', 'new', 'and', 'add', 'insert', 'for', 'the',
    'a', 'an', 'agenda',
}
_BUDGET_STOP_WORDS = {
    'budget', 'finance', 'spending', 'sheet', 'table', 'excel', 'spreadsheet',
    'tracker', 'create', 'make', 'build', 'new', 'and', 'add', 'insert',
    'for', 'the', 'a', 'an',
}


def _infer_domain_headers(lower: str):
    """Return (headers, domain) for the first matching domain, or (None, None).

    Checks the first 6 words first (the command context, e.g. "create a meeting log")
    so that data values later in the sentence (e.g. "AI project discussion") cannot
    hijack the domain.  Falls back to a full-sentence word-boundary scan.
    """
    first_words = ' '.join(lower.split()[:6])
    for keywords, headers in _DOMAIN_HEADERS:
        if any(re.search(r'\b' + re.escape(kw) + r'\b', first_words) for kw in keywords):
            return headers, keywords[0]
    for keywords, headers in _DOMAIN_HEADERS:
        if any(re.search(r'\b' + re.escape(kw) + r'\b', lower) for kw in keywords):
            return headers, keywords[0]
    return None, None


def _heuristic_natural_language_parse(text: str):
    """Last-resort parser for common commands when LLM output is unusable."""
    if not text or not isinstance(text, str):
        return None
    # Strip trailing punctuation that STT (Whisper) appends when the user
    # trails off — e.g. "rename sheet style to book." fails \s*$ without this.
    t = text.strip().rstrip('.,!?…')
    lower = t.lower()

    # --- undo: absolute earliest guard — token-based, runs before every
    # other pattern so no regex or LLM call can ever intercept it.
    # Catches: "undo", "undo please", "undo that", "undo last action", etc.
    tokens = lower.strip().split()
    if tokens and tokens[0] == 'undo':
        return {'type': 'undo'}
    # Context-enriched form: "[file:data.xlsx sheet:Sheet1] undo"
    # _enrich_with_context() prepends a [file:… sheet:… columns:…] prefix which
    # shifts 'undo' away from tokens[0].  Strip the prefix once and recheck so
    # undo is always deterministic regardless of context enrichment.
    _ctx_stripped = re.sub(r'^\[.*?\]\s*', '', lower).strip()
    if _ctx_stripped.split() and _ctx_stripped.split()[0] == 'undo':
        return {'type': 'undo'}

    # --- undo: phrase variants — "go back" / "restore" / "revert" ----------
    if re.fullmatch(r'(?:go\s+back|restore|revert)', lower):
        return {'type': 'undo'}

    # --- read_cell: "read cell B3" / "what's in A1" / "go to B3" -----------
    # "go to [cell ref]" is treated as cell navigation, NOT sheet switch.
    # This check runs before switch_sheet_match so a cell ref like B3 is never
    # captured as a sheet name.  "go to sheet Sales" is safe: "Sales" has no
    # digits, so [A-Za-z]+\d+ won't match and we fall through to switch_sheet.
    read_cell_match = re.search(
        r'\b(?:read|get|show|what(?:\'s|\s+is)\s+in?|display|go\s+to)\s+(?:cell\s+)?([A-Za-z]+\d+)',
        t, re.IGNORECASE,
    )
    if read_cell_match:
        return {'type': 'read_cell', 'cell': read_cell_match.group(1).upper()}

    # --- switch_sheet: "switch to Sheet2" / "switch sheet Report" / "go to sheet Sales" ---
    switch_sheet_match = re.search(
        r'\b(?:switch\s+(?:to|sheet)|go\s+to|change\s+to|select|activate)\s+(?:sheet\s+)?["\']?(\w[\w\s]*?)["\']?\s*(?:sheet)?$',
        t, re.IGNORECASE,
    )
    if switch_sheet_match and not any(
        w in lower for w in ('create', 'make', 'new', 'add')
    ):
        sheet = switch_sheet_match.group(1).strip()
        # Reject if the captured name contains "excel" — this is almost certainly
        # an STT misrouting of an app-level command (e.g. "go to teacher Excel"
        # should not become switch_sheet "teacher Excel").
        if sheet and 'excel' not in sheet.lower():
            return {'type': 'switch_sheet', 'sheet_name': sheet}

    # --- create_sheet: "create sheet Sales" / "add a new sheet called Q1" -
    create_sheet_match = re.search(
        r'\b(?:create|add|make|new)\s+(?:a\s+)?(?:new\s+)?sheet\s+(?:called|named|titled)?\s*["\']?(\w[\w\s]*?)["\']?\s*$',
        t, re.IGNORECASE,
    )
    if create_sheet_match and not any(
        w in lower for w in ('spreadsheet', 'workbook', 'excel file')
    ):
        sheet = create_sheet_match.group(1).strip()
        if sheet:
            return {'type': 'create_sheet', 'sheet_name': sheet}

    # --- delete_sheet: "delete sheet Sales" / "remove sheet Q1" -------------
    delete_sheet_match = re.search(
        r'\b(?:delete|remove|drop)\s+sheet\s+["\']?(\w[\w\s]*?)["\']?\s*$',
        t, re.IGNORECASE,
    )
    if delete_sheet_match:
        sheet = delete_sheet_match.group(1).strip()
        if sheet:
            return {'type': 'delete_sheet', 'sheet_name': sheet}

    # --- rename_sheet: "rename sheet Sales to Report" -----------------------
    rename_sheet_match = re.search(
        r'\brename\s+sheet\s+["\']?(\w[\w\s]*?)["\']?\s+to\s+["\']?(\w[\w\s]*?)["\']?\s*$',
        t, re.IGNORECASE,
    )
    if rename_sheet_match:
        old_name = rename_sheet_match.group(1).strip()
        new_name = rename_sheet_match.group(2).strip()
        if old_name and new_name:
            return {'type': 'rename_sheet', 'old_name': old_name, 'new_name': new_name}

    # --- list_sheets: "list sheets" / "show all sheets" / "what sheets" ----
    if re.search(r'\b(?:list|show)\s+(?:all\s+)?sheets?\b|\bwhat\s+sheets?\b', lower):
        return {'type': 'list_sheets'}

    # --- rename_file: "rename file to report" / "rename this file budget" ----
    # Handles:
    #   "rename file to report.xlsx"
    #   "rename file demo.xlsx to report.xlsx"   (old_name to new_name)
    #   "rename this file to report"
    #   "rename workbook to budget.xlsx"
    rename_file_match = re.search(
        r'\brename\s+(?:this\s+)?(?:file|workbook|spreadsheet)\s+'
        r'(?:[\w][\w.\-]*\s+to\s+)?'   # optional: old_name to
        r'(?:to\s+)?'                   # optional: bare "to"
        r'["\']?([\w][\w\s.\-]*?)["\']?\s*$',
        t, re.IGNORECASE,
    )
    if rename_file_match:
        new_name = rename_file_match.group(1).strip()
        if new_name:
            return {'type': 'rename_file', 'new_name': new_name}

    # --- Explicit column extraction: "with columns X, Y, Z" or "columns X Y Z" ---
    explicit_cols_match = re.search(
        r'\bcolumns?\s+(.+)',
        t, re.IGNORECASE,
    )
    explicit_headers = None
    if explicit_cols_match:
        raw_cols = explicit_cols_match.group(1)
        # Split on commas, "and", or multiple spaces
        parts = re.split(r'[,\s]+and\s+|,\s*|\s{2,}', raw_cols)
        parts = [p.strip().strip('"\'') for p in parts if p.strip()]
        if parts:
            explicit_headers = [p.capitalize() for p in parts]

    is_create = any(w in lower for w in ('create', 'make', 'build', 'new'))
    is_add = any(w in lower for w in ('add', 'insert', 'put', 'mark', 'record'))

    # --- insert_column_at: "insert column C" / "insert column before B" /
    #     "insert before column B" / "insert column at AB".
    #     Fires ONLY when argument is a 1-3 char column letter with nothing after it.
    #     Anything longer (e.g. "City", "Score") falls through to add_column below.
    ins_col_match = re.search(
        r'\binsert\s+(?:a\s+)?(?:before\s+)?column\s+(?:(?:at|before|after)\s+)?([A-Za-z]{1,3})\s*$',
        t, re.IGNORECASE,
    )
    if ins_col_match and not is_create:
        raw_col = ins_col_match.group(1).strip().upper()
        # Accept only if it looks like a column letter, not a capitalized word
        # (heuristic: 1-2 chars are unambiguous column refs; 3 chars accepted too)
        try:
            from openpyxl.utils import column_index_from_string
            column_index_from_string(raw_col)  # raises if invalid
            return {'type': 'insert_column_at', 'column': raw_col}
        except Exception:
            pass  # fall through to add_column

    # --- add_column: "add column Name" / "add column Name Score Age" ---
    add_col_match = re.search(
        r'\b(?:add|insert|create)\s+(?:a\s+)?columns?\s+(?:(?:called|named|titled)\s+)?(.+?)\s*$',
        t, re.IGNORECASE,
    )
    if add_col_match and not is_create:
        raw = add_col_match.group(1).strip().strip('"\'')
        col_names = [n.strip().capitalize() for n in raw.split() if n.strip()]
        if len(col_names) == 1:
            return {'type': 'add_column', 'name': col_names[0]}
        elif len(col_names) > 1:
            return {'steps': [{'type': 'add_column', 'name': n} for n in col_names]}

    # --- delete_row: "delete row 3" ---
    del_row_match = re.search(r'\bdelete\s+(?:the\s+)?(?:row\s+)?(\d+)(?:st|nd|rd|th)?\s*(?:row)?', lower)
    if del_row_match:
        return {'type': 'delete_row', 'index': int(del_row_match.group(1))}

    # --- delete_column: "delete column C" / "remove column B" / "delete column 3" ---
    del_col_match = re.search(
        r'\b(?:delete|remove)\s+(?:the\s+)?column\s+([A-Za-z]+|\d+)\b',
        t, re.IGNORECASE,
    )
    if del_col_match:
        col_id = del_col_match.group(1)
        # Normalise: letters → upper-case, digits → keep as string (controller handles both)
        return {'type': 'delete_column', 'column': col_id.upper() if col_id.isalpha() else col_id}

    # --- delete_cell: "delete cell A1" / "delete cell B10" ---
    del_cell_match = re.fullmatch(r'delete\s+cell\s+([a-z]+\d+)', lower)
    if del_cell_match:
        return {'type': 'delete_cell', 'cell': del_cell_match.group(1).upper()}

    # --- clear_cell: "clear cell A1" / "erase cell B3" / "empty cell C2" ---
    clear_cell_match = re.search(
        r'\b(?:clear|empty|erase)\s+(?:the\s+)?cell\s+([A-Za-z]+\d+)\b',
        t, re.IGNORECASE,
    )
    if clear_cell_match:
        return {'type': 'clear_cell', 'cell': clear_cell_match.group(1).upper()}

    # --- delete_range: "delete A2:C2" / "delete range A1:B5" ---
    del_range_match = re.search(
        r'\bdelete\s+(?:range\s+)?([A-Za-z]+\d+:[A-Za-z]+\d+)\b',
        t, re.IGNORECASE,
    )
    if del_range_match:
        return {'type': 'delete_range', 'range': del_range_match.group(1).upper()}

    # --- write_formula: "write formula SUM(A1:A10) in A11" / "put =AVERAGE(B2:B9) in B10" ---
    formula_match = re.search(
        r'(?:write|put|set|add|insert|use)\s+(?:(?:a\s+)?formula\s+)?'
        r'=?\s*((?:SUM|AVERAGE|AVG|COUNT|MAX|MIN|IF|VLOOKUP|SUMIF|COUNTIF)'
        r'\s*\([^)]+\))\s*'
        r'(?:to|in|into|at|for)?\s*(?:cell\s+)?([A-Za-z]+\d+)',
        t, re.IGNORECASE,
    )
    if formula_match:
        func = formula_match.group(1).strip()
        cell = formula_match.group(2).upper()
        return {'type': 'write_formula', 'cell': cell, 'formula': '=' + func.upper()}

    # --- write_formula (reverse order): "calculate total in B10 for B2:B9" ---
    formula_rev = re.search(
        r'(?:calculate|compute|find)\s+(?:the\s+)?(?:total|sum|average|avg|count|max|min)'
        r'\s+(?:(?:using|with)\s+(?:a\s+)?(?:sum|average|avg|count|max|min)\s+formula\s+)?'
        r'(?:in|at|to)\s*(?:cell\s+)?([A-Za-z]+\d+)\s+'
        r'(?:for|from|of)\s+([A-Za-z]+\d+)\s*(?:to|:|-)\s*([A-Za-z]+\d+)',
        t, re.IGNORECASE,
    )
    if formula_rev:
        cell = formula_rev.group(1).upper()
        start = formula_rev.group(2).upper()
        end = formula_rev.group(3).upper()
        # Infer function from keyword
        kw = re.search(r'(sum|total|average|avg|count|max|min)', t, re.IGNORECASE)
        func = 'SUM'
        if kw:
            fn = kw.group(1).upper()
            func = {'AVG': 'AVERAGE', 'TOTAL': 'SUM'}.get(fn, fn)
        return {'type': 'write_formula', 'cell': cell, 'formula': f'={func}({start}:{end})'}

    # --- write_cell: "write cell C2 score" / "set cell A1 hello" ---
    # This pattern handles the natural "write cell CELLREF VALUE" form where
    # the cell reference comes BEFORE the value with no preposition.
    # Matched first — it is more specific than the preposition-based pattern.
    # The value is kept as-is (no int/float coercion) so text like "score"
    # is stored as the string "score", not converted to a number.
    write_cell_direct = re.search(
        r'\b(?:write|set|update)\s+cell\s+([A-Za-z]+\d+)\s+(.+)',
        t, re.IGNORECASE,
    )
    if write_cell_direct:
        cell = write_cell_direct.group(1).upper()
        raw_val = write_cell_direct.group(2).strip().strip('"\'')
        return {'type': 'write_cell', 'cell': cell, 'value': raw_val}

    # --- write_cell: "write 100 to B3" / "put hello in A1" ---
    write_match = re.search(
        r'\b(?:write|put|set)\s+(.+?)\s+(?:to|in|into|at)\s+([A-Za-z]+\d+)',
        t, re.IGNORECASE,
    )
    if write_match:
        raw_val = write_match.group(1).strip().strip('"\'')
        cell = write_match.group(2).upper()
        try:
            val = int(raw_val)
        except ValueError:
            try:
                val = float(raw_val)
            except ValueError:
                val = raw_val
        return {'type': 'write_cell', 'cell': cell, 'value': val}

    # --- write_cell fallback: "write C1 100" / "write B3 hello" (no "cell" keyword) ---
    # Catches voice output after phonetic correction where "write cell" collapsed
    # to "write CELLREF" because the user omitted or STT dropped the word "cell".
    # Runs AFTER the two patterns above so it only fires when they both missed.
    write_cell_bare = re.search(
        r'\bwrite\s+([A-Za-z]\d+)\s+(.+)',
        t, re.IGNORECASE,
    )
    if write_cell_bare:
        cell = write_cell_bare.group(1).upper()
        raw_val = write_cell_bare.group(2).strip().strip('"\'')
        try:
            bare_val: object = int(raw_val)
        except ValueError:
            try:
                bare_val = float(raw_val)
            except ValueError:
                bare_val = raw_val
        return {'type': 'write_cell', 'cell': cell, 'value': bare_val}

    # --- calculate_row_sum: "calculate sum of row 2" ---
    sum_match = re.search(r'\b(?:sum|total)\s+(?:of\s+)?(?:row\s+)?(\d+)', lower)
    if sum_match:
        return {'type': 'calculate_row_sum', 'index': int(sum_match.group(1))}

    # --- write_sum_formula: "sum B2:B10 into C2" / "calculate row sum B2:B10 in C2" ---
    # Matched AFTER calculate_row_sum (which handles row-index form) so the two
    # patterns do not conflict.  Requires both a range reference and a target cell.
    sum_formula_match = re.search(
        r'\b(?:sum|calculate\s+(?:row\s+)?sum|write\s+sum)\s+'
        r'([A-Za-z]+\d+:[A-Za-z]+\d+)'       # range like B2:B10
        r'\s+(?:in(?:to)?|to)\s+'
        r'([A-Za-z]+\d+)',                     # target cell like C2
        t, re.IGNORECASE,
    )
    if sum_formula_match:
        range_ref = sum_formula_match.group(1).upper()
        cell = sum_formula_match.group(2).upper()
        return {'type': 'write_sum_formula', 'range_ref': range_ref, 'cell': cell}

    # --- write_average_formula: "average B2:B10 into C2" / "calculate average B2:B10 in C2" ---
    # Matched BEFORE the column-level calculate_average so the range+cell form
    # takes priority over the bare column form.
    avg_formula_match = re.search(
        r'\b(?:average|calculate\s+average|write\s+average)\s+'
        r'([A-Za-z]+\d+:[A-Za-z]+\d+)'        # range like B2:B10
        r'\s+(?:in(?:to)?|to)\s+'
        r'([A-Za-z]+\d+)',                      # target cell like C2
        t, re.IGNORECASE,
    )
    if avg_formula_match:
        range_ref = avg_formula_match.group(1).upper()
        cell = avg_formula_match.group(2).upper()
        return {'type': 'write_average_formula', 'range_ref': range_ref, 'cell': cell}

    # --- calculate_average: "calculate average" / "what's the average of column B" ---
    avg_match = re.search(
        r'\b(?:calculate|compute|find|get|show|what(?:\'s|\s+is)?\s+the)\s+(?:the\s+)?average'
        r'(?:\s+(?:of|for|in)\s+(?:column\s+)?([A-Za-z]))?',
        lower,
    )
    if avg_match:
        result = {'type': 'calculate_average'}
        col = avg_match.group(1)
        if col:
            result['column'] = col.upper()
        return result

    # --- close_excel_app: "close excel" / "exit excel" / "quit excel"
    if re.search(r'\b(?:close|exit|quit|stop)\s+(?:microsoft\s+)?excel\b', lower):
        return {'type': 'close_excel_app'}

    # --- open_excel_app: "open excel" / "launch excel" / "start excel"
    #     These phrases refer to launching the Excel application itself.
    if re.search(r'\b(?:open|launch|start)\s+(?:microsoft\s+)?excel\b', lower):
        return {'type': 'open_excel_app'}

    # --- open_file: "open file" / "open the spreadsheet" / "open workbook"
    #     These phrases refer to opening a specific workbook file.
    if re.search(r'\bopen\s+(?:the\s+)?(?:file|spreadsheet|workbook|sheet)\b', lower):
        # Extract the full tail after "open [the] file/spreadsheet/workbook/sheet" so
        # multi-word filenames ("file server", "client list") are captured correctly.
        # Old approach: only matched \S+\.xlsx, which missed "file server.xlsx" (space)
        # and bare names without extensions ("client", "filterr").
        _tail_m = re.search(
            r'\bopen\s+(?:the\s+)?(?:file|spreadsheet|workbook|sheet)\s+(.+)',
            text, re.IGNORECASE,
        )
        filename: str | None = None
        if _tail_m:
            tail = _tail_m.group(1).strip()
            tail = re.sub(r'[.,!?;]+$', '', tail).strip()  # strip trailing STT punctuation
            if tail:
                if not re.search(r'\.(?:xlsx|xls|csv|xlsm|xlsb)$', tail, re.IGNORECASE):
                    tail += '.xlsx'
                filename = tail
        return {'type': 'open_file', **({'filename': filename} if filename else {})}

    # --- create_chart: "create chart" / "make a bar chart" / "create pie chart" ---
    chart_match = re.search(
        r'\b(?:create|make|build|generate|draw|add)\s+(?:a\s+)?'
        r'(?:(bar|line|pie|scatter)\s+)?chart',
        lower,
    )
    if chart_match:
        # Capture an explicit A1-style range if present, e.g. "create bar chart A1:B6"
        # or "create chart from A1:B5". Search original text (preserves letter case).
        _range_m = re.search(r'\b([A-Za-z]{1,3}\d{1,7}):([A-Za-z]{1,3}\d{1,7})\b', t)
        _data_range = (_range_m.group(0).upper() if _range_m else None)
        return {'type': 'create_chart', 'chart_type': (chart_match.group(1) or 'bar').lower(),
                'data_range': _data_range}

    # --- format_cells: "make header bold" ---
    if re.search(r'\bmake\s+(?:the\s+)?header(?:s)?\s+bold\b', lower):
        return {'type': 'format_cells', 'range': '1:1', 'bold': True}

    # --- format_cells: "bold row 11" / "italic row 4" / "format row 3 bold" ---
    fmt_row_style = re.search(
        r'\b(?:bold|italic)\s+row\s+(\d+)\b'
        r'|\b(?:format\s+)?row\s+(\d+)\s+(?:bold|italic)\b',
        t, re.IGNORECASE,
    )
    if fmt_row_style:
        rn = (fmt_row_style.group(1) or fmt_row_style.group(2))
        if re.search(r'\bitalic\b', t, re.IGNORECASE):
            return {'type': 'format_cells', 'range': f'{rn}:{rn}', 'italic': True}
        return {'type': 'format_cells', 'range': f'{rn}:{rn}', 'bold': True}

    # --- format_cells: "make column A bold" / "bold column B" ---
    fmt_col_bold = re.search(
        r'\b(?:(?:make|mark)\s+)?column\s+([A-Za-z]+)\s+bold\b'
        r'|\bbold\s+column\s+([A-Za-z]+)\b',
        t, re.IGNORECASE,
    )
    if fmt_col_bold:
        col = (fmt_col_bold.group(1) or fmt_col_bold.group(2)).upper()
        return {'type': 'format_cells', 'range': f'{col}:{col}', 'bold': True}

    # --- format_cells: "italic column C" / "format column E italic" ---
    fmt_col_italic = re.search(
        r'\b(?:(?:make|mark|format)\s+)?column\s+([A-Za-z]+)\s+italic\b'
        r'|\bitalic\s+column\s+([A-Za-z]+)\b',
        t, re.IGNORECASE,
    )
    if fmt_col_italic:
        col = (fmt_col_italic.group(1) or fmt_col_italic.group(2)).upper()
        return {'type': 'format_cells', 'range': f'{col}:{col}', 'italic': True}

    # --- format_cells: "bold A1:C1" / "make A1 bold" ---
    fmt_bold = re.search(
        r'\b(?:(?:make|mark)\s+)?([A-Za-z]+\d+(?::[A-Za-z]+\d+)?)\s+bold\b'
        r'|\bbold\s+([A-Za-z]+\d+(?::[A-Za-z]+\d+)?)\b',
        t, re.IGNORECASE,
    )
    if fmt_bold:
        rng = (fmt_bold.group(1) or fmt_bold.group(2)).upper()
        return {'type': 'format_cells', 'range': rng, 'bold': True}

    # --- format_cells: "italic B2" / "make B2 italic" ---
    fmt_italic = re.search(
        r'\b(?:(?:make|mark)\s+)?([A-Za-z]+\d+(?::[A-Za-z]+\d+)?)\s+italic\b'
        r'|\bitalic\s+([A-Za-z]+\d+(?::[A-Za-z]+\d+)?)\b',
        t, re.IGNORECASE,
    )
    if fmt_italic:
        rng = (fmt_italic.group(1) or fmt_italic.group(2)).upper()
        return {'type': 'format_cells', 'range': rng, 'italic': True}

    # --- format_cells: "make A3 orange" / "make B2:C3 red" → font_color ---
    # "make [cell] [color]" = FONT COLOR (not fill). "fill"/"highlight" = background fill.
    # Must come after bold/italic checks so "make A3 bold" is already handled above.
    _FILL_COLOR_WORDS = frozenset({
        'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink',
        'black', 'white', 'gray', 'grey', 'cyan', 'magenta', 'brown',
        'lime', 'teal', 'navy', 'maroon', 'olive', 'coral', 'salmon',
    })
    fmt_make_color = re.search(
        r'\bmake\s+([A-Za-z]+\d+(?::[A-Za-z]+\d+)?)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b',
        t, re.IGNORECASE,
    )
    if fmt_make_color:
        _color_candidate = fmt_make_color.group(2).lower().strip()
        if _color_candidate in _FILL_COLOR_WORDS:
            return {
                'type': 'format_cells',
                'range': fmt_make_color.group(1).upper(),
                'font_color': _color_candidate,
            }

    # --- format_cells: "make row 3 blue" → font_color, range=3:3 ---
    fmt_make_row_color = re.search(
        r'\bmake\s+row\s+(\d+)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b',
        t, re.IGNORECASE,
    )
    if fmt_make_row_color:
        _rc = fmt_make_row_color.group(2).lower().strip()
        if _rc in _FILL_COLOR_WORDS:
            rn = fmt_make_row_color.group(1)
            return {'type': 'format_cells', 'range': f'{rn}:{rn}', 'font_color': _rc}

    # --- format_cells: "make column C orange" → font_color, range=C:C ---
    fmt_make_col_color = re.search(
        r'\bmake\s+column\s+([A-Za-z]+)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b',
        t, re.IGNORECASE,
    )
    if fmt_make_col_color:
        _cc = fmt_make_col_color.group(2).lower().strip()
        if _cc in _FILL_COLOR_WORDS:
            col = fmt_make_col_color.group(1).upper()
            return {'type': 'format_cells', 'range': f'{col}:{col}', 'font_color': _cc}

    # --- format_cells: "color row 1 blue" / "color column B green" → font_color ---
    # "color" keyword = font color; "fill/highlight" (below) = background fill color

    # "color row A1 blue" — cell ref instead of row number → treat as cell coloring
    fmt_fc_row_cellref = re.search(
        r'\bcolou?r\s+row\s+([A-Za-z]+\d+(?::[A-Za-z]+\d+)?)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b',
        t, re.IGNORECASE,
    )
    if fmt_fc_row_cellref:
        return {
            'type': 'format_cells',
            'range': fmt_fc_row_cellref.group(1).upper(),
            'font_color': fmt_fc_row_cellref.group(2).lower(),
        }

    fmt_fc_row = re.search(
        r'\bcolor\s+row\s+(\d+)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b',
        t, re.IGNORECASE,
    )
    if fmt_fc_row:
        color = fmt_fc_row.group(2).lower()
        rn = fmt_fc_row.group(1)
        return {'type': 'format_cells', 'range': f'{rn}:{rn}', 'font_color': color}

    fmt_fc_col = re.search(
        r'\bcolor\s+column\s+([A-Za-z]+)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b',
        t, re.IGNORECASE,
    )
    if fmt_fc_col:
        col = fmt_fc_col.group(1).upper()
        color = fmt_fc_col.group(2).lower()
        return {'type': 'format_cells', 'range': f'{col}:{col}', 'font_color': color}

    fmt_fc_cell = re.search(
        r'\bcolor\s+([A-Za-z]+\d+(?::[A-Za-z]+\d+)?)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\b',
        t, re.IGNORECASE,
    )
    if fmt_fc_cell:
        rng = fmt_fc_cell.group(1).upper()
        color = fmt_fc_cell.group(2).lower()
        return {'type': 'format_cells', 'range': rng, 'font_color': color}

    # --- format_cells: "fill row 2 with red" / "highlight row 1" → fill_color ---
    fmt_row_color = re.search(
        r'\b(?:fill|highlight)\s+row\s+(\d+)\s+(?:with\s+)?([A-Za-z]+(?:\s+[A-Za-z]+)?)\b',
        t, re.IGNORECASE,
    )
    if fmt_row_color:
        row_num = fmt_row_color.group(1)
        color = fmt_row_color.group(2).lower()
        return {'type': 'format_cells', 'range': f'{row_num}:{row_num}', 'fill_color': color}

    # "highlight row 1" with no color → default yellow
    fmt_hl_row = re.search(r'\bhighlight\s+row\s+(\d+)\b', t, re.IGNORECASE)
    if fmt_hl_row:
        return {'type': 'format_cells', 'range': f'{fmt_hl_row.group(1)}:{fmt_hl_row.group(1)}',
                'fill_color': 'yellow'}

    # --- format_cells: "fill column B light blue" → fill_color + column range ---
    fmt_fill_col = re.search(
        r'\b(?:fill|highlight)\s+column\s+([A-Za-z]+)\s+(?:with\s+)?([A-Za-z]+(?:\s+[A-Za-z]+)?)\b',
        t, re.IGNORECASE,
    )
    if fmt_fill_col:
        col = fmt_fill_col.group(1).upper()
        color = fmt_fill_col.group(2).lower()
        return {'type': 'format_cells', 'range': f'{col}:{col}', 'fill_color': color}

    # --- format_cells: "fill A1:C1 with blue" / "highlight B2 yellow" → fill_color ---
    fmt_cell_color = re.search(
        r'\b(?:fill|highlight)\s+([A-Za-z]+\d+(?::[A-Za-z]+\d+)?)\s+(?:with\s+)?([A-Za-z]+(?:\s+[A-Za-z]+)?)\b',
        t, re.IGNORECASE,
    )
    if fmt_cell_color:
        rng = fmt_cell_color.group(1).upper()
        color = fmt_cell_color.group(2).lower()
        return {'type': 'format_cells', 'range': rng, 'fill_color': color}

    # --- format_cells: "highlight A1:C1" / "highlight B2" with no color → fill_color=yellow ---
    # Must come after fmt_cell_color so "highlight B2 yellow" (with color) is already handled.
    fmt_hl_cell = re.search(
        r'\bhighlight\s+([A-Za-z]+\d+(?::[A-Za-z]+\d+)?)\b',
        t, re.IGNORECASE,
    )
    if fmt_hl_cell:
        return {'type': 'format_cells', 'range': fmt_hl_cell.group(1).upper(), 'fill_color': 'yellow'}

    # --- format_cells: "format column B as number" / "format row 3 as currency" / "format B2:B10 as currency" ---
    fmt_num = re.search(
        r'\bformat\s+(?:row\s+(\d+)|column\s+([A-Za-z]+)|([A-Za-z]+\d+(?::[A-Za-z]+\d+)?))\s+as\s+'
        r'(number|integer|currency|percentage|percent|decimal|date|text)\b',
        t, re.IGNORECASE,
    )
    if fmt_num:
        if fmt_num.group(1):
            rn = fmt_num.group(1)
            rng = f'{rn}:{rn}'
        elif fmt_num.group(2):
            col = fmt_num.group(2).upper()
            rng = f'{col}:{col}'
        else:
            rng = fmt_num.group(3).upper()
        return {'type': 'format_cells', 'range': rng, 'number_format': fmt_num.group(4).lower()}

    # --- format_cells: "bold row 1" ---
    fmt_bold_row = re.search(r'\bbold\s+row\s+(\d+)\b', t, re.IGNORECASE)
    if fmt_bold_row:
        rn = fmt_bold_row.group(1)
        return {'type': 'format_cells', 'range': f'{rn}:{rn}', 'bold': True}

    # --- format_cells: font size ---
    # "set font size A1 16" / "set font size B2:C3 14" / "font size A2 20"
    # Optional "set" prefix so "font size A2 20" (no set) is also caught.
    fmt_set_size = re.search(
        r'\b(?:set\s+)?(?:font\s+)?size\s+([A-Za-z]+\d+(?::[A-Za-z]+\d+)?)\s+(\d+)\b',
        t, re.IGNORECASE,
    )
    if fmt_set_size:
        return {'type': 'format_cells', 'range': fmt_set_size.group(1).upper(),
                'font_size': int(fmt_set_size.group(2))}

    # "set font size row 3 16" / "set font size column B 14" / "font size row 3 16"
    # Also handles "column B14" where cell-ref normalization merged "B 14" → "B14".
    # Groups: (row_num, row_size) | (col_letter, col_size) | (col_letter_merged, col_size_merged)
    fmt_set_size_row_col = re.search(
        r'\b(?:set\s+)?(?:font\s+)?size\s+'
        r'(?:row\s+(\d+)\s+(\d+)|column\s+([A-Za-z]+)\s+(\d+)|column\s+([A-Za-z])(\d+))\b',
        t, re.IGNORECASE,
    )
    if fmt_set_size_row_col:
        if fmt_set_size_row_col.group(1):
            rn = fmt_set_size_row_col.group(1)
            rng = f'{rn}:{rn}'
            sz = int(fmt_set_size_row_col.group(2))
        elif fmt_set_size_row_col.group(3):
            col = fmt_set_size_row_col.group(3).upper()
            rng = f'{col}:{col}'
            sz = int(fmt_set_size_row_col.group(4))
        else:
            col = fmt_set_size_row_col.group(5).upper()
            rng = f'{col}:{col}'
            sz = int(fmt_set_size_row_col.group(6))
        return {'type': 'format_cells', 'range': rng, 'font_size': sz}

    # "make header size 18" / "make the header font size 20"
    fmt_header_size = re.search(
        r'\bmake\s+(?:the\s+)?header\s+(?:font\s+)?size\s+(\d+)\b', t, re.IGNORECASE
    )
    if fmt_header_size:
        return {'type': 'format_cells', 'range': '1:1', 'font_size': int(fmt_header_size.group(1))}

    # "increase/decrease font size A1" / "increase font size row 1" / "increase font size column B"
    fmt_inc_size = re.search(
        r'\b(increase|decrease)\s+(?:font\s+)?size\s+'
        r'(?:row\s+(\d+)|column\s+([A-Za-z]+)|([A-Za-z]+\d+(?::[A-Za-z]+\d+)?))\b',
        t, re.IGNORECASE,
    )
    if fmt_inc_size:
        delta = 2 if fmt_inc_size.group(1).lower() == 'increase' else -2
        if fmt_inc_size.group(2):
            rn = fmt_inc_size.group(2)
            rng = f'{rn}:{rn}'
        elif fmt_inc_size.group(3):
            col = fmt_inc_size.group(3).upper()
            rng = f'{col}:{col}'
        else:
            rng = fmt_inc_size.group(4).upper()
        return {'type': 'format_cells', 'range': rng, 'font_size_change': delta}

    # --- format_cells: alignment ---
    # "center A1:C1" / "center row 1"
    fmt_center = re.search(
        r'\bcenter\s+(?:row\s+(\d+)|column\s+([A-Za-z]+)|([A-Za-z]+\d+(?::[A-Za-z]+\d+)?))\b',
        t, re.IGNORECASE,
    )
    if fmt_center:
        if fmt_center.group(1):
            rn = fmt_center.group(1)
            rng = f'{rn}:{rn}'
        elif fmt_center.group(2):
            col = fmt_center.group(2).upper()
            rng = f'{col}:{col}'
        else:
            rng = fmt_center.group(3).upper()
        return {'type': 'format_cells', 'range': rng, 'align': 'center'}

    # "align column B right" / "align A2:B5 left" / "align row 1 center"
    fmt_align = re.search(
        r'\balign\s+(?:row\s+(\d+)|column\s+([A-Za-z]+)|([A-Za-z]+\d+(?::[A-Za-z]+\d+)?))\s+'
        r'(left|right|center|centre)\b',
        t, re.IGNORECASE,
    )
    if fmt_align:
        direction = fmt_align.group(4).lower().replace('centre', 'center')
        if fmt_align.group(1):
            rn = fmt_align.group(1)
            rng = f'{rn}:{rn}'
        elif fmt_align.group(2):
            col = fmt_align.group(2).upper()
            rng = f'{col}:{col}'
        else:
            rng = fmt_align.group(3).upper()
        return {'type': 'format_cells', 'range': rng, 'align': direction}

    # --- unmerge_cells: "unmerge A1:C1" / "unmerge cells A1:C3" ---
    fmt_unmerge_range = re.search(
        r'\bunmerge\s+(?:cells?\s+)?([A-Za-z]+\d+)\s*(?::|to|\s)\s*([A-Za-z]+\d+)\b',
        t, re.IGNORECASE,
    )
    if fmt_unmerge_range:
        start = fmt_unmerge_range.group(1).upper()
        end = fmt_unmerge_range.group(2).upper()
        return {'type': 'unmerge_cells', 'range': f'{start}:{end}'}

    # --- merge_cells: "merge A1:C1" / "merge header" / "merge cells A1 B1" ---
    fmt_merge_header = re.search(r'\bmerge\s+(?:the\s+)?header\b', lower)
    if fmt_merge_header:
        return {'type': 'merge_cells', 'range': 'A1:C1'}

    fmt_merge_range = re.search(
        r'\bmerge\s+(?:cells?\s+)?([A-Za-z]+\d+)\s*(?::|to|\s)\s*([A-Za-z]+\d+)\b',
        t, re.IGNORECASE,
    )
    if fmt_merge_range:
        start = fmt_merge_range.group(1).upper()
        end = fmt_merge_range.group(2).upper()
        return {'type': 'merge_cells', 'range': f'{start}:{end}'}

    # --- set_column_width: "auto fit column A" / "auto fit all columns" ---
    fmt_autofit_all = re.search(r'\bauto\s*fit\s+all\s+columns\b', lower)
    if fmt_autofit_all:
        return {'type': 'set_column_width', 'column': 'all', 'auto': True}

    fmt_autofit_col = re.search(r'\bauto\s*fit\s+column\s+([A-Za-z]+)\b', t, re.IGNORECASE)
    if fmt_autofit_col:
        return {'type': 'set_column_width', 'column': fmt_autofit_col.group(1).upper(), 'auto': True}

    # "adjust column width A 20" / "set column width B 15"
    # Also handles the merged form "A20" produced by the cell-ref normalizer
    # when "A 20" (column letter + width) is adjacent: \s* accepts zero spaces.
    fmt_col_width = re.search(
        r'\b(?:adjust|set)\s+column\s+(?:width\s+)?'
        r'(?!(?:with|be|to|as|at|the|and|or|in|of)\b)([A-Za-z]+)\s*(\d+(?:\.\d+)?)\b',
        t, re.IGNORECASE,
    )
    if fmt_col_width:
        col = fmt_col_width.group(1).upper()
        width = float(fmt_col_width.group(2))
        return {'type': 'set_column_width', 'column': col, 'width': width}

    # --- sort_range ---
    # Supports: "sort A2:C10 by column B", "sort A2 by B descending",
    #           "sort the table by score", "sort by B ascending"
    sort_match = re.search(
        r'\bsort(?:\s+(?:the\s+)?(?:table|range|data|sheet))?'
        r'(?:\s+[A-Za-z]+\d+(?::[A-Za-z]+\d+)?)?'
        r'\s+by\s+(?:column\s+)?([A-Za-z]+|\d+)'
        r'(?:\s+(ascending|descending))?',
        t, re.IGNORECASE,
    )
    if sort_match:
        order = sort_match.group(2) or 'ascending'
        ascending = not order.lower().startswith('desc')
        return {'type': 'sort_range', 'column': sort_match.group(1).upper(),
                'ascending': ascending}

    # --- find_replace: "replace Tom with John" / "replace 90 with 95 in column B" ---
    replace_match = re.search(
        r'\breplace\s+(.+?)\s+with\s+(.+?)(?:\s+in\s+(?:column\s+)?([A-Za-z]+\d*(?::[A-Za-z]+\d+)?))?\s*$',
        t, re.IGNORECASE,
    )
    if replace_match:
        result: dict = {'type': 'find_replace',
                        'find': replace_match.group(1).strip(),
                        'replace': replace_match.group(2).strip()}
        if replace_match.group(3):
            result['range'] = replace_match.group(3).upper()
        return result

    # --- clear_filter: "clear filter" / "show all" / "reset filter" ---
    if re.search(r'\b(?:clear|remove|reset)\s+filter\b|\bshow\s+all\b', lower):
        return {'type': 'clear_filter'}

    # --- filter_table: "filter column B = 90" / "filter Score = 90" / "filter Name = John" ---
    filter_match = re.search(
        r'\bfilter\s+(?:column\s+)?([A-Za-z]\w*)\s*(=|==|!=|<>|>=|<=|>|<)\s*(.+?)\s*$',
        t, re.IGNORECASE,
    )
    if filter_match:
        col = filter_match.group(1).strip()
        op  = filter_match.group(2).strip().replace('==', '=').replace('<>', '!=')
        val = filter_match.group(3).strip()
        return {'type': 'filter_table', 'column': col, 'operator': op, 'value': val}

    # Voice forms without explicit operator symbol:
    #   "filter column B 90"        → column=B, op=, value=90
    #   "filter B 90"               → column=B, op=, value=90
    #   "filter column B equals 90" → column=B, op=, value=90
    #   "filter column B is 90"     → column=B, op=, value=90
    #   "Filter Column B 90"        → same (after comma-strip + case normalise)
    _FILTER_KW = {'by', 'where', 'the', 'equals', 'equal', 'is', 'are', 'column'}
    voice_filter = re.search(
        r'\bfilter\s+(?:column\s+)?([A-Za-z]\w*)\s+'
        r'(?:(?:equals?|is)\s+)?(.+?)\s*$',
        t, re.IGNORECASE,
    )
    if voice_filter:
        col = voice_filter.group(1).strip().lower()
        val = voice_filter.group(2).strip().rstrip('.,!?…')
        if col not in _FILTER_KW and val:
            return {'type': 'filter_table', 'column': col.upper(),
                    'operator': '=', 'value': val}

    # --- unfreeze_panes: "unfreeze panes" / "unfreeze" ---
    if re.search(r'\bunfreeze(?:\s+panes?)?\b', lower):
        return {'type': 'unfreeze_panes'}

    # --- freeze_panes: "freeze header" / "freeze row 1" / "freeze panes" / "freeze A2" ---
    if re.search(r'\bfreeze\s+(?:the\s+)?(?:header(?:s)?|row\s+1)\b', lower):
        return {'type': 'freeze_panes', 'cell': 'A2'}
    if re.search(r'\bfreeze\s+(?:first\s+col(?:umn)?|col(?:umn)?\s+[A-Za-z])\b', lower):
        return {'type': 'freeze_panes', 'cell': 'B1'}
    if re.search(r'\bfreeze\s+panes?\b', lower):
        return {'type': 'freeze_panes', 'cell': 'B1'}
    freeze_cell = re.search(r'\bfreeze\s+([A-Za-z]+\d+)\b', t, re.IGNORECASE)
    if freeze_cell:
        return {'type': 'freeze_panes', 'cell': freeze_cell.group(1).upper()}

    # --- save ---
    if re.search(r'\bsave\b', lower):
        return {'type': 'save'}

    headers, domain = _infer_domain_headers(lower)

    # Use explicit column names over inferred domain headers
    if explicit_headers:
        headers = explicit_headers

    # --- Filename extraction (applies to all create branches below) ---
    # Priority 1: explicit "called/named X.xlsx"
    # Priority 2: bare "X.xlsx" extension anywhere in the command (e.g. "create excel demo.xlsx")
    # Priority 3: bare word after type keyword: "create spreadsheet test" → test.xlsx
    # We run this once here so all create branches share the same logic,
    # even when _enrich_with_context prepends a [file:... columns:...] prefix.
    _cmd_text_for_fn = re.sub(r'^\[.*?\]\s*', '', lower)  # strip context prefix once
    _fn_explicit = re.search(r'(?:called|named|titled)\s+["\']?(\S+\.xlsx?)\b', _cmd_text_for_fn)
    _extracted_filename: str | None = _fn_explicit.group(1) if _fn_explicit else None
    if _extracted_filename is None:
        # Match a bare "something.xlsx" / "something.xls" anywhere in the command text
        _fn_bare_ext = re.search(r'\b([A-Za-z0-9][A-Za-z0-9_\-]*\.xlsx?)\b', _cmd_text_for_fn)
        if _fn_bare_ext:
            _extracted_filename = _fn_bare_ext.group(1)
    if _extracted_filename is None and is_create:
        _fn_skip = {
            'new', 'the', 'a', 'an', 'my', 'me', 'with', 'and', 'for',
            'name', 'called', 'named', 'titled', 'file', 'sheet',
            'excel', 'workbook', 'spreadsheet',
        }
        # Priority 3a: "create excel name animal" / "create excel called foo"
        # Handle marker word ("name", "called", "named", "titled") before actual name.
        _named = re.search(
            r'\b(?:spreadsheet|workbook|excel\s+file|excel|sheet|file)\s+'
            r'(?:name|called|named|titled)\s+'
            r'([A-Za-z0-9][A-Za-z0-9_\-]*)\s*$',
            _cmd_text_for_fn,
        )
        if _named:
            _name = _named.group(1)
            if _name not in _fn_skip:
                _extracted_filename = _name + '.xlsx'

        # Priority 3b: "create spreadsheet WORD" / "make excel WORD" — bare word at end
        if _extracted_filename is None:
            _bare = re.search(
                r'\b(?:spreadsheet|workbook|excel\s+file|excel|sheet|file)\s+'
                r'([A-Za-z0-9][A-Za-z0-9_\-]*)\s*$',
                _cmd_text_for_fn,
            )
            if _bare:
                _name = _bare.group(1)
                if _name not in _fn_skip:
                    _extracted_filename = _name + '.xlsx'

    # Create-only (no data to add)
    if is_create and not is_add and headers:
        result = {'type': 'create_excel', 'headers': headers}
        if _extracted_filename:
            result['filename'] = _extracted_filename
        return result

    # Create with explicit columns but no domain match
    if is_create and not is_add and explicit_headers:
        result = {'type': 'create_excel', 'headers': explicit_headers}
        if _extracted_filename:
            result['filename'] = _extracted_filename
        return result

    # Generic create: "create excel file" / "create a new spreadsheet" with no domain
    # New files start completely empty — no preset headers.
    if is_create and not is_add and not headers:
        if any(w in lower for w in ('excel', 'spreadsheet', 'sheet', 'file', 'workbook')):
            result = {'type': 'create_excel', 'headers': []}
            if _extracted_filename:
                result['filename'] = _extracted_filename
            return result

    # Create + add
    if is_create and is_add and headers:
        rows = []

        if domain == 'attendance':
            # Match patterns like "Alice present", "Bob absent"
            matches = re.findall(
                r'([A-Za-z][A-Za-z\-]+)\s+(present|absent|yes|no|here)',
                t, re.IGNORECASE
            )
            for name, status in matches:
                if name.lower() in _ATTENDANCE_STOP_WORDS:
                    continue
                mapped = 'Present' if status.lower() in ('present', 'yes', 'here') else 'Absent'
                rows.append({'type': 'add_row', 'values': [name.capitalize(), '', mapped]})

        elif domain in ('inventory', 'expense'):
            stop = _ITEM_STOP_WORDS
            # Match "keyword number" pairs
            matches = re.findall(
                r'([A-Za-z][A-Za-z\-]*)\s+(\d+(?:\.\d+)?)', t
            )
            for item, qty in matches:
                if item.lower() in stop:
                    continue
                val = float(qty) if '.' in qty else int(qty)
                rows.append({'type': 'add_row', 'values': [item.capitalize(), val]})

        elif domain == 'budget':
            matches = re.findall(r'([A-Za-z][A-Za-z\-]*)\s+(\d+(?:\.\d+)?)', t)
            for cat, amt in matches:
                if cat.lower() in _BUDGET_STOP_WORDS:
                    continue
                val = float(amt) if '.' in amt else int(amt)
                rows.append({'type': 'add_row', 'values': [cat.capitalize(), val, '']})

        elif domain == 'project':
            status_m = re.search(
                r'\b(pending|done|completed?|in[\s\-]progress|wip|blocked|todo)\b',
                t, re.IGNORECASE
            )
            status = status_m.group(0).capitalize() if status_m else 'Pending'
            add_m = re.search(r'\b(?:add|insert|put|record)\b\s+', t, re.IGNORECASE)
            if add_m:
                start_pos = add_m.end()
                end_pos = (
                    status_m.start()
                    if status_m and status_m.start() > start_pos
                    else len(t)
                )
                task_name = re.sub(r'\s+', ' ', t[start_pos:end_pos]).strip(' ,.')
                if task_name and task_name.lower() not in _PROJECT_STOP_WORDS:
                    rows.append({'type': 'add_row', 'values': [task_name.title(), '', status, '']})

        elif domain == 'meeting':
            add_m = re.search(r'\b(?:add|insert|put|record)\b\s+(.+)', t, re.IGNORECASE)
            if add_m:
                topic = re.sub(r'\s+', ' ', add_m.group(1)).strip(' ,.')
                if topic and topic.lower() not in _MEETING_STOP_WORDS:
                    rows.append({'type': 'add_row', 'values': [topic.title(), '', '']})

        else:
            # Default: name + numeric score
            matches = re.findall(
                r'([A-Za-z\u00C0-\u024F\u4e00-\u9fff]+)\s*(?:with\s+|score\s+)?(\d+)', t
            )
            for name, score in matches:
                if name.lower() in _SCORE_STOP_WORDS:
                    continue
                rows.append({
                    'type': 'add_row',
                    'values': [name.capitalize() if name.isascii() else name, int(score)]
                })

        if rows:
            return {'steps': [{'type': 'create_excel', 'headers': headers}] + rows}
        # create without extractable rows — still return a create step
        return {'type': 'create_excel', 'headers': headers}

    # --- insert_row_at: "insert row 3" / "insert row before 3" /
    #     "insert before row 3" / "insert row at 3".
    #     Fires ONLY when argument is a bare integer (no data values follow).
    #     "insert row Tom 75" has non-digit content → falls through to add_row below.
    ins_row_match = re.search(
        r'\binsert\s+(?:a\s+)?(?:before\s+)?row\s+(?:(?:at|before|after)\s+)?(\d+)\s*$',
        t, re.IGNORECASE,
    )
    if ins_row_match and not is_create:
        idx = int(ins_row_match.group(1))
        if idx >= 2:
            return {'type': 'insert_row_at', 'index': idx}
        # idx == 1 → fall through, let controller reject it with clear message;
        # but returning None here lets the LLM handle it instead — so we return
        # the action and let the controller produce the friendly rejection.
        return {'type': 'insert_row_at', 'index': idx}

    # --- Standalone add_row: "add row Tom 90" / "insert Tom 90" ---
    if is_add and not is_create:
        add_row_match = re.search(
            r'\b(?:add|insert)\s+(?:a\s+)?(?:row\s+)?(.+)',
            t, re.IGNORECASE,
        )
        if add_row_match:
            payload = add_row_match.group(1).strip()
            # Try to split into tokens and convert numbers
            tokens = re.split(r'[,\s]+', payload)
            # 'at' and 'ad' are STT duplication artifacts (e.g. "add row John 90 at row John"
            # → payload "John 90 at row John" → 'at' must not become a data cell value).
            # 'column'/'col' guard: "add column" with no name falls through here and
            # would otherwise emit add_row with values=['column'], which is wrong.
            tokens = [tk for tk in tokens if tk and tk.lower() not in
                      ('row', 'data', 'with', 'at', 'ad', 'least', 'draw',
                       'uh', 'um', 'hmm', 'hm', 'column', 'col')]
            # If only a bare number survived, the name slot is missing.
            if len(tokens) == 1 and re.match(r'^\d+$', tokens[0]):
                return None  # let check_command_slots request clarification
            if tokens:
                values = []
                for tk in tokens:
                    try:
                        values.append(int(tk))
                    except ValueError:
                        try:
                            values.append(float(tk))
                        except ValueError:
                            values.append(tk)
                return {'type': 'add_row', 'values': values}

    return None


def generate_suggestions(text: str, n=5):
    """Return simple suggestions; attempt Groq if available, else heuristics."""
    # Simple heuristics as fallback
    if not text:
        return ["create excel", "add row", "add column score"]
    t = text.lower()
    out = []
    if any(x in t for x in ('create', 'new', 'make')):
        out.append('create excel')
    if any(x in t for x in ('add row', 'insert row', 'add a row', 'add rows')):
        out.append('add row')
    m = re.search(r'delete .*row\s*(\d+)', t)
    if m:
        out.append(f'delete row {m.group(1)}')
    if any(x in t for x in ('column', 'add column', 'insert column')):
        mm = re.search(r'column\s+(\w+)', t)
        if mm:
            out.append(f'add column {mm.group(1)}')
        else:
            out.append('add column score')
    if 'sheet' in t:
        out.append('create sheet Sheet1')
    defaults = ['create excel', 'add row', 'delete row 2', 'add column score']
    for d in defaults:
        if d not in out:
            out.append(d)
        if len(out) >= n:
            break
    return out[:n]


# ---------------------------------------------------------------------------
# Pydantic command schema validation (Improvement 2)
# ---------------------------------------------------------------------------
# Provides a formal typed schema on top of the manual _validate_single_action
# checks.  Benefits:
#   • Automatic type coercion (e.g. "2" -> 2 for integer fields)
#   • Self-documenting field constraints for thesis reviewers
#   • Clean ValidationError messages that map to safe speech responses
#
# Graceful fallback: if pydantic is not installed the module still works —
# validate_command() simply returns the action dict unchanged.

def validate_command(action: dict) -> dict:
    """Run the parsed action dict through the Pydantic schema.

    Returns the (possibly coerced) action dict on success.
    Returns ``{"type": "unknown", "raw": "validation_error"}`` if the action
    fails schema validation so the pipeline can fall back gracefully.

    If pydantic is not installed this function is a transparent no-op.
    """
    try:
        from pydantic import BaseModel, field_validator, model_validator, ValidationError
        from typing import Any, List, Literal, Optional, Union
    except ImportError:
        # pydantic not available — pass through unchanged
        return action

    if not isinstance(action, dict):
        log.warning('[PARSE] validate_command: received non-dict action: %r', type(action))
        return {"type": "unknown", "raw": "validation_error"}

    atype = action.get('type')

    # Multi-step plans bypass per-action schema (steps validated individually)
    if 'steps' in action:
        validated_steps = []
        for step in action.get('steps', []):
            result = validate_command(step)
            if result.get('type') != 'unknown':
                validated_steps.append(result)
        if not validated_steps:
            log.warning('[PARSE] validate_command: all steps failed schema validation')
            return {"type": "unknown", "raw": "validation_error"}
        return {"steps": validated_steps}

    # ---- Per-type Pydantic models ----------------------------------------

    try:
        if atype == 'add_row':
            class AddRowSchema(BaseModel):
                type: Literal['add_row']
                values: List[Any]
            AddRowSchema(**{k: v for k, v in action.items()})

        elif atype == 'delete_row':
            class DeleteRowSchema(BaseModel):
                type: Literal['delete_row']
                index: int
                # 'row' is accepted as a synonym for 'index' (LLM sometimes
                # returns row=N instead of index=N).  model_validator promotes
                # it to 'index' before field validation runs.
                row: Optional[int] = None

                @model_validator(mode='before')
                @classmethod
                def _normalise_row_key(cls, data):
                    if isinstance(data, dict) and 'index' not in data and 'row' in data:
                        data = dict(data, index=data['row'])
                    return data

                @field_validator('index')
                @classmethod
                def index_must_be_gte_2(cls, v):
                    if v < 2:
                        raise ValueError('index must be >= 2 to protect header row')
                    return v

            validated = DeleteRowSchema(**{k: v for k, v in action.items()})
            action = dict(action, index=validated.index)  # apply coercion

        elif atype == 'write_cell':
            class WriteCellSchema(BaseModel):
                type: Literal['write_cell']
                cell: str
                value: Any
            WriteCellSchema(**{k: v for k, v in action.items()})

        elif atype == 'write_formula':
            class WriteFormulaSchema(BaseModel):
                type: Literal['write_formula']
                cell: str
                formula: str
            WriteFormulaSchema(**{k: v for k, v in action.items()})

        elif atype == 'create_excel':
            class CreateExcelSchema(BaseModel):
                type: Literal['create_excel']
                headers: List[str]
            CreateExcelSchema(**{k: v for k, v in action.items()})

        elif atype == 'calculate_row_sum':
            class CalcRowSumSchema(BaseModel):
                type: Literal['calculate_row_sum']
                index: int
            validated = CalcRowSumSchema(**{k: v for k, v in action.items()})
            action = dict(action, index=validated.index)

        elif atype == 'write_sum_formula':
            class WriteSumFormulaSchema(BaseModel):
                type: Literal['write_sum_formula']
                range_ref: str
                cell: str
            WriteSumFormulaSchema(**{k: v for k, v in action.items()})

        elif atype == 'write_average_formula':
            class WriteAverageFormulaSchema(BaseModel):
                type: Literal['write_average_formula']
                range_ref: str
                cell: str
            WriteAverageFormulaSchema(**{k: v for k, v in action.items()})

        elif atype == 'format_cells':
            class FormatCellsSchema(BaseModel):
                type: Literal['format_cells']
                range: str
            FormatCellsSchema(**{k: v for k, v in action.items()})

        # Types with no required extra fields beyond 'type' pass through

        log.debug('[PARSE] validate_command: schema OK for action type=%r', atype)
        return action

    except ValidationError as exc:
        log.warning(
            '[PARSE] validate_command: Pydantic validation failed for type=%r: %s',
            atype, exc.errors(),
        )
        return {"type": "unknown", "raw": "validation_error"}
    except Exception as exc:
        log.warning('[PARSE] validate_command: unexpected error for type=%r: %s', atype, exc)
        return action  # fail open on unexpected errors
