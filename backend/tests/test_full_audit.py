"""
Sara Voice-Excel Assistant — Full Command Audit
================================================
QA-ONLY script.  Does NOT modify any project source files.
Uses a temporary file (test_audit.xlsx) isolated in a temp directory.
Runs process_command() for every registered command type.

Run:
    python backend/tests/test_full_audit.py
"""
from __future__ import annotations

import os
import sys
import time
import tempfile
import shutil
import textwrap
from typing import Any

# ── path bootstrap ────────────────────────────────────────────────────────────
_TESTS = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_TESTS)
for _p in (_BACKEND, os.path.dirname(_BACKEND)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# load .env so GROQ_* vars are available (heuristic path doesn't need them,
# but the import chain initialises them at module level)
try:
    from dotenv import load_dotenv
    import pathlib
    _env = pathlib.Path(_BACKEND) / '.env'
    if _env.exists():
        load_dotenv(_env, override=True)
except ImportError:
    pass

from agent import process_command        # noqa: E402  (after path setup)
import openpyxl                          # noqa: E402

# ── helpers ───────────────────────────────────────────────────────────────────

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

_results: list[dict] = []
_audit_file: str = ""        # absolute path to the temp xlsx


def _wb():
    """Open the audit workbook read-only for verification."""
    return openpyxl.load_workbook(_audit_file)


def _cmd(text: str, fname: str | None = None) -> str:
    """Call process_command and return the speech response."""
    target = fname or os.path.basename(_audit_file)
    return process_command(text, filename=target)


def record(category: str, cmd_type: str, natural: str,
           result: str, status: str, detail: str = "") -> None:
    _results.append(dict(
        category=category,
        cmd_type=cmd_type,
        natural=natural,
        response=result[:120] if result else "",
        status=status,
        detail=detail,
    ))
    icon = "✔" if status == PASS else ("⚠" if status == SKIP else "✘")
    print(f"  {icon} [{cmd_type}]  {status}  — {detail or result[:80]}")


def section(title: str) -> None:
    print(f"\n{'═'*64}")
    print(f"  {title}")
    print(f"{'═'*64}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def test_file_operations() -> None:
    section("1. FILE OPERATIONS")

    # ── create_excel ─────────────────────────────────────────────────────────
    r = _cmd("create excel file test_audit.xlsx")
    ok = "creat" in r.lower() or "exist" in r.lower() or "open" in r.lower() or "xlsx" in r.lower()
    record("File", "create_excel", "create excel file test_audit.xlsx",
           r, PASS if ok else FAIL, "file created / already exists" if ok else r)

    time.sleep(0.3)

    # ── open_file ─────────────────────────────────────────────────────────────
    r = _cmd("open file test_audit.xlsx")
    ok = any(w in r.lower() for w in ("open", "ready", "load", "active", "using"))
    record("File", "open_file", "open file test_audit.xlsx",
           r, PASS if ok else FAIL, "")

    # ── save ─────────────────────────────────────────────────────────────────
    r = _cmd("save")
    ok = any(w in r.lower() for w in ("save", "saved", "done", "ok", "success"))
    record("File", "save", "save", r, PASS if ok else FAIL, "")

    # ── rename_file ───────────────────────────────────────────────────────────
    # rename then rename back so the rest of the tests still work
    r = _cmd("rename file to test_audit_renamed.xlsx")
    ok = any(w in r.lower() for w in ("renam", "done", "success", "ok"))
    detail = ""
    if ok:
        # rename back
        r2 = _cmd("rename file to test_audit.xlsx", fname="test_audit_renamed.xlsx")
        ok2 = any(w in r2.lower() for w in ("renam", "done", "success", "ok"))
        detail = "renamed → renamed back" if ok2 else f"rename-back failed: {r2}"
    record("File", "rename_file", "rename file to test_audit_renamed.xlsx",
           r, PASS if ok else FAIL, detail)


def test_row_operations() -> None:
    section("2. ROW OPERATIONS")

    # ── add_row ───────────────────────────────────────────────────────────────
    r = _cmd("add row Alice 85")
    ok = any(w in r.lower() for w in ("add", "row", "success", "done", "ok"))
    record("Row", "add_row", "add row Alice 85", r, PASS if ok else FAIL, "")

    r = _cmd("add row Bob 72")
    ok = any(w in r.lower() for w in ("add", "row", "success", "done", "ok"))
    record("Row", "add_row", "add row Bob 72", r, PASS if ok else FAIL, "2nd row")

    r = _cmd("add row Charlie 90")
    ok = any(w in r.lower() for w in ("add", "row", "success", "done", "ok"))
    record("Row", "add_row", "add row Charlie 90", r, PASS if ok else FAIL, "3rd row")

    # verify via openpyxl
    try:
        wb = _wb()
        ws = wb.active
        names = [ws.cell(row=r, column=1).value for r in range(2, ws.max_row + 1)]
        ok_v = "Alice" in names and "Bob" in names and "Charlie" in names
        record("Row", "add_row[verify]", "openpyxl check",
               str(names), PASS if ok_v else FAIL, "rows present in xlsx" if ok_v else f"got {names}")
        wb.close()
    except Exception as e:
        record("Row", "add_row[verify]", "openpyxl check", "", FAIL, str(e))

    # ── insert_row_at ─────────────────────────────────────────────────────────
    r = _cmd("insert row at position 2")
    ok = any(w in r.lower() for w in ("insert", "row", "done", "ok", "success"))
    record("Row", "insert_row_at", "insert row at position 2", r, PASS if ok else FAIL, "")

    # ── delete_row ────────────────────────────────────────────────────────────
    r = _cmd("delete row 2")
    ok = any(w in r.lower() for w in ("delet", "row", "done", "ok", "success", "remov"))
    record("Row", "delete_row", "delete row 2", r, PASS if ok else FAIL, "")


def test_cell_operations() -> None:
    section("3. CELL OPERATIONS")

    # ── write_cell ────────────────────────────────────────────────────────────
    r = _cmd("write cell A1 Name")
    ok = any(w in r.lower() for w in ("write", "cell", "updat", "done", "ok", "set", "success"))
    record("Cell", "write_cell", "write cell A1 Name", r, PASS if ok else FAIL, "")

    r = _cmd("write cell B1 Score")
    ok = any(w in r.lower() for w in ("write", "cell", "updat", "done", "ok", "set", "success"))
    record("Cell", "write_cell", "write cell B1 Score", r, PASS if ok else FAIL, "")

    r = _cmd("write cell A2 Dave")
    ok = any(w in r.lower() for w in ("write", "cell", "updat", "done", "ok", "success"))
    record("Cell", "write_cell", "write cell A2 Dave", r, PASS if ok else FAIL, "")

    r = _cmd("write cell B2 95")
    ok = any(w in r.lower() for w in ("write", "cell", "updat", "done", "ok", "success"))
    record("Cell", "write_cell", "write cell B2 95", r, PASS if ok else FAIL, "")

    # verify
    try:
        wb = _wb()
        ws = wb.active
        ok_v = ws["A1"].value == "Name" and ws["B1"].value == "Score"
        record("Cell", "write_cell[verify]", "openpyxl check",
               f"A1={ws['A1'].value} B1={ws['B1'].value}",
               PASS if ok_v else FAIL, "")
        wb.close()
    except Exception as e:
        record("Cell", "write_cell[verify]", "openpyxl check", "", FAIL, str(e))

    # ── read_cell ─────────────────────────────────────────────────────────────
    r = _cmd("read cell A1")
    ok = any(w in r.lower() for w in ("name", "a1", "value", "cell", "content", "score"))
    record("Cell", "read_cell", "read cell A1", r, PASS if ok else FAIL, r)

    # ── delete_cell ───────────────────────────────────────────────────────────
    r = _cmd("write cell C1 TEMP")
    r = _cmd("delete cell C1")
    ok = any(w in r.lower() for w in ("delet", "clear", "done", "ok", "success", "remov"))
    record("Cell", "delete_cell", "delete cell C1", r, PASS if ok else FAIL, "")

    # ── delete_range ──────────────────────────────────────────────────────────
    r = _cmd("write cell D1 X")
    r = _cmd("write cell D2 Y")
    r = _cmd("delete range D1:D2")
    ok = any(w in r.lower() for w in ("delet", "clear", "done", "ok", "success", "remov", "rang"))
    record("Cell", "delete_range", "delete range D1:D2", r, PASS if ok else FAIL, "")

    # ── read_range ────────────────────────────────────────────────────────────
    r = _cmd("read range A1:B2")
    ok = any(w in r.lower() for w in ("name", "score", "a1", "b1", "range", "cell", "dave"))
    record("Cell", "read_range", "read range A1:B2", r, PASS if ok else FAIL, r[:60])


def test_formula_operations() -> None:
    section("4. FORMULA OPERATIONS")

    # Ensure data: write cells B2–B4 with numbers
    _cmd("write cell B2 95")
    _cmd("write cell B3 80")
    _cmd("write cell B4 75")
    _cmd("write cell A3 Eve")
    _cmd("write cell A4 Frank")

    # ── write_formula ─────────────────────────────────────────────────────────
    r = _cmd("write formula SUM B2:B4 in C1")
    ok = any(w in r.lower() for w in ("formula", "write", "done", "ok", "success", "sum", "c1"))
    record("Formula", "write_formula", "write formula SUM B2:B4 in C1", r, PASS if ok else FAIL, "")

    # ── write_sum_formula ─────────────────────────────────────────────────────
    r = _cmd("calculate sum B2:B4 in C2")
    ok = any(w in r.lower() for w in ("sum", "formula", "done", "ok", "success", "calcul"))
    record("Formula", "write_sum_formula", "calculate sum B2:B4 in C2", r, PASS if ok else FAIL, "")

    # ── write_average_formula ─────────────────────────────────────────────────
    r = _cmd("calculate average B2:B4 in C3")
    ok = any(w in r.lower() for w in ("average", "formula", "done", "ok", "success", "calcul"))
    record("Formula", "write_average_formula", "calculate average B2:B4 in C3",
           r, PASS if ok else FAIL, "")

    # ── calculate_row_sum ────────────────────────────────────────────────────
    r = _cmd("calculate row sum for row 2")
    ok = any(w in r.lower() for w in ("sum", "row", "done", "ok", "success", "calcul", "total"))
    record("Formula", "calculate_row_sum", "calculate row sum for row 2",
           r, PASS if ok else FAIL, "")

    # verify formula written
    try:
        wb = _wb()
        ws = wb.active
        c1 = ws["C1"].value
        ok_v = c1 is not None and ("SUM" in str(c1).upper() or "=" in str(c1))
        record("Formula", "write_formula[verify]", "openpyxl check",
               f"C1={c1}", PASS if ok_v else FAIL, "")
        wb.close()
    except Exception as e:
        record("Formula", "write_formula[verify]", "openpyxl check", "", FAIL, str(e))


def test_column_operations() -> None:
    section("5. COLUMN OPERATIONS")

    # ── add_column ────────────────────────────────────────────────────────────
    r = _cmd("add column Grade")
    ok = any(w in r.lower() for w in ("column", "add", "done", "ok", "success"))
    record("Column", "add_column", "add column Grade", r, PASS if ok else FAIL, "")

    # ── insert_column_at ──────────────────────────────────────────────────────
    r = _cmd("insert column at E")
    ok = any(w in r.lower() for w in ("insert", "column", "done", "ok", "success"))
    record("Column", "insert_column_at", "insert column at E", r, PASS if ok else FAIL, "")

    # ── delete_column ─────────────────────────────────────────────────────────
    r = _cmd("delete column E")
    ok = any(w in r.lower() for w in ("delet", "column", "done", "ok", "success", "remov"))
    record("Column", "delete_column", "delete column E", r, PASS if ok else FAIL, "")

    # ── set_column_width ──────────────────────────────────────────────────────
    r = _cmd("set column width A to 20")
    ok = any(w in r.lower() for w in ("width", "column", "done", "ok", "set", "success"))
    record("Column", "set_column_width", "set column width A to 20", r, PASS if ok else FAIL, "")


def test_sheet_operations() -> None:
    section("6. SHEET OPERATIONS")

    # ── create_sheet ─────────────────────────────────────────────────────────
    r = _cmd("create sheet Results")
    ok = any(w in r.lower() for w in ("creat", "sheet", "done", "ok", "success"))
    record("Sheet", "create_sheet", "create sheet Results", r, PASS if ok else FAIL, "")

    # verify sheet exists
    try:
        wb = _wb()
        ok_v = "Results" in wb.sheetnames
        record("Sheet", "create_sheet[verify]", "openpyxl check",
               str(wb.sheetnames), PASS if ok_v else FAIL, "")
        wb.close()
    except Exception as e:
        record("Sheet", "create_sheet[verify]", "", "", FAIL, str(e))

    # ── switch_sheet ──────────────────────────────────────────────────────────
    r = _cmd("switch sheet Results")
    ok = any(w in r.lower() for w in ("switch", "sheet", "results", "done", "ok", "activ", "now on"))
    record("Sheet", "switch_sheet", "switch sheet Results", r, PASS if ok else FAIL, "")

    # ── list_sheets ───────────────────────────────────────────────────────────
    r = _cmd("list sheets")
    ok = any(w in r.lower() for w in ("sheet", "results", "list", "sheet1", "available"))
    record("Sheet", "list_sheets", "list sheets", r, PASS if ok else FAIL, r[:80])

    # ── rename_sheet ──────────────────────────────────────────────────────────
    r = _cmd("rename sheet Results to Summary")
    ok = any(w in r.lower() for w in ("renam", "sheet", "done", "ok", "success", "summary"))
    record("Sheet", "rename_sheet", "rename sheet Results to Summary",
           r, PASS if ok else FAIL, "")

    # verify rename
    try:
        wb = _wb()
        ok_v = "Summary" in wb.sheetnames and "Results" not in wb.sheetnames
        record("Sheet", "rename_sheet[verify]", "openpyxl check",
               str(wb.sheetnames), PASS if ok_v else FAIL, "")
        wb.close()
    except Exception as e:
        record("Sheet", "rename_sheet[verify]", "", "", FAIL, str(e))

    # ── switch back to Sheet1 ─────────────────────────────────────────────────
    _cmd("switch sheet Sheet1")

    # ── delete_sheet ──────────────────────────────────────────────────────────
    r = _cmd("delete sheet Summary")
    ok = any(w in r.lower() for w in ("delet", "sheet", "done", "ok", "success", "remov"))
    record("Sheet", "delete_sheet", "delete sheet Summary", r, PASS if ok else FAIL, "")

    try:
        wb = _wb()
        ok_v = "Summary" not in wb.sheetnames
        record("Sheet", "delete_sheet[verify]", "openpyxl check",
               str(wb.sheetnames), PASS if ok_v else FAIL, "")
        wb.close()
    except Exception as e:
        record("Sheet", "delete_sheet[verify]", "", "", FAIL, str(e))


def test_formatting() -> None:
    section("7. FORMATTING")

    # ── format_cells — bold ───────────────────────────────────────────────────
    r = _cmd("format cells A1:B1 bold")
    ok = any(w in r.lower() for w in ("format", "bold", "done", "ok", "success", "appl"))
    record("Format", "format_cells[bold]", "format cells A1:B1 bold",
           r, PASS if ok else FAIL, "")

    # verify bold
    try:
        wb = _wb()
        ws = wb.active
        ok_v = ws["A1"].font and ws["A1"].font.bold is True
        record("Format", "format_cells[bold][verify]", "openpyxl check",
               f"A1 bold={ws['A1'].font.bold}", PASS if ok_v else FAIL, "")
        wb.close()
    except Exception as e:
        record("Format", "format_cells[bold][verify]", "", "", FAIL, str(e))

    # ── format_cells — font size ──────────────────────────────────────────────
    r = _cmd("format cells A1:B1 font size 14")
    ok = any(w in r.lower() for w in ("format", "font", "size", "done", "ok", "success", "appl"))
    record("Format", "format_cells[size]", "format cells A1:B1 font size 14",
           r, PASS if ok else FAIL, "")

    # ── format_cells — background color ──────────────────────────────────────
    r = _cmd("format cells A1:B1 background color yellow")
    ok = any(w in r.lower() for w in ("format", "color", "background", "done", "ok", "success", "appl"))
    record("Format", "format_cells[bg]", "format cells A1:B1 background color yellow",
           r, PASS if ok else FAIL, "")

    # ── merge_cells ───────────────────────────────────────────────────────────
    r = _cmd("merge cells E1:F1")
    ok = any(w in r.lower() for w in ("merge", "cell", "done", "ok", "success"))
    record("Format", "merge_cells", "merge cells E1:F1", r, PASS if ok else FAIL, "")

    try:
        wb = _wb()
        ws = wb.active
        merged = [str(m) for m in ws.merged_cells.ranges]
        ok_v = any("E1" in m for m in merged)
        record("Format", "merge_cells[verify]", "openpyxl check",
               str(merged), PASS if ok_v else FAIL, "")
        wb.close()
    except Exception as e:
        record("Format", "merge_cells[verify]", "", "", FAIL, str(e))

    # ── unmerge_cells ─────────────────────────────────────────────────────────
    r = _cmd("unmerge cells E1:F1")
    ok = any(w in r.lower() for w in ("unmerge", "merge", "cell", "done", "ok", "success"))
    record("Format", "unmerge_cells", "unmerge cells E1:F1", r, PASS if ok else FAIL, "")

    # ── freeze_panes ──────────────────────────────────────────────────────────
    r = _cmd("freeze panes B2")
    ok = any(w in r.lower() for w in ("freeze", "pane", "done", "ok", "success"))
    record("Format", "freeze_panes", "freeze panes B2", r, PASS if ok else FAIL, "")

    try:
        wb = _wb()
        ws = wb.active
        ok_v = ws.freeze_panes is not None
        record("Format", "freeze_panes[verify]", "openpyxl check",
               f"freeze_panes={ws.freeze_panes}", PASS if ok_v else FAIL, "")
        wb.close()
    except Exception as e:
        record("Format", "freeze_panes[verify]", "", "", FAIL, str(e))

    # ── unfreeze_panes ────────────────────────────────────────────────────────
    r = _cmd("unfreeze panes")
    ok = any(w in r.lower() for w in ("unfreeze", "freeze", "pane", "done", "ok", "success", "remov"))
    record("Format", "unfreeze_panes", "unfreeze panes", r, PASS if ok else FAIL, "")


def test_data_operations() -> None:
    section("8. DATA OPERATIONS")

    # Ensure data exists: write a clean Score column
    _cmd("write cell A1 Name")
    _cmd("write cell B1 Score")
    _cmd("write cell A2 Alice")
    _cmd("write cell B2 85")
    _cmd("write cell A3 Bob")
    _cmd("write cell B3 72")
    _cmd("write cell A4 Charlie")
    _cmd("write cell B4 90")
    _cmd("write cell A5 Dave")
    _cmd("write cell B5 60")

    # ── sort_range ────────────────────────────────────────────────────────────
    r = _cmd("sort range by column B ascending")
    ok = any(w in r.lower() for w in ("sort", "done", "ok", "success", "ascend", "order"))
    record("Data", "sort_range[asc]", "sort range by column B ascending",
           r, PASS if ok else FAIL, "")

    # verify sort order
    try:
        wb = _wb()
        ws = wb.active
        scores = [ws.cell(row=i, column=2).value for i in range(2, 6)]
        ok_v = scores == sorted(scores)
        record("Data", "sort_range[verify]", "openpyxl check",
               str(scores), PASS if ok_v else FAIL,
               "ascending" if ok_v else f"not sorted: {scores}")
        wb.close()
    except Exception as e:
        record("Data", "sort_range[verify]", "", "", FAIL, str(e))

    # ── sort descending ───────────────────────────────────────────────────────
    r = _cmd("sort range by column B descending")
    ok = any(w in r.lower() for w in ("sort", "done", "ok", "success", "descend", "order"))
    record("Data", "sort_range[desc]", "sort range by column B descending",
           r, PASS if ok else FAIL, "")

    # ── find_replace ──────────────────────────────────────────────────────────
    r = _cmd("find Alice replace with Alicia")
    ok = any(w in r.lower() for w in ("find", "replac", "done", "ok", "success", "alicia", "alicia"))
    record("Data", "find_replace", "find Alice replace with Alicia",
           r, PASS if ok else FAIL, "")

    try:
        wb = _wb()
        ws = wb.active
        vals = [ws.cell(row=i, column=1).value for i in range(1, ws.max_row + 1)]
        ok_v = "Alicia" in vals and "Alice" not in vals
        record("Data", "find_replace[verify]", "openpyxl check",
               str(vals), PASS if ok_v else FAIL,
               "'Alicia' present, 'Alice' gone" if ok_v else f"got {vals}")
        wb.close()
    except Exception as e:
        record("Data", "find_replace[verify]", "", "", FAIL, str(e))

    # ── filter_table ──────────────────────────────────────────────────────────
    r = _cmd("filter table column B greater than 75")
    ok = any(w in r.lower() for w in ("filter", "column", "done", "ok", "success", "hidden", "row"))
    record("Data", "filter_table", "filter table column B greater than 75",
           r, PASS if ok else FAIL, r[:60])

    # ── clear_filter ──────────────────────────────────────────────────────────
    r = _cmd("clear filter")
    ok = any(w in r.lower() for w in ("clear", "filter", "done", "ok", "success", "remov", "show"))
    record("Data", "clear_filter", "clear filter", r, PASS if ok else FAIL, "")


def test_chart_operations() -> None:
    section("9. CHART OPERATIONS")

    r = _cmd("create bar chart from B2:B5 titled Scores")
    ok = any(w in r.lower() for w in ("chart", "creat", "done", "ok", "success", "bar", "add"))
    record("Chart", "create_chart", "create bar chart from B2:B5 titled Scores",
           r, PASS if ok else FAIL, r[:80])


def test_undo() -> None:
    section("10. UNDO")

    # Make a known change then undo it
    _cmd("write cell Z99 UNDO_TEST")

    try:
        wb = _wb()
        ws = wb.active
        before = ws["Z99"].value
        wb.close()
    except Exception as e:
        before = None

    r = _cmd("undo")
    ok = any(w in r.lower() for w in ("undo", "revert", "done", "ok", "success", "restor"))
    record("Undo", "undo", "undo", r, PASS if ok else FAIL, "")

    try:
        wb = _wb()
        ws = wb.active
        after = ws["Z99"].value
        ok_v = after != "UNDO_TEST"
        record("Undo", "undo[verify]", "openpyxl check",
               f"Z99 before={before!r} after={after!r}",
               PASS if ok_v else FAIL,
               "cell reverted" if ok_v else "cell NOT reverted — undo may not have fired")
        wb.close()
    except Exception as e:
        record("Undo", "undo[verify]", "", "", FAIL, str(e))


def test_excel_app() -> None:
    section("11. EXCEL APPLICATION")

    r = _cmd("open excel")
    ok = any(w in r.lower() for w in ("open", "excel", "launch", "start", "done", "ok", "alread"))
    record("App", "open_excel_app", "open excel", r, PASS if ok else FAIL, r[:80])

    # Note: close_excel_app is destructive — only test if user has no real file open
    # We skip auto-execution here to avoid killing the user's Excel session
    record("App", "close_excel_app", "(skipped — would close user Excel)", "",
           SKIP, "destructive — skip in automated audit")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    global _audit_file

    # ── Create isolated temp directory with the audit file ───────────────────
    tmp_dir = tempfile.mkdtemp(prefix="sara_audit_")
    _audit_file = os.path.join(tmp_dir, "test_audit.xlsx")
    print(f"\n{'━'*64}")
    print(f"  Sara Command Audit — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Temp file: {_audit_file}")
    print(f"{'━'*64}")

    # Bootstrap: point the agent's files_dir at our temp dir
    # so all file ops land there, not in the real /files/ folder.
    import agent as _agent_mod
    _agent_mod._default_agent = None   # reset singleton
    _agent_mod.Agent.__init__.__doc__  # just access, no change
    # Patch the files dir for this run
    old_agent = _agent_mod._default_agent
    _agent_mod._default_agent = _agent_mod.Agent(tmp_dir)

    try:
        test_file_operations()
        test_row_operations()
        test_cell_operations()
        test_formula_operations()
        test_column_operations()
        test_sheet_operations()
        test_formatting()
        test_data_operations()
        test_chart_operations()
        test_undo()
        test_excel_app()
    finally:
        # ── Print summary ─────────────────────────────────────────────────────
        passed  = sum(1 for r in _results if r["status"] == PASS)
        failed  = sum(1 for r in _results if r["status"] == FAIL)
        skipped = sum(1 for r in _results if r["status"] == SKIP)
        total   = len(_results)

        print(f"\n{'━'*64}")
        print(f"  RESULTS: {passed}/{total} passed   {failed} failed   {skipped} skipped")
        print(f"{'━'*64}")

        if failed:
            print("\n  ── FAILURES ──")
            for r in _results:
                if r["status"] == FAIL:
                    print(f"  ✘ [{r['cmd_type']}]  '{r['natural']}'")
                    print(f"      response: {r['response']}")
                    if r['detail']:
                        print(f"      detail:   {r['detail']}")

        print("\n  ── FULL REPORT ──")
        col_w = [12, 28, 6, 50]
        header = f"  {'Category':<{col_w[0]}} {'Command':<{col_w[1]}} {'Status':<{col_w[2]}} {'Detail / Response'}"
        print(header)
        print("  " + "─" * 100)
        for r in _results:
            icon = "✔" if r["status"] == PASS else ("⚠" if r["status"] == SKIP else "✘")
            detail = r["detail"] or r["response"]
            print(f"  {icon} {r['category']:<{col_w[0]}} {r['cmd_type']:<{col_w[1]}} "
                  f"{r['status']:<{col_w[2]}} {detail[:col_w[3]]}")

        # ── Risk report ───────────────────────────────────────────────────────
        print(f"\n{'━'*64}")
        print("  RISK REPORT")
        print(f"{'━'*64}")

        # Categorise failures
        cmd_fail = [r for r in _results if r["status"] == FAIL and "[verify]" not in r["cmd_type"]]
        verify_fail = [r for r in _results if r["status"] == FAIL and "[verify]" in r["cmd_type"]]

        if not cmd_fail and not verify_fail:
            print("  ✔ No risks detected — all commands stable.")
        else:
            if cmd_fail:
                print(f"\n  HIGH RISK — Command returned unexpected response ({len(cmd_fail)}):")
                for r in cmd_fail:
                    print(f"    • {r['cmd_type']}: {r['response'][:80]}")
            if verify_fail:
                print(f"\n  MEDIUM RISK — Command ran but Excel state not as expected ({len(verify_fail)}):")
                for r in verify_fail:
                    print(f"    • {r['cmd_type']}: {r['detail'] or r['response']}")

        # ── Cleanup ───────────────────────────────────────────────────────────
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
        print(f"\n  Temp dir cleaned up: {tmp_dir}")
        print(f"{'━'*64}\n")

        # Return exit code for CI
        sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
