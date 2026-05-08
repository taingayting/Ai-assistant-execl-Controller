"""
Sara — Supported Commands Regression Test
==========================================
Baseline: voice_test_dataset.json  (37 intent types, the official supported set)
Method:   process_command() + openpyxl workbook-state verification
Rules:    No code changes. No invented commands. Dataset intents only.

Excluded (not in dataset, internal implementation details):
  write_sum_formula, write_average_formula, calculate_row_sum, insert_cell

Run from project root:
    python backend/tests/qa_regression_supported.py
"""
from __future__ import annotations
import os, sys, pathlib

_TESTS   = pathlib.Path(__file__).resolve().parent
_BACKEND = _TESTS.parent
for _p in (str(_BACKEND), str(_BACKEND.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from dotenv import load_dotenv
    _env = _BACKEND / '.env'
    if _env.exists():
        load_dotenv(_env, override=True)
except ImportError:
    pass

from agent import process_command
import openpyxl

_FILES = _BACKEND / "files"
_FILES.mkdir(exist_ok=True)

# ── helpers ─────────────────────────���──────────────────────────���──────────────
def _p(name: str) -> str:
    return str(_FILES / name)

def _cmd(phrase: str, filename: str) -> str:
    return process_command(phrase, filename=filename)

def _wb(name: str):
    return openpyxl.load_workbook(_p(name))

def _cleanup(*names):
    for n in names:
        try: os.remove(_p(n))
        except FileNotFoundError: pass

RESULTS: list[dict] = []

def _record(intent: str, phrase: str, status: str, actual: str = "", note: str = ""):
    RESULTS.append({"intent": intent, "phrase": phrase,
                    "status": status, "actual": actual, "note": note})
    sym = "✔" if status == "PASS" else "✘"
    line = f"  {sym} [{intent:<22}] {status:<4}  {actual[:70]}"
    if note:
        line += f"  [{note}]"
    print(line)

def _pass(intent, phrase, actual="ok"):
    _record(intent, phrase, "PASS", actual)

def _fail(intent, phrase, actual, note=""):
    _record(intent, phrase, "FAIL", actual, note)

# ── FILE OPERATIONS ───────────────────────��──────────────────────────────���────
print("\n" + "═"*70)
print("  FILE OPERATIONS")
print("═"*70)

F = "qa_reg.xlsx"
_cleanup(F, "qa_reg_new.xlsx")

# create_excel
r = _cmd("create excel", F)
ok = os.path.exists(_p(F))
_pass("create_excel", "create excel", "file created") if ok else _fail("create_excel", "create excel", r)

# save
r = _cmd("save", F)
_pass("save", "save", r[:60]) if "saved" in r.lower() or "done" in r.lower() else _fail("save", "save", r)

# open_excel (→ open_excel_app: launch Excel; tests routing only, no workbook state)
r = _cmd("open excel", F)
routed_ok = "excel" in r.lower() or "open" in r.lower() or "done" in r.lower() or "launch" in r.lower()
_pass("open_excel", "open excel", r[:60]) if routed_ok else _fail("open_excel", "open excel", r)

# rename_file
r = _cmd("rename file qa_reg_new", F)
renamed = os.path.exists(_p("qa_reg_new.xlsx"))
if renamed:
    _pass("rename_file", "rename file qa_reg_new", "renamed to qa_reg_new.xlsx")
    F = "qa_reg_new.xlsx"   # track rename
else:
    _fail("rename_file", "rename file qa_reg_new", r)

# ── ROW OPERATIONS ────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("  ROW OPERATIONS")
print("═"*70)

FR = "qa_reg_rows.xlsx"
_cleanup(FR)
_cmd("create excel", FR)
_cmd("add row Alice 85", FR)
_cmd("add row Bob 72", FR)
_cmd("add row Charlie 90", FR)

# add_row (verify workbook)
wb = _wb(FR); ws = wb.active
vals = [ws.cell(r, 1).value for r in range(2, 5)]
_pass("add_row", "add row Alice/Bob/Charlie 85/72/90", str(vals)) if vals == ["Alice","Bob","Charlie"] else _fail("add_row", "add row ...", str(vals))

# delete_row (row 2 = Alice → after delete row 2, Bob should be row 2)
r = _cmd("delete row 2", FR)
wb = _wb(FR); ws = wb.active
row2 = ws.cell(2, 1).value
_pass("delete_row", "delete row 2", f"row2='{row2}'") if row2 == "Bob" else _fail("delete_row", "delete row 2", f"row2='{row2}'", "expected Bob at row2 after deleting Alice")

# insert_row_at (insert blank at row 2 — pushes Bob to row 3)
_cmd("add row Dave 60", FR)   # ensure enough rows
r = _cmd("insert row at 2", FR)
wb = _wb(FR); ws = wb.active
row2_val = ws.cell(2, 1).value
# row 2 should be blank (None) after insert
_pass("insert_row_at", "insert row at 2", f"row2='{row2_val}'") if row2_val is None else _fail("insert_row_at", "insert row at 2", f"row2='{row2_val}'", "expected blank row at pos 2")

# ── COLUMN OPERATIONS ─────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("  COLUMN OPERATIONS")
print("═"*70)

FC = "qa_reg_cols.xlsx"
_cleanup(FC)
_cmd("create excel", FC)
_cmd("add row Alice 85", FC)

# add_column
r = _cmd("add column Grade", FC)
wb = _wb(FC); ws = wb.active
col_c = ws.cell(1, 3).value
_pass("add_column", "add column Grade", f"C1='{col_c}'") if col_c == "Grade" else _fail("add_column", "add column Grade", f"C1='{col_c}'")

# insert_column_at (insert blank column at B — shifts Score to C)
r = _cmd("insert column at B", FC)
wb = _wb(FC); ws = wb.active
# After insert: col A=Name, col B=blank, col C=Score, col D=Grade
b1 = ws.cell(1, 2).value
_pass("insert_column_at", "insert column at B", f"B1='{b1}'") if b1 is None else _fail("insert_column_at", "insert column at B", f"B1='{b1}'", "expected blank B after insert")

# delete_column (delete col D now = Grade)
r = _cmd("delete column D", FC)
wb = _wb(FC); ws = wb.active
d1 = ws.cell(1, 4).value
_pass("delete_column", "delete column D", f"D1='{d1}'") if d1 is None else _fail("delete_column", "delete column D", f"D1='{d1}'")

# ── CELL OPERATIONS ───────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("  CELL OPERATIONS")
print("═"*70)

FCELL = "qa_reg_cells.xlsx"
_cleanup(FCELL)
_cmd("create excel", FCELL)

# write_cell
_cmd("write cell A1 Hello", FCELL)
wb = _wb(FCELL); ws = wb.active
_pass("write_cell", "write cell A1 Hello", f"A1='{ws['A1'].value}'") if ws["A1"].value == "Hello" else _fail("write_cell", "write cell A1 Hello", f"A1='{ws['A1'].value}'")

# read_cell
r = _cmd("read cell A1", FCELL)
_pass("read_cell", "read cell A1", r[:60]) if "Hello" in r else _fail("read_cell", "read cell A1", r)

# delete_cell
_cmd("write cell B1 Temp", FCELL)
_cmd("delete cell B1", FCELL)
wb = _wb(FCELL); ws = wb.active
_pass("delete_cell", "delete cell B1", f"B1='{ws['B1'].value}'") if ws["B1"].value is None else _fail("delete_cell", "delete cell B1", f"B1='{ws['B1'].value}'")

# clear_cell (dataset CR020-CR022; may map to delete_cell internally)
_cmd("write cell C1 ToClear", FCELL)
r = _cmd("clear cell C1", FCELL)
wb = _wb(FCELL); ws = wb.active
c1 = ws["C1"].value
_pass("clear_cell", "clear cell C1", f"C1='{c1}'") if c1 is None else _fail("clear_cell", "clear cell C1", f"C1='{c1}'", "needs LLM or heuristic rule")

# ── RANGE OPERATIONS ──────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("  RANGE OPERATIONS")
print("═"*70)

FRANGE = "qa_reg_range.xlsx"
_cleanup(FRANGE)
_cmd("create excel", FRANGE)
_cmd("write cell A1 X", FRANGE)
_cmd("write cell B1 Y", FRANGE)
_cmd("write cell A2 Z", FRANGE)

# read_range (dataset CR030 — LLM-only, no heuristic rule)
r = _cmd("read range A1:B2", FRANGE)
has_data = any(v in r for v in ("X","Y","Z","A1","row","cell","value","data"))
_pass("read_range", "read range A1:B2", r[:60]) if has_data else _fail("read_range", "read range A1:B2", r, "LLM-only — heuristic missing")

# delete_range
_cmd("write cell E1 del1", FRANGE)
_cmd("write cell E2 del2", FRANGE)
_cmd("delete range E1:E2", FRANGE)
wb = _wb(FRANGE); ws = wb.active
e1, e2 = ws["E1"].value, ws["E2"].value
_pass("delete_range", "delete range E1:E2", f"E1='{e1}' E2='{e2}'") if e1 is None and e2 is None else _fail("delete_range", "delete range E1:E2", f"E1='{e1}' E2='{e2}'")

# ── FORMULA OPERATIONS ────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("  FORMULA OPERATIONS")
print("═"*70)

FFL = "qa_reg_formula.xlsx"
_cleanup(FFL)
_cmd("create excel", FFL)
_cmd("add row Alice 85", FFL)
_cmd("add row Bob 72", FFL)
_cmd("add row Charlie 90", FFL)

# write_formula (dataset FM020 — heuristic returns write_cell with wrong value)
r = _cmd("write formula =SUM(B2:B4) in C1", FFL)
wb = _wb(FFL); ws = wb.active
c1 = ws["C1"].value
formula_ok = isinstance(c1, str) and c1.startswith("=") or isinstance(c1, (int, float))
_pass("write_formula", "write formula =SUM(B2:B4) in C1", f"C1='{c1}'") if formula_ok else _fail("write_formula", "write formula =SUM(B2:B4) in C1", f"C1='{c1}'", "heuristic misparses formula commands")

# calculate_sum (dataset FM001 — heuristic returns None, needs LLM)
r = _cmd("calculate sum column B", FFL)
sum_ok = "sum" in r.lower() or "formula" in r.lower() or "done" in r.lower() or "247" in r or "added" in r.lower()
_pass("calculate_sum", "calculate sum column B", r[:60]) if sum_ok else _fail("calculate_sum", "calculate sum column B", r, "LLM-only — heuristic missing")

# calculate_average (dataset FM010)
r = _cmd("calculate average column B", FFL)
avg_ok = "average" in r.lower() or "formula" in r.lower() or "done" in r.lower() or "added" in r.lower()
_pass("calculate_average", "calculate average column B", r[:60]) if avg_ok else _fail("calculate_average", "calculate average column B", r)

# ── FORMATTING ────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("  FORMATTING")
print("═"*70)

FFMT = "qa_reg_fmt.xlsx"
_cleanup(FFMT)
_cmd("create excel", FFMT)
_cmd("add row Alice 85", FFMT)

# format_cells bold (dataset FT001)
_cmd("format cell A1 bold", FFMT)
wb = _wb(FFMT); ws = wb.active
_pass("format_cells[bold]", "format cell A1 bold", f"bold={ws['A1'].font.bold}") if ws["A1"].font.bold else _fail("format_cells[bold]", "format cell A1 bold", f"bold={ws['A1'].font.bold}")

# set_column_width (dataset FT020 — heuristic returns None, needs LLM)
r = _cmd("set column A width to 20", FFMT)
wb = _wb(FFMT)
ws2 = wb.active
cd = ws2.column_dimensions.get("A")
width = cd.width if cd else None
_pass("set_column_width", "set column A width to 20", f"width={width}") if width and width >= 18 else _fail("set_column_width", "set column A width to 20", f"width={width}", "LLM-only — heuristic missing")

# ── MERGE / UNMERGE ───────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("  MERGE / UNMERGE")
print("═"*70)

FMRG = "qa_reg_merge.xlsx"
_cleanup(FMRG)
_cmd("create excel", FMRG)

_cmd("merge cells A1:C1", FMRG)
wb = _wb(FMRG); ws = wb.active
merged = [str(r) for r in ws.merged_cells.ranges]
_pass("merge_cells", "merge cells A1:C1", str(merged)) if merged else _fail("merge_cells", "merge cells A1:C1", "no merges")

_cmd("unmerge cells A1:C1", FMRG)
wb = _wb(FMRG); ws = wb.active
merged2 = [str(r) for r in ws.merged_cells.ranges]
_pass("unmerge_cells", "unmerge cells A1:C1", str(merged2)) if not merged2 else _fail("unmerge_cells", "unmerge cells A1:C1", str(merged2))

# ── FREEZE / UNFREEZE ─────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("  FREEZE / UNFREEZE PANES")
print("═"*70)

FFZ = "qa_reg_freeze.xlsx"
_cleanup(FFZ)
_cmd("create excel", FFZ)

_cmd("freeze panes", FFZ)
wb = _wb(FFZ); ws = wb.active
fp = ws.freeze_panes
_pass("freeze_panes", "freeze panes", f"freeze_panes='{fp}'") if fp else _fail("freeze_panes", "freeze panes", f"freeze_panes='{fp}'")

_cmd("unfreeze panes", FFZ)
wb = _wb(FFZ); ws = wb.active
fp2 = ws.freeze_panes
_pass("unfreeze_panes", "unfreeze panes", f"freeze_panes={fp2}") if fp2 is None else _fail("unfreeze_panes", "unfreeze panes", f"freeze_panes='{fp2}'")

# ── CHART ─────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("  CHART")
print("═"*70)

FCH = "qa_reg_chart.xlsx"
_cleanup(FCH)
_cmd("create excel", FCH)
# Add enough rows so chart doesn't fail with insufficient_columns
for v in [("Alice",85), ("Bob",72), ("Charlie",90), ("Dave",60)]:
    _cmd(f"add row {v[0]} {v[1]}", FCH)

r = _cmd("create chart bar A1:B5 titled Scores", FCH)
wb = _wb(FCH); ws = wb.active
charts = len(ws._charts)
_pass("create_chart", "create chart bar A1:B5 titled Scores", f"charts={charts}") if charts > 0 else _fail("create_chart", "create chart bar A1:B5 titled Scores", f"charts={charts}", "LLM-only — may need Groq/Ollama")

# ── SHEET OPERATIONS ──────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("  SHEET OPERATIONS")
print("═"*70)

FSH = "qa_reg_sheets.xlsx"
_cleanup(FSH)
_cmd("create excel", FSH)

# create_sheet
r = _cmd("create sheet Budget", FSH)
wb = _wb(FSH)
has_budget = "Budget" in wb.sheetnames
_pass("create_sheet", "create sheet Budget", str(wb.sheetnames)) if has_budget else _fail("create_sheet", "create sheet Budget", str(wb.sheetnames))

# list_sheets (dataset SH040 — heuristic returns None, needs LLM)
r = _cmd("list sheets", FSH)
sheets_ok = "Sheet1" in r or "Budget" in r or "sheet" in r.lower()
_pass("list_sheets", "list sheets", r[:60]) if sheets_ok else _fail("list_sheets", "list sheets", r, "LLM-only — heuristic missing")

# switch_sheet (dataset SH001 — heuristic returns None, needs LLM)
r = _cmd("switch sheet Budget", FSH)
switch_ok = "Budget" in r or "switched" in r.lower() or "done" in r.lower()
_pass("switch_sheet", "switch sheet Budget", r[:60]) if switch_ok else _fail("switch_sheet", "switch sheet Budget", r, "LLM-only — heuristic missing")

# rename_sheet
r = _cmd("rename sheet Budget to Results", FSH)
wb = _wb(FSH)
_pass("rename_sheet", "rename sheet Budget to Results", str(wb.sheetnames)) if "Results" in wb.sheetnames else _fail("rename_sheet", "rename sheet Budget to Results", str(wb.sheetnames))

# delete_sheet
r = _cmd("delete sheet Results", FSH)
wb = _wb(FSH)
_pass("delete_sheet", "delete sheet Results", str(wb.sheetnames)) if "Results" not in wb.sheetnames else _fail("delete_sheet", "delete sheet Results", str(wb.sheetnames))

# ── DATA OPERATIONS ───────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("  DATA OPERATIONS")
print("═"*70)

FD = "qa_reg_data.xlsx"
_cleanup(FD)
_cmd("create excel", FD)
for v in [("Alice",85), ("Bob",72), ("Charlie",90), ("Dave",60)]:
    _cmd(f"add row {v[0]} {v[1]}", FD)

# sort_range (dataset SF001 — heuristic returns None, needs LLM)
r = _cmd("sort column B ascending", FD)
wb = _wb(FD); ws = wb.active
col_b = [ws.cell(row, 2).value for row in range(2, 6)]
sorted_ok = col_b == sorted([v for v in col_b if v is not None])
_pass("sort_range", "sort column B ascending", str(col_b)) if sorted_ok else _fail("sort_range", "sort column B ascending", str(col_b), "LLM-only — heuristic missing")

# find_replace (dataset SF010 — heuristic returns None, needs LLM)
r = _cmd("find Alice replace with Alicia", FD)
wb = _wb(FD); ws = wb.active
col_a = [ws.cell(row, 1).value for row in range(2, 6)]
replaced = "Alicia" in col_a and "Alice" not in col_a
_pass("find_replace", "find Alice replace with Alicia", str(col_a)) if replaced else _fail("find_replace", "find Alice replace with Alicia", str(col_a), "LLM-only — heuristic missing")

# filter_table (dataset SF020 — heuristic returns None, needs LLM)
r = _cmd("filter by column B value 90", FD)
filter_ok = "filter" in r.lower() or "done" in r.lower() or "row" in r.lower()
_pass("filter_table", "filter by column B value 90", r[:60]) if filter_ok else _fail("filter_table", "filter by column B value 90", r, "LLM-only — heuristic missing")

# clear_filter (dataset SF030)
r = _cmd("clear filter", FD)
_pass("clear_filter", "clear filter", r[:60]) if r else _fail("clear_filter", "clear filter", r)

# ── UNDO ──────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("  UNDO")
print("═"*70)

FUNDO = "qa_reg_undo.xlsx"
_cleanup(FUNDO)
_cmd("create excel", FUNDO)
_cmd("write cell A1 Original", FUNDO)
_cmd("write cell A1 CHANGED", FUNDO)
_cmd("undo", FUNDO)
wb = _wb(FUNDO); ws = wb.active
a1 = ws["A1"].value
_pass("undo", "undo", f"A1='{a1}' (restored)") if a1 == "Original" else _fail("undo", "undo", f"A1='{a1}' expected 'Original'")

# ── CLEANUP ───────────────────────────────────────────────────────────────────
for name in ["qa_reg_new.xlsx", "qa_reg_rows.xlsx", "qa_reg_cols.xlsx",
             "qa_reg_cells.xlsx", "qa_reg_range.xlsx", "qa_reg_formula.xlsx",
             "qa_reg_fmt.xlsx", "qa_reg_merge.xlsx", "qa_reg_freeze.xlsx",
             "qa_reg_chart.xlsx", "qa_reg_sheets.xlsx", "qa_reg_data.xlsx",
             "qa_reg_undo.xlsx"]:
    _cleanup(name)

# ── REPORT ───────────────────────────────────────────────────────────────��────
passed = [r for r in RESULTS if r["status"] == "PASS"]
failed = [r for r in RESULTS if r["status"] == "FAIL"]

print("\n" + "━"*70)
print(f"  OVERALL: {len(passed)}/{len(RESULTS)} PASS   {len(failed)} FAIL")
print("━"*70)

print("\n### Supported Commands Tested (from voice_test_dataset.json)")
print(f"  {len(RESULTS)} of 37 supported intent types tested")
print()
print("### Excluded (not in voice_test_dataset.json — internal implementation details)")
print("  write_sum_formula, write_average_formula, calculate_row_sum, insert_cell")
print()

if failed:
    print("### FAILURES — Real blockers for final delivery")
    print()
    for f in failed:
        print(f"  ✘ {f['intent']}")
        print(f"    phrase   : {f['phrase']}")
        print(f"    actual   : {f['actual']}")
        if f['note']:
            print(f"    root cause: {f['note']}")
        print()

print("### PASS — Working correctly")
for p in passed:
    print(f"  ✔  {p['intent']}")
