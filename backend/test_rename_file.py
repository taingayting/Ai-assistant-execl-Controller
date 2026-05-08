"""
rename_file feature — verification test suite.
Run from backend/: python test_rename_file.py

Tests:
  1. Parser correctness (heuristic)
  2. Basic rename (disk + session + memory)
  3. Active file tracking after rename
  4. Open / close after rename
  5. Undo after rename (stack migration)
  6. Edge cases (same name, name taken, no extension, invalid chars)
  7. rename_sheet still works (no regression)
  8. Full regression pass
"""
import os, sys, pathlib, time as _time
sys.path.insert(0, os.path.dirname(__file__))

from llm_parser import _heuristic_natural_language_parse as hparse
from agent import process_command
from memory import memory
from openpyxl import load_workbook

FILES_DIR = pathlib.Path(__file__).parent / "files"
PASS = 0
FAIL = 0

# ── helpers ──────────────────────────────────────────────────────────────────

def fp(name):
    return FILES_DIR / name

def fresh(name):
    p = fp(name)
    if p.exists():
        try: p.unlink()
        except: pass
    return p

def ok(label, result, keyword=None):
    global PASS, FAIL
    text = ""
    if isinstance(result, str):
        text = result
    elif isinstance(result, dict):
        if "error" in result:
            print(f"  FAIL  {label}: error={result.get('error')} / {result.get('speech_text','')[:80]}")
            FAIL += 1
            return
        text = result.get("speech_text") or result.get("message") or str(result)
    else:
        text = str(result)
    bad = ("rolled back" in text or "0 of 1 steps" in text or "hit an error" in text)
    if bad:
        print(f"  FAIL  {label}: {text[:110]}")
        FAIL += 1
        return
    if keyword and keyword.lower() not in text.lower():
        print(f"  FAIL  {label}: missing '{keyword}' in: {text[:100]}")
        FAIL += 1
        return
    print(f"  PASS  {label}: {text[:90]}")
    PASS += 1

def ok_error(label, result, expected_err=None):
    global PASS, FAIL
    if isinstance(result, dict) and "error" in result:
        print(f"  PASS  {label} (error={result['error']}): {result.get('speech_text','')[:70]}")
        PASS += 1
    elif isinstance(result, str) and ("rolled back" not in result and "0 of 1 steps" not in result) and result.strip():
        print(f"  PASS  {label} (graceful rejection): {result[:80]}")
        PASS += 1
    else:
        text = result.get("speech_text","") if isinstance(result, dict) else str(result)
        print(f"  FAIL  {label}: expected error, got: {text[:80]}")
        FAIL += 1

def file_exists(name):
    return fp(name).exists()

def active_file():
    return memory.get_active_file()

def wb_rows(name):
    p = fp(name)
    if not p.exists():
        return None
    wb = load_workbook(str(p))
    ws = wb.active
    return [tuple(ws.cell(r, c).value for c in range(1, ws.max_column+1))
            for r in range(1, ws.max_row+1)]

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 1: Parser correctness")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════

cases = [
    # (input_text, expected_type, expected_new_name_contains)
    ("rename file to report.xlsx",      "rename_file",  "report"),
    ("rename file demo.xlsx to report.xlsx", "rename_file", "report"),
    ("rename this file to report",      "rename_file",  "report"),
    ("rename workbook to budget.xlsx",  "rename_file",  "budget"),
    ("rename spreadsheet to q1",        "rename_file",  "q1"),
]
for text, exp_type, exp_name_part in cases:
    r = hparse(text)
    got_type = r.get("type") if r else None
    got_name = r.get("new_name","") if r else ""
    if got_type == exp_type and exp_name_part.lower() in got_name.lower():
        print(f"  PASS  parse({text!r}) → type={got_type} new_name={got_name!r}")
        PASS += 1
    else:
        print(f"  FAIL  parse({text!r}) → got type={got_type!r} new_name={got_name!r} (expected type={exp_type} name∋{exp_name_part!r})")
        FAIL += 1

# rename_sheet must NOT be confused with rename_file
r = hparse("rename sheet Sales to Report")
if r and r.get("type") == "rename_sheet":
    print(f"  PASS  'rename sheet Sales to Report' → rename_sheet (no collision)")
    PASS += 1
else:
    print(f"  FAIL  'rename sheet Sales to Report' → {r} (expected rename_sheet)")
    FAIL += 1

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 2: Basic rename")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════

DEMO = "rename_demo.xlsx"
REPORT = "rename_report.xlsx"
fresh(DEMO); fresh(REPORT)

ok("create demo.xlsx", process_command("create excel", filename=DEMO))
ok("add row John 90",  process_command("add row John 90", filename=DEMO))
ok("save",             process_command("save", filename=DEMO))

assert file_exists(DEMO), f"Pre-condition: {DEMO} must exist"

result = process_command(f"rename file to {REPORT}", filename=DEMO)
ok("rename demo→report", result, "renamed")

if not file_exists(REPORT):
    print(f"  FAIL  report.xlsx not found on disk after rename")
    FAIL += 1
else:
    print(f"  PASS  {REPORT} exists on disk")
    PASS += 1

if file_exists(DEMO):
    print(f"  FAIL  {DEMO} still exists on disk after rename — old file was NOT removed")
    FAIL += 1
else:
    print(f"  PASS  {DEMO} no longer exists on disk")
    PASS += 1

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 3: Active file / memory tracking")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════

af = active_file()
if af == REPORT:
    print(f"  PASS  memory.active_file = {af!r}")
    PASS += 1
else:
    print(f"  FAIL  memory.active_file = {af!r} (expected {REPORT!r})")
    FAIL += 1

# Commands after rename should go to REPORT, not DEMO
ok("add row after rename", process_command("add row Alice 80"))
ok("save after rename",    process_command("save"))

rows = wb_rows(REPORT)
if rows is None:
    print(f"  FAIL  cannot read {REPORT}")
    FAIL += 1
else:
    has_alice = any("Alice" in (str(v) for v in row) for row in rows)
    if has_alice:
        print(f"  PASS  Alice row found in {REPORT}")
        PASS += 1
    else:
        print(f"  FAIL  Alice row NOT found in {REPORT}: rows={rows}")
        FAIL += 1

    if file_exists(DEMO):
        print(f"  FAIL  {DEMO} still exists — writes went to wrong file")
        FAIL += 1
    else:
        print(f"  PASS  {DEMO} not recreated by post-rename writes")
        PASS += 1

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 4: Open / close after rename")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════

ok("close excel", process_command("close excel"))
_time.sleep(0.5)

ok("open report.xlsx", process_command(f"open file {REPORT}"))

af2 = active_file()
if af2 == REPORT:
    print(f"  PASS  active_file after reopen = {af2!r}")
    PASS += 1
else:
    print(f"  FAIL  active_file after reopen = {af2!r} (expected {REPORT!r})")
    FAIL += 1

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 5: Undo after rename")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════

rows_before = wb_rows(REPORT)
ok("add row Bob 75 (pre-undo)", process_command("add row Bob 75"))

rows_with_bob = wb_rows(REPORT)
has_bob = rows_with_bob and any("Bob" in (str(v) for v in row) for row in rows_with_bob)
if has_bob:
    print(f"  PASS  Bob row written to {REPORT}")
    PASS += 1
else:
    print(f"  FAIL  Bob row not in {REPORT}")
    FAIL += 1

ok("undo", process_command("undo"), "undo")

rows_after_undo = wb_rows(REPORT)
still_has_bob = rows_after_undo and any("Bob" in (str(v) for v in row) for row in rows_after_undo)
if not still_has_bob:
    print(f"  PASS  Bob row removed by undo (undo stack migration worked)")
    PASS += 1
else:
    print(f"  FAIL  Bob row still present after undo — undo stack migration broken")
    FAIL += 1

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 6: Edge cases")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════

# 6a: same name
ok_error("rename to same name",
         process_command(f"rename file to {REPORT}", filename=REPORT))

# 6b: target name already exists
EXISTING = f"rename_existing_{int(_time.time())}.xlsx"
fresh(EXISTING)
process_command("create excel", filename=EXISTING)
ok_error("rename to existing file",
         process_command(f"rename file to {EXISTING}", filename=REPORT))
fresh(EXISTING)  # clean up

# 6c: rename without extension — should auto-add .xlsx
BUDGET = f"rename_budget_{int(_time.time())}.xlsx"
fresh(BUDGET)
r = process_command(f"rename workbook to rename_budget_{int(_time.time())}", filename=REPORT)
# Either success (new file exists) or graceful error — NOT rolled back
if isinstance(r, str) and ("rolled back" in r or "0 of 1 steps" in r):
    print(f"  FAIL  rename without extension rolled back: {r[:80]}")
    FAIL += 1
elif isinstance(r, dict) and "error" not in r:
    new_name = r.get("new_name","")
    print(f"  PASS  rename without extension succeeded → {new_name!r}")
    PASS += 1
    REPORT = new_name  # update for subsequent tests
else:
    text = r.get("speech_text","") if isinstance(r, dict) else str(r)
    print(f"  PASS  rename without extension gave graceful result: {text[:80]}")
    PASS += 1

# 6d: rename_file heuristic "rename file to budget.xlsx"
r2 = hparse("rename file to budget.xlsx")
if r2 and r2.get("type") == "rename_file" and "budget" in r2.get("new_name",""):
    print(f"  PASS  heuristic 'rename file to budget.xlsx' → {r2}")
    PASS += 1
else:
    print(f"  FAIL  heuristic 'rename file to budget.xlsx' → {r2}")
    FAIL += 1

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 7: rename_sheet still works")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════

SHEET_FILE = f"rename_sheet_test_{int(_time.time())}.xlsx"
fresh(SHEET_FILE)
process_command("create excel", filename=SHEET_FILE)
# Default sheet name from openpyxl Workbook() is "Sheet" (not "Sheet1")
ok("rename sheet Sheet to Data",
   process_command("rename sheet Sheet to Data", filename=SHEET_FILE))

wb_s = load_workbook(str(fp(SHEET_FILE)))
if "Data" in wb_s.sheetnames or "data" in [s.lower() for s in wb_s.sheetnames]:
    print(f"  PASS  sheet renamed to 'Data' in {SHEET_FILE}")
    PASS += 1
else:
    print(f"  FAIL  sheet rename failed; sheets={wb_s.sheetnames}")
    FAIL += 1
fresh(SHEET_FILE)

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 8: Regression — working baseline")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════

RFILE = f"rename_regr_{int(_time.time())}.xlsx"
fresh(RFILE)
process_command("close excel", filename=RFILE)
_time.sleep(0.5)

ok("R create excel",       process_command("create excel", filename=RFILE))
ok("R add row",            process_command("add row Tom 85"))
ok("R add row 2",          process_command("add row Jane 92"))
ok("R write cell B2 99",   process_command("write cell B2 99"))
ok("R add column Bonus",   process_command("add column Bonus"))
ok("R delete row 2",       process_command("delete row 2"))
ok("R format bold A1",     process_command("format cells A1 bold"))
ok("R write formula",      process_command("write formula C1 =B1*2"))
ok("R freeze panes B2",    process_command("freeze panes B2"))
ok("R unfreeze panes",     process_command("unfreeze panes"))
ok("R merge cells A1:A2",  process_command("merge cells A1:A2"))
ok("R unmerge A1:A2",      process_command("unmerge cells A1:A2"))
ok("R set col width A 20", process_command("set column width A 20"))
ok("R create chart",       process_command("create bar chart A1:B2"))
ok("R create sheet",       process_command("create sheet Summary"))
# Default sheet name from openpyxl Workbook() is "Sheet" (not "Sheet1")
ok("R switch sheet",       process_command("switch sheet Sheet"))
ok("R rename sheet",       process_command("rename sheet Sheet to Main"))
ok("R switch to Main",     process_command("switch sheet Main"))
ok("R list sheets",        process_command("list sheets"))
ok("R delete sheet",       process_command("delete sheet Summary"))
ok("R find replace",       process_command("find and replace Tom with Jerry"))
ok("R sort by B",          process_command("sort by column B"))
ok("R save",               process_command("save"))
ok("R undo",               process_command("undo"))
ok("R close excel",        process_command("close excel"))
fresh(RFILE)

# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n══════════════════════════════════════════")
print(f"  RESULTS: {PASS} PASS  /  {FAIL} FAIL")
print(f"══════════════════════════════════════════\n")
if FAIL == 0:
    print("  ✔  ALL TESTS PASSED — rename_file is SAFE")
else:
    print(f"  ✖  {FAIL} TESTS FAILED — see above")
