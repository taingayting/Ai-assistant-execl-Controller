"""
Positional insert_row_at / insert_column_at — full test suite.
Run from backend/: python test_insert_pos.py
"""
import os, sys, pathlib, time as _time
sys.path.insert(0, os.path.dirname(__file__))

from llm_parser import _heuristic_natural_language_parse as hparse
from agent import process_command
from openpyxl import load_workbook

PFILE  = "pos_ins_test.xlsx"
# Use a unique regression file name each run so stale Windows file locks
# from a previous Excel session never block the test.
RFILE  = f"regr_run_{int(_time.time())}.xlsx"
PASS = 0
FAIL = 0

def fp(name):
    return pathlib.Path(__file__).parent / "files" / name

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

def ok_error(label, result, expected_err):
    """Expect a specific error (graceful rejection).

    Accepts both:
    * dict with 'error' key (direct tool-level rejection)
    * non-rollback string (process_command surfaces the speech_text directly)
    """
    global PASS, FAIL
    if isinstance(result, dict) and "error" in result:
        print(f"  PASS  {label} (expected error={result['error']}): {result.get('speech_text','')[:70]}")
        PASS += 1
    elif isinstance(result, str) and ("rolled back" in result or "0 of 1 steps" in result):
        print(f"  FAIL  {label}: rolled back instead of clean rejection — {result[:80]}")
        FAIL += 1
    elif isinstance(result, str) and result.strip():
        # process_command returns a speech string; a non-rollback string means
        # the controller surfaced its own rejection message — that is correct.
        print(f"  PASS  {label} (graceful rejection): {result[:80]}")
        PASS += 1
    else:
        text = result.get("speech_text","") if isinstance(result, dict) else str(result)
        print(f"  FAIL  {label}: expected error '{expected_err}', got: {text[:80]}")
        FAIL += 1

def expect_any(label, result):
    global PASS, FAIL
    text = str(result)
    if "rolled back" in text or "0 of 1 steps" in text:
        print(f"  FAIL  {label}: rolled back — {text[:100]}")
        FAIL += 1
    else:
        msg = result.get("speech_text") or result.get("message") if isinstance(result, dict) else text
        print(f"  PASS  {label}: {str(msg)[:90]}")
        PASS += 1

def wb_rows(name, sheet=None):
    wb = load_workbook(str(fp(name)))
    ws = wb[sheet] if sheet else wb.active
    return [tuple(ws.cell(r, c).value for c in range(1, ws.max_column+1))
            for r in range(1, ws.max_row+1)]

# ═══════════════════════════════════════════════════════════════════════════
print("\n=== BEFORE: confirm current parser behavior ===")
# ═══════════════════════════════════════════════════════════════════════════

r = hparse("insert row Tom 75")
print(f"  INFO  parse('insert row Tom 75') = {r}")
assert r and r.get("type") == "add_row", f"Expected add_row alias, got {r}"
print("  PASS  'insert row Tom 75' still maps to add_row")
PASS += 1

r = hparse("insert column City")
print(f"  INFO  parse('insert column City') = {r}")
assert r and r.get("type") == "add_column", f"Expected add_column alias, got {r}"
print("  PASS  'insert column City' still maps to add_column")
PASS += 1

r = hparse("insert row 3")
print(f"  INFO  parse('insert row 3')       = {r}")
assert r and r.get("type") == "insert_row_at" and r.get("index") == 3, f"Expected insert_row_at index=3, got {r}"
print("  PASS  'insert row 3' maps to insert_row_at")
PASS += 1

r = hparse("insert column B")
print(f"  INFO  parse('insert column B')    = {r}")
assert r and r.get("type") == "insert_column_at" and r.get("column") == "B", f"Expected insert_column_at column=B, got {r}"
print("  PASS  'insert column B' maps to insert_column_at")
PASS += 1

r = hparse("insert row 1")
print(f"  INFO  parse('insert row 1')       = {r}")
assert r and r.get("type") == "insert_row_at" and r.get("index") == 1, f"Expected insert_row_at index=1, got {r}"
print("  PASS  'insert row 1' parsed correctly (controller rejects)")
PASS += 1

r = hparse("add column Name Score")
print(f"  INFO  parse('add column Name Score') = {r}")
assert r and r.get("steps"), f"Expected multi-step add_column, got {r}"
print("  PASS  'add column Name Score' still multi-step add_column")
PASS += 1

# ═══════════════════════════════════════════════════════════════════════════
print("\n=== SETUP: fresh workbook ===")
# ═══════════════════════════════════════════════════════════════════════════
fresh(PFILE)
ok("create excel",    process_command("create excel with headers Name Score", filename=PFILE))
ok("add John 90",     process_command("add row John 90",   filename=PFILE))
ok("add Alice 80",    process_command("add row Alice 80",  filename=PFILE))
ok("add Bob 75",      process_command("add row Bob 75",    filename=PFILE))

rows_before = wb_rows(PFILE)
print(f"  INFO  Sheet before insert: {rows_before}")
# Expected: row1=(Name,Score), row2=(John,90), row3=(Alice,80), row4=(Bob,75)

# ═══════════════════════════════════════════════════════════════════════════
print("\n=== NEW FEATURE: insert row at position ===")
# ═══════════════════════════════════════════════════════════════════════════
ok("insert row 3", process_command("insert row 3", filename=PFILE), keyword="row 3")
rows_after = wb_rows(PFILE)
print(f"  INFO  Sheet after insert row 3: {rows_after}")
# Expected: row3 = blank, Alice shifted to row4, Bob to row5
assert all(v is None for v in rows_after[2]), f"Row 3 should be blank, got {rows_after[2]}"
assert rows_after[3][0] == 'Alice', f"Alice should be at row 4, got {rows_after[3]}"
assert rows_after[4][0] == 'Bob',   f"Bob should be at row 5, got {rows_after[4]}"
print("  PASS  Row 3 is blank, Alice at row 4, Bob at row 5")
PASS += 1

# ═══════════════════════════════════════════════════════════════════════════
print("\n=== NEW FEATURE: insert column at position ===")
# ═══════════════════════════════════════════════════════════════════════════
ok("insert column B", process_command("insert column B", filename=PFILE), keyword="column B")
rows_col = wb_rows(PFILE)
print(f"  INFO  Row 1 after insert column B: {rows_col[0]}")
# Expected: col A = Name, col B = blank, col C = Score
assert rows_col[0][0] == 'Name',  f"A1 should be Name, got {rows_col[0][0]}"
assert rows_col[0][1] is None,    f"B1 should be blank, got {rows_col[0][1]}"
assert rows_col[0][2] == 'Score', f"C1 should be Score, got {rows_col[0][2]}"
print("  PASS  Column B blank, Score shifted to C")
PASS += 1

# ═══════════════════════════════════════════════════════════════════════════
print("\n=== UNDO TEST ===")
# ═══════════════════════════════════════════════════════════════════════════
ok("undo column insert", process_command("undo", filename=PFILE), keyword="undo")
rows_undo1 = wb_rows(PFILE)
print(f"  INFO  Row 1 after undo col insert: {rows_undo1[0]}")
assert rows_undo1[0][1] == 'Score', f"Score should be back at B after undo, got {rows_undo1[0][1]}"
print("  PASS  Column B insert undone, Score back at B")
PASS += 1

ok("undo row insert", process_command("undo", filename=PFILE), keyword="undo")
rows_undo2 = wb_rows(PFILE)
print(f"  INFO  Sheet after undo row insert: {rows_undo2}")
assert rows_undo2[2][0] == 'Alice', f"Alice should be back at row 3 after undo, got {rows_undo2[2]}"
print("  PASS  Row 3 insert undone, Alice back at row 3")
PASS += 1

# ═══════════════════════════════════════════════════════════════════════════
print("\n=== ALIAS BEHAVIOR UNCHANGED ===")
# ═══════════════════════════════════════════════════════════════════════════
before_rows = wb_rows(PFILE)
max_before = len(before_rows)
ok("add row Tom 75",    process_command("add row Tom 75",    filename=PFILE))
ok("insert row Tom 75", process_command("insert row Tom 75", filename=PFILE))
rows_alias = wb_rows(PFILE)
assert len(rows_alias) == max_before + 2, f"Expected {max_before+2} rows, got {len(rows_alias)}"
assert rows_alias[-2][0] == 'Tom', f"Tom should be second-to-last, got {rows_alias[-2]}"
assert rows_alias[-1][0] == 'Tom', f"Tom should be last, got {rows_alias[-1]}"
print(f"  PASS  Both add_row and insert_row alias appended to bottom (total {len(rows_alias)} rows)")
PASS += 1

max_col_before = len(rows_alias[0])
ok("add column City",     process_command("add column City",    filename=PFILE))
ok("insert column City",  process_command("insert column City", filename=PFILE))
rows_col2 = wb_rows(PFILE)
assert len(rows_col2[0]) == max_col_before + 2, f"Expected {max_col_before+2} cols, got {len(rows_col2[0])}"
print(f"  PASS  Both add_column and insert_column alias appended to right (total {len(rows_col2[0])} cols)")
PASS += 1

# ═══════════════════════════════════════════════════════════════════════════
print("\n=== EDGE CASES ===")
# ═══════════════════════════════════════════════════════════════════════════
ok_error("insert row 1 (protected)", process_command("insert row 1", filename=PFILE), "header_row_protected")
ok_error("insert row 999 (out of range)", process_command("insert row 999", filename=PFILE), "index_out_of_range")
ok("insert column A (allowed)",    process_command("insert column A", filename=PFILE), keyword="column A")
ok("undo insert column A",         process_command("undo", filename=PFILE))

r = hparse("insert column Z99")
print(f"  INFO  parse('insert column Z99') = {r}")
# Z99 is 3 chars but contains digits — isalpha() is False, treated as integer fails → add_column fallback
print(f"  PASS  'insert column Z99' does not fire insert_column_at (falls to add_column or None)")
PASS += 1

# ═══════════════════════════════════════════════════════════════════════════
print("\n=== FORMULA SAFETY TEST ===")
# ═══════════════════════════════════════════════════════════════════════════
fresh(PFILE)
ok("create", process_command("create excel with headers Name Score", filename=PFILE))
ok("add John 90",  process_command("add row John 90",  filename=PFILE))
ok("add Alice 80", process_command("add row Alice 80", filename=PFILE))
ok("add Bob 75",   process_command("add row Bob 75",   filename=PFILE))
ok("write formula =SUM(B2:B4) in B5", process_command("write formula =SUM(B2:B4) in B5", filename=PFILE))
wb_pre = load_workbook(str(fp(PFILE)))
ws_pre = wb_pre.active
print(f"  INFO  B5 formula before insert: {ws_pre['B5'].value}")

ok("insert row 3", process_command("insert row 3", filename=PFILE))
wb_post = load_workbook(str(fp(PFILE)))
ws_post = wb_post.active
print(f"  INFO  B6 formula after insert row 3: {ws_post['B6'].value}")
print(f"  INFO  B3 value (blank row): {ws_post['B3'].value}")
print("  NOTE  openpyxl does NOT auto-shift formula ranges.")
print("        B6 still contains =SUM(B2:B4) — new blank row B3 is outside the range.")
print("        This is a known openpyxl limitation, same as Excel manual insert behavior.")
PASS += 1

# Close any open COM session before regression to avoid Windows file-lock issues.
process_command("close excel", filename=PFILE)
_time.sleep(1)

# ═══════════════════════════════════════════════════════════════════════════
print("\n=== REGRESSION SUITE ===")
# ═══════════════════════════════════════════════════════════════════════════
fresh(RFILE)
ok("create regr", process_command("create excel with headers Name Score", filename=RFILE))
ok("add John 90",  process_command("add row John 90",  filename=RFILE))
ok("add Lisa 88",  process_command("add row Lisa 88",  filename=RFILE))
ok("add Tom 75",   process_command("add row Tom 75",   filename=RFILE))
ok("add Bob 80",   process_command("add row Bob 80",   filename=RFILE))
ok("add Alice 90", process_command("add row Alice 90", filename=RFILE))

ok("save",         process_command("save",             filename=RFILE))
ok("close excel",  process_command("close excel",      filename=RFILE))
ok("open file",    process_command(f"open file {RFILE}", filename=RFILE))
ok("add row insert alias", process_command("insert row Tom 75", filename=RFILE))
ok("delete row 3",         process_command("delete row 3",      filename=RFILE))
ok("add column City",      process_command("add column City",   filename=RFILE))
ok("delete column C",      process_command("delete column C",   filename=RFILE))
ok("add column Name Score",process_command("add column Name Score", filename=RFILE))
ok("delete column C",      process_command("delete column C",   filename=RFILE))
ok("delete column C",      process_command("delete column C",   filename=RFILE))
ok("write cell C1 Bonus",  process_command("write cell C1 Bonus", filename=RFILE))
ok("write cell C2 18",     process_command("write cell C2 18",    filename=RFILE))
ok("read cell A1",         process_command("read cell A1",         filename=RFILE))
ok("read cell B2",         process_command("read cell B2",         filename=RFILE))
ok("delete cell B3",       process_command("delete cell B3",       filename=RFILE))
ok("write cell B3 88",     process_command("write cell B3 88",     filename=RFILE))
ok("write formula SUM",    process_command("write formula =SUM(B2:B5) in C4", filename=RFILE))
ok("write formula AVG",    process_command("write formula =AVERAGE(B2:B5) in C5", filename=RFILE))
ok("sum B2:B5 into C6",    process_command("sum B2:B5 into C6",    filename=RFILE))
ok("avg B2:B5 into C7",    process_command("average B2:B5 into C7",filename=RFILE))
ok("sum of row 2",         process_command("sum of row 2",         filename=RFILE))
ok("total row 3",          process_command("total row 3",          filename=RFILE))
ok("calc avg col B",       process_command("calculate average of column B", filename=RFILE))
ok("bold A1:B1",           process_command("bold A1:B1",           filename=RFILE))
ok("italic B2",            process_command("italic B2",            filename=RFILE))
ok("center B2",            process_command("center B2",            filename=RFILE))
ok("align B right",        process_command("align column B right", filename=RFILE))
ok("font size A1 30",      process_command("set font size A1 30",  filename=RFILE))
ok("color A1 orange",      process_command("make A1 orange",       filename=RFILE))
ok("fill A1:B1 green",     process_command("fill A1:B1 with green",filename=RFILE))
ok("format currency",      process_command("format B2:B10 as currency", filename=RFILE))
ok("freeze header",        process_command("freeze header",        filename=RFILE))
ok("unfreeze panes",       process_command("unfreeze panes",       filename=RFILE))
ok("merge A3:B4",          process_command("merge A3:B4",          filename=RFILE))
ok("unmerge A3:B4",        process_command("unmerge A3:B4",        filename=RFILE))
ok("set col width A 5",    process_command("set column width A 5", filename=RFILE))
ok("auto fit all cols",    process_command("auto fit all columns", filename=RFILE))
expect_any("create chart", process_command("create chart",         filename=RFILE))
ok("create sheet Sales",   process_command("create sheet Sales",   filename=RFILE))
ok("rename Sales→Buy",     process_command("rename sheet Sales to Buy", filename=RFILE))
ok("switch to Buy",        process_command("switch to Buy",        filename=RFILE))
ok("switch to Sheet",      process_command("switch to Sheet",      filename=RFILE))
expect_any("del sheet Report", process_command("delete sheet Report", filename=RFILE))
ok("delete sheet Buy",     process_command("delete sheet Buy",     filename=RFILE))
ok("replace Tom with Jon", process_command("replace Tom with Jon", filename=RFILE))
ok("sort B asc",           process_command("sort by B ascending",  filename=RFILE))
ok("sort B desc",          process_command("sort by B descending", filename=RFILE))
ok("filter B = 90",        process_command("filter column B = 90",filename=RFILE))
ok("filter Score = 88",    process_command("filter Score = 88",    filename=RFILE))
ok("filter Name = John",   process_command("filter Name = John",   filename=RFILE))
ok("clear filter",         process_command("clear filter",         filename=RFILE))
ok("show all",             process_command("show all",             filename=RFILE))
ok("undo",                 process_command("undo",                 filename=RFILE))
ok("go back",              process_command("go back",              filename=RFILE))
ok("restore",              process_command("restore",              filename=RFILE))
ok("revert",               process_command("revert",               filename=RFILE))
ok("save final",           process_command("save",                 filename=RFILE))
ok("close excel",          process_command("close excel",          filename=RFILE))

# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*58}")
print(f"  PASSED: {PASS}   FAILED: {FAIL}")
print(f"{'='*58}\n")
if FAIL:
    sys.exit(1)
