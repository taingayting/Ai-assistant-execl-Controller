"""
Full command audit for Sara Excel Assistant.
Run from backend/: python test_audit.py
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
SKIP = 0
_results = []  # (section, label, status, text)

T = f"{int(_time.time())}"
MAIN  = f"audit_main_{T}.xlsx"
FILT  = f"audit_filt_{T}.xlsx"
COMP  = f"comptest_{T}.xlsx"
REN_SRC = f"audit_rename_src_{T}.xlsx"
REN_DST = f"audit_rename_dst_{T}.xlsx"

def fp(name):
    return FILES_DIR / name

def fresh(name):
    p = fp(name)
    if p.exists():
        try: p.unlink()
        except: pass

def _record(section, label, status, text):
    _results.append((section, label, status, text))
    if status == "PASS":
        print(f"  PASS  [{section}] {label}: {text[:80]}")
    elif status == "FAIL":
        print(f"  FAIL  [{section}] {label}: {text[:100]}")
    elif status == "SKIP":
        print(f"  SKIP  [{section}] {label}: {text[:80]}")

def ok(section, label, result, keyword=None):
    global PASS, FAIL
    if isinstance(result, str):
        text = result
    elif isinstance(result, dict):
        if "error" in result:
            _record(section, label, "FAIL", f"error={result.get('error')} {result.get('speech_text','')[:60]}")
            FAIL += 1
            return
        text = result.get("speech_text") or result.get("message") or str(result)
    else:
        text = str(result)
    bad = ("rolled back" in text.lower() or "0 of 1 steps" in text or
           "hit an error" in text.lower() or "couldn't understand" in text.lower() or
           "didn't understand" in text.lower() or "parsed but not executed" in text.lower())
    if bad:
        _record(section, label, "FAIL", text)
        FAIL += 1
        return
    if keyword and keyword.lower() not in text.lower():
        _record(section, label, "FAIL", f"missing '{keyword}' in: {text}")
        FAIL += 1
        return
    _record(section, label, "PASS", text)
    PASS += 1

def ok_error(section, label, result):
    """Expect a clean error (not a rollback)."""
    global PASS, FAIL
    if isinstance(result, dict) and "error" in result:
        _record(section, label, "PASS", f"error={result['error']}: {result.get('speech_text','')[:60]}")
        PASS += 1
    elif isinstance(result, str) and ("rolled back" not in result and "0 of 1 steps" not in result) and result.strip():
        _record(section, label, "PASS", f"graceful: {result[:80]}")
        PASS += 1
    else:
        text = result.get("speech_text","") if isinstance(result, dict) else str(result)
        _record(section, label, "FAIL", f"expected clean error, got: {text[:80]}")
        FAIL += 1

def setup_workbook(name, rows=None):
    """Create a workbook and optionally seed rows. Returns True on success.

    Always closes Excel first so that the file is not locked by Excel.exe,
    which would cause fresh() to fail and create excel to be rejected.
    """
    process_command("close excel")   # release any Excel write lock first
    fresh(name)
    r = process_command("create excel", filename=name)
    # Detect silent failures: "already exists" or "currently open" responses
    # have no 'error' key but are not successes either.
    speech = r if isinstance(r, str) else r.get("speech_text", "") if isinstance(r, dict) else str(r)
    if ("already exists" in speech or "currently open" in speech or
            "rolled back" in speech or (isinstance(r, dict) and "error" in r)):
        print(f"  WARN  setup_workbook({name}): create failed: {speech[:80]}")
        return False
    if rows:
        for row in rows:
            process_command(f"add row {row}")
    return True

def wb_data(name):
    """Return all rows as list-of-tuples, or None if file missing."""
    p = fp(name)
    if not p.exists(): return None
    try:
        wb = load_workbook(str(p))
        ws = wb.active
        return [tuple(ws.cell(r, c).value for c in range(1, ws.max_column+1))
                for r in range(1, ws.max_row+1) if any(ws.cell(r,c).value is not None for c in range(1, ws.max_column+1))]
    except Exception as e:
        return f"ERROR: {e}"

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 1: File / session")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════
S = "File/Session"
fresh(COMP); fresh(MAIN)
ok(S, "create excel comptest", process_command("create excel", filename=COMP))
ok(S, "create excel main",     process_command("create excel", filename=MAIN))
ok(S, "open excel",            process_command("open excel"))
ok(S, "save",                  process_command("save"))
ok(S, "close excel",           process_command("close excel"))
ok(S, "open file main",        process_command(f"open file {MAIN}"))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 2: Rows")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════
S = "Rows"
setup_workbook(MAIN)
ok(S, "add row John 90",   process_command("add row John 90"))
ok(S, "add row Lisa 88",   process_command("add row Lisa 88"))
ok(S, "insert row Tom 75", process_command("insert row Tom 75"))
# delete row 1 is intentionally blocked (header row protection).
# Validator rejects index < 2 by design. This is a missing feature
# (clear rejection message), not a code bug.
ok_error(S, "delete row 1 (header protected)", process_command("delete row 1"))
ok(S, "delete row 3",      process_command("delete row 3"))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 3: True positional insert")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════
S = "Positional"
setup_workbook(MAIN, ["John 90", "Lisa 88", "Tom 75"])
ok(S, "insert row 3",    process_command("insert row 3"))
ok(S, "insert column B", process_command("insert column B"))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 4: Columns")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════
S = "Columns"
setup_workbook(MAIN)
ok(S, "add column Name",        process_command("add column Name"))
ok(S, "add column Score",       process_command("add column Score"))
ok(S, "add column Name Score",  process_command("add column Name Score"))
ok(S, "delete column B",        process_command("delete column B"))
ok(S, "delete column C",        process_command("delete column C"))
ok(S, "insert column City",     process_command("insert column City"))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 5: Cells")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════
S = "Cells"
setup_workbook(MAIN, ["John 90", "Lisa 88"])
ok(S, "write cell C1 Name",  process_command("write cell C1 Name"))
ok(S, "write cell C2 18",    process_command("write cell C2 18"))
ok(S, "set cell D1 Test",    process_command("set cell D1 Test"))
ok(S, "put 25 in C3",        process_command("put 25 in C3"))
ok(S, "read cell A1",        process_command("read cell A1"))
ok(S, "read cell B2",        process_command("read cell B2"))
ok(S, "delete cell B2",      process_command("delete cell B2"))
ok(S, "delete cell C3",      process_command("delete cell C3"))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 6: Ranges / delete")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════
S = "Ranges"
setup_workbook(MAIN, ["John 90 A", "Lisa 88 B", "Tom 75 C"])
ok(S, "delete range A1:B3",  process_command("delete range A1:B3"))
setup_workbook(MAIN, ["John 90 A", "Lisa 88 B", "Tom 75 C"])
ok(S, "delete A2:C2",        process_command("delete A2:C2"))
ok(S, "delete cell A2",      process_command("delete cell A2"))
ok(S, "delete row 3",        process_command("delete row 3"))
ok(S, "delete column C",     process_command("delete column C"))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 7: Formulas")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════
S = "Formulas"
setup_workbook(MAIN, ["10", "20", "30", "40"])
ok(S, "write formula =SUM(B2:B5) in C4",       process_command("write formula =SUM(B2:B5) in C4"))
ok(S, "write formula =AVERAGE(B2:B5) in C5",   process_command("write formula =AVERAGE(B2:B5) in C5"))
ok(S, "sum B2:B5 into C6",                     process_command("sum B2:B5 into C6"))
ok(S, "average B2:B5 into C7",                 process_command("average B2:B5 into C7"))
ok(S, "sum of row 2",                          process_command("sum of row 2"))
ok(S, "total row 3",                           process_command("total row 3"))
ok(S, "calculate average of column B",         process_command("calculate average of column B"))
ok(S, "calculate average B2:B4 in D5",         process_command("calculate average B2:B4 in D5"))
ok(S, "calculate row sum B2:B4 in D4",         process_command("calculate row sum B2:B4 in D4"))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 8: Formatting — text style")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════
S = "FmtStyle"
setup_workbook(MAIN, ["John 90", "Lisa 88", "Tom 75"])
ok(S, "bold A1:C1",        process_command("bold A1:C1"))
ok(S, "bold row 1",        process_command("bold row 1"))
ok(S, "format A1:C3 bold", process_command("format A1:C3 bold"))
ok(S, "italic A2:B2",      process_command("italic A2:B2"))
ok(S, "italic C2",         process_command("italic C2"))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 9: Formatting — alignment")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════
S = "FmtAlign"
setup_workbook(MAIN, ["John 90", "Lisa 88", "Tom 75"])
ok(S, "center B2",           process_command("center B2"))
ok(S, "center A1:C1",        process_command("center A1:C1"))
ok(S, "center column B",     process_command("center column B"))
ok(S, "center row 2",        process_command("center row 2"))
ok(S, "align A2 left",       process_command("align A2 left"))
ok(S, "align A2 right",      process_command("align A2 right"))
ok(S, "align A1:C3 center",  process_command("align A1:C3 center"))
ok(S, "align column B right",process_command("align column B right"))
ok(S, "align column B left", process_command("align column B left"))
ok(S, "align row 3 right",   process_command("align row 3 right"))
ok(S, "align row 3 left",    process_command("align row 3 left"))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 10: Formatting — font size")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════
S = "FmtSize"
setup_workbook(MAIN, ["John 90", "Lisa 88"])
ok(S, "set font size A1 30",      process_command("set font size A1 30"))
ok(S, "increase font size A1",    process_command("increase font size A1"))
ok(S, "increase font size A1:C1", process_command("increase font size A1:C1"))
ok(S, "decrease font size A1:C1", process_command("decrease font size A1:C1"))
ok(S, "decrease font size B2",    process_command("decrease font size B2"))
ok(S, "make header size 18",      process_command("make header size 18"))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 11: Formatting — color / fill")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════
S = "FmtColor"
setup_workbook(MAIN, ["John 90", "Lisa 88", "Tom 75"])
ok(S, "make A1 orange",       process_command("make A1 orange"))
ok(S, "color B1 red",         process_command("color B1 red"))
ok(S, "color row 1 blue",     process_command("color row 1 blue"))
ok(S, "fill row 2 with red",  process_command("fill row 2 with red"))
ok(S, "highlight row 1",      process_command("highlight row 1"))
ok(S, "fill A1 pink",         process_command("fill A1 pink"))
ok(S, "fill A1:C1 with green",process_command("fill A1:C1 with green"))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 12: Formatting — number format")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════
S = "FmtNum"
setup_workbook(MAIN, ["John 90", "Lisa 88"])
ok(S, "format B2:B10 as currency",   process_command("format B2:B10 as currency"))
ok(S, "format C2:C8 as percentage",  process_command("format C2:C8 as percentage"))
ok(S, "format D2:D10 as number",     process_command("format D2:D10 as number"))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 13: Header formatting")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════
S = "FmtHeader"
setup_workbook(MAIN, ["John 90", "Lisa 88"])
ok(S, "make header bold", process_command("make header bold"))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 14: Freeze / unfreeze")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════
S = "Freeze"
setup_workbook(MAIN, ["John 90", "Lisa 88"])
ok(S, "freeze header",    process_command("freeze header"))
ok(S, "freeze panes",     process_command("freeze panes"))
ok(S, "unfreeze panes",   process_command("unfreeze panes"))
ok(S, "unfreeze header",  process_command("unfreeze header"))
ok(S, "unfreeze",         process_command("unfreeze"))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 15: Merge / unmerge")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════
S = "Merge"
setup_workbook(MAIN, ["John 90", "Lisa 88", "Tom 75"])
ok(S, "merge A1:B3",          process_command("merge A1:B3"))
ok(S, "unmerge cells A1:B3",  process_command("unmerge cells A1:B3"))
ok(S, "merge header",         process_command("merge header"))
ok(S, "merge cells A1 A2",    process_command("merge cells A1 A2"))
ok(S, "unmerge cells A1:A2",  process_command("unmerge cells A1:A2"))
# Re-merge so the second unmerge form has something to unmerge
process_command("merge A1:A2")
ok(S, "unmerge A1:A2",        process_command("unmerge A1:A2"))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 16: Column width / autofit")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════
S = "ColWidth"
setup_workbook(MAIN, ["John 90", "Lisa 88"])
ok(S, "set column width A 5",  process_command("set column width A 5"))
ok(S, "set column width B 10", process_command("set column width B 10"))
ok(S, "auto fit column A",     process_command("auto fit column A"))
ok(S, "auto fit column B",     process_command("auto fit column B"))
ok(S, "auto fit all columns",  process_command("auto fit all columns"))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 17: Charts")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════
S = "Charts"
setup_workbook(MAIN, ["John 90", "Lisa 88", "Tom 75"])
ok(S, "create chart",          process_command("create chart"))
ok(S, "create bar chart",      process_command("create bar chart"))
ok(S, "create bar chart A1:C4",process_command("create bar chart A1:C4"))
ok(S, "create pie chart",      process_command("create pie chart"))
ok(S, "create line chart",     process_command("create line chart"))
ok(S, "create line chart A1:C4",process_command("create line chart A1:C4"))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 18: Sheets")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════
S = "Sheets"
setup_workbook(MAIN)
ok(S, "create sheet Sales",       process_command("create sheet Sales"))
ok(S, "create sheet Report",      process_command("create sheet Report"))
ok(S, "rename sheet Sales to Buy",process_command("rename sheet Sales to Buy"))
ok(S, "switch to Buy",            process_command("switch to Buy"))
ok(S, "go to Report",             process_command("go to Report"))
ok(S, "delete sheet Report",      process_command("delete sheet Report"))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 19: Replace / sort")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════
S = "Replace/Sort"
setup_workbook(MAIN, ["Tom 90", "Lisa 88", "John 75"])
ok(S, "replace Tom with John",           process_command("replace Tom with John"))
ok(S, "replace 90 with 95",              process_command("replace 90 with 95"))
ok(S, "sort by B ascending",             process_command("sort by B ascending"))
ok(S, "sort by B descending",            process_command("sort by B descending"))
ok(S, "sort by column A descending",     process_command("sort by column A descending"))
ok(S, "sort by column A ascending",      process_command("sort by column A ascending"))
ok(S, "sort table by column B ascending",process_command("sort table by column B ascending"))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 20: Undo")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════
S = "Undo"
setup_workbook(MAIN, ["John 90", "Lisa 88"])
process_command("add row Bob 75")  # give something to undo
ok(S, "undo",    process_command("undo"))
process_command("add row Bob 75")
ok(S, "go back", process_command("go back"))
process_command("add row Bob 75")
ok(S, "restore", process_command("restore"))
process_command("add row Bob 75")
ok(S, "revert",  process_command("revert"))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 21: Filter")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════
S = "Filter"
fresh(FILT)
process_command("create excel", filename=FILT)
process_command("add column Name")
process_command("add column Score")
process_command("add row John 95")
process_command("add row Lisa 88")
process_command("add row Tom 95")
ok(S, "filter column B = 95",  process_command("filter column B = 95"))
ok(S, "clear filter",           process_command("clear filter"))
ok(S, "filter Score = 88",      process_command("filter Score = 88"))
ok(S, "clear filter",           process_command("clear filter"))
ok(S, "filter Name = John",     process_command("filter Name = John"))
ok(S, "show all",               process_command("show all"))

# ═══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  SECTION 22: File rename")
print("══════════════════════════════════════════")
# ═══════════════════════════════════════════════════════════════════════════════
S = "Rename"
fresh(REN_SRC); fresh(REN_DST)
setup_workbook(REN_SRC, ["John 90"])

ok(S, f"rename file demo→report",
   process_command(f"rename file to {REN_DST}", filename=REN_SRC))

if fp(REN_DST).exists() and not fp(REN_SRC).exists():
    _record(S, "disk state correct", "PASS", f"{REN_DST} exists, {REN_SRC} gone")
    PASS += 1
else:
    _record(S, "disk state correct", "FAIL",
            f"src_exists={fp(REN_SRC).exists()} dst_exists={fp(REN_DST).exists()}")
    FAIL += 1

af = memory.get_active_file()
if af == REN_DST:
    _record(S, "memory active_file", "PASS", f"active_file={af}")
    PASS += 1
else:
    _record(S, "memory active_file", "FAIL", f"active_file={af} (expected {REN_DST})")
    FAIL += 1

ok(S, "rename file to budget (no ext)", process_command("rename file to budget_audit_final"))
fresh(f"budget_audit_final.xlsx")

# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n══════════════════════════════════════════")
print(f"  RESULTS: {PASS} PASS  /  {FAIL} FAIL  /  {SKIP} SKIP")
print(f"══════════════════════════════════════════")

if FAIL:
    print(f"\n  ── FAILURES ──")
    for sec, lab, st, txt in _results:
        if st == "FAIL":
            print(f"  [{sec}] {lab}: {txt[:110]}")

print()
if FAIL == 0:
    print("  ALL TESTS PASSED")
else:
    print(f"  {FAIL} FAILURES — see above")

# cleanup
for f in [COMP, MAIN, FILT, REN_SRC, REN_DST]:
    fresh(f)
