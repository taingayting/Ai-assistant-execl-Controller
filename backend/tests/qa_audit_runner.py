"""
Sara Non-Voice Command Audit
============================
QA TEST-ONLY script.  Reads code; runs process_command(); verifies workbook
state with openpyxl.  NEVER modifies any project source file.

Run from the project root:
    python backend/tests/qa_audit_runner.py

All test files created under  backend/files/  and named  qa_audit_*.xlsx
They are deleted at the end of the run unless --keep is passed.
"""
from __future__ import annotations

import os, sys, re, time, shutil
from dataclasses import dataclass, field
from typing import Callable

# ── path bootstrap ────────────────────────────────────────────────────────────
_TESTS   = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_TESTS)
for _p in (_BACKEND, os.path.dirname(_BACKEND)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from dotenv import load_dotenv
    _env = os.path.join(_BACKEND, '.env')
    if os.path.exists(_env):
        load_dotenv(_env, override=True)
except ImportError:
    pass

from agent import process_command   # type: ignore
import openpyxl                     # type: ignore

# ── result record ─────────────────────────────────────────────────────────────
@dataclass
class R:
    category: str
    cmd_id:   str
    phrase:   str
    expected: str
    status:   str       # PASS | FAIL | SKIP
    actual:   str = ""
    note:     str = ""

_results: list[R] = []
_KEEP_FILES = "--keep" in sys.argv

# ── helpers ───────────────────────────────────────────────────────────────────
_FILES_DIR = os.path.join(_BACKEND, "files")
os.makedirs(_FILES_DIR, exist_ok=True)

def _path(name: str) -> str:
    return os.path.join(_FILES_DIR, name)

def _wb(name: str):
    return openpyxl.load_workbook(_path(name))

def _cmd(phrase: str, filename: str) -> str:
    return process_command(phrase, filename=filename)

def _ok(resp: str, *keywords) -> bool:
    low = resp.lower()
    return any(k.lower() in low for k in keywords)

def log(cat: str, cid: str, phrase: str, expected: str,
        status: str, actual: str = "", note: str = "") -> None:
    r = R(cat, cid, phrase, expected, status, actual, note)
    _results.append(r)
    icon = "✔" if status == "PASS" else ("⚠" if status == "SKIP" else "✘")
    print(f"  {icon} [{cid:<28}] {status}  {note or actual[:60]}")

def section(title: str) -> None:
    print(f"\n{'═'*68}")
    print(f"  {title}")
    print(f"{'═'*68}")

def cleanup(*names: str) -> None:
    if _KEEP_FILES:
        return
    for n in names:
        p = _path(n)
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

# ══════════════════════════════════════════════════════════════════════════════
#  CATEGORY 1 — FILE OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════
def test_file():
    section("1. FILE OPERATIONS")
    F = "qa_audit_file.xlsx"

    # 1-A  create_excel
    r = _cmd(f"create excel file {F}", F)
    ok_r = _ok(r, "creat", "exist", "open", "ready", F.replace(".xlsx",""))
    ok_f = os.path.exists(_path(F))
    log("File", "create_excel", f"create excel file {F}",
        "file created on disk",
        "PASS" if ok_f else "FAIL",
        r, "file present" if ok_f else "file NOT found")

    # 1-B  open_file
    r = _cmd(f"open file {F}", F)
    ok_r = _ok(r, "open", "ready", "load", "active", "using", "ok")
    log("File", "open_file", f"open file {F}",
        "response mentions open/ready",
        "PASS" if ok_r else "FAIL", r)

    # 1-C  save
    # Write something first so the file is dirty
    _cmd("write cell A1 SaveTest", F)
    r = _cmd("save", F)
    ok_r = _ok(r, "save", "saved", "done", "ok", "success")
    log("File", "save", "save",
        "response mentions saved/ok",
        "PASS" if ok_r else "FAIL", r)

    # 1-D  rename_file  (rename → verify → rename back)
    NEW = "qa_audit_file_renamed.xlsx"
    r = _cmd(f"rename file to {NEW}", F)
    ok_r = _ok(r, "renam", "done", "ok", "success")
    renamed_exists = os.path.exists(_path(NEW))
    original_gone  = not os.path.exists(_path(F))
    if renamed_exists and original_gone:
        # rename back so remaining tests can still use qa_audit_file.xlsx
        _cmd(f"rename file to {F}", NEW)
        note = "renamed ok, restored"
        status = "PASS"
    else:
        note = f"new={renamed_exists} old_gone={original_gone}"
        status = "FAIL"
    log("File", "rename_file", f"rename file to {NEW}",
        "new file exists, old gone", status, r, note)

    cleanup(F, NEW)

# ══════════════════════════════════════════════════════════════════════════════
#  CATEGORY 2 — ROW OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════
def test_rows():
    section("2. ROW OPERATIONS")
    F = "qa_audit_rows.xlsx"

    # Setup: create file with headers
    _cmd(f"create excel file {F}", F)
    _cmd("write cell A1 Name", F)
    _cmd("write cell B1 Score", F)

    # 2-A  add_row ×3
    for name, score in [("Alice", 85), ("Bob", 72), ("Charlie", 90)]:
        r = _cmd(f"add row {name} {score}", F)
        ok_r = _ok(r, "add", "row", "success", "done", "ok")
        log("Row", f"add_row({name})", f"add row {name} {score}",
            "row added", "PASS" if ok_r else "FAIL", r)

    # verify 3 data rows
    try:
        wb = _wb(F); ws = wb.active
        names = [ws.cell(i,1).value for i in range(2, ws.max_row+1)]
        wb.close()
        ok_v = set(names) == {"Alice","Bob","Charlie"}
        log("Row", "add_row[verify]", "openpyxl: rows 2-4",
            "Alice,Bob,Charlie", "PASS" if ok_v else "FAIL",
            str(names))
    except Exception as e:
        log("Row", "add_row[verify]", "", "", "FAIL", str(e))

    # 2-B  insert_row_at  (insert blank row at position 2 → Alice shifts to row 3)
    r = _cmd("insert row at position 2", F)
    ok_r = _ok(r, "insert", "row", "done", "ok", "success")
    try:
        wb = _wb(F); ws = wb.active
        row2_val = ws.cell(2,1).value
        wb.close()
        # after insert, row 2 should be blank (None), Alice at row 3
        ok_v = row2_val is None
        log("Row", "insert_row_at", "insert row at position 2",
            "row 2 is blank after insert",
            "PASS" if ok_v else "FAIL",
            f"row2={row2_val!r}")
    except Exception as e:
        log("Row", "insert_row_at[verify]", "", "", "FAIL", str(e))

    # 2-C  delete_row  (delete row 2, the blank one)
    r = _cmd("delete row 2", F)
    ok_r = _ok(r, "delet", "row", "done", "ok", "success", "remov")
    try:
        wb = _wb(F); ws = wb.active
        row2_val = ws.cell(2,1).value
        wb.close()
        ok_v = row2_val == "Alice"
        log("Row", "delete_row", "delete row 2",
            "Alice back at row 2 after deleting blank",
            "PASS" if ok_v else "FAIL",
            f"row2={row2_val!r}")
    except Exception as e:
        log("Row", "delete_row[verify]", "", "", "FAIL", str(e))

    cleanup(F)

# ══════════════════════════════════════════════════════════════════════════════
#  CATEGORY 3 — COLUMN OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════
def test_columns():
    section("3. COLUMN OPERATIONS")
    F = "qa_audit_cols.xlsx"
    _cmd(f"create excel file {F}", F)
    _cmd("write cell A1 Name", F)
    _cmd("write cell B1 Score", F)
    _cmd("add row Alice 85", F)
    _cmd("add row Bob 72", F)

    # 3-A  add_column
    r = _cmd("add column Grade", F)
    ok_r = _ok(r, "column", "add", "done", "ok", "success", "grade")
    try:
        wb = _wb(F); ws = wb.active
        hdrs = [ws.cell(1,c).value for c in range(1, ws.max_column+1)]
        wb.close()
        ok_v = "Grade" in hdrs
        log("Column", "add_column", "add column Grade",
            "Grade header in row 1",
            "PASS" if ok_v else "FAIL", str(hdrs))
    except Exception as e:
        log("Column", "add_column[verify]", "", "", "FAIL", str(e))

    # 3-B  insert_column_at  (insert blank column at C)
    r = _cmd("insert column C", F)
    ok_r = _ok(r, "insert", "column", "done", "ok", "success")
    try:
        wb = _wb(F); ws = wb.active
        col_c = ws.cell(1,3).value   # should now be blank
        wb.close()
        ok_v = col_c is None
        log("Column", "insert_column_at", "insert column C",
            "column C header is blank after insert",
            "PASS" if ok_v else "FAIL", f"C1={col_c!r}")
    except Exception as e:
        log("Column", "insert_column_at[verify]", "", "", "FAIL", str(e))

    # 3-C  delete_column  (delete the blank column C)
    r = _cmd("delete column C", F)
    ok_r = _ok(r, "delet", "column", "done", "ok", "success", "remov")
    try:
        wb = _wb(F); ws = wb.active
        col_c = ws.cell(1,3).value   # should be Grade again
        wb.close()
        ok_v = col_c == "Grade"
        log("Column", "delete_column", "delete column C",
            "Grade back at column C after delete",
            "PASS" if ok_v else "FAIL", f"C1={col_c!r}")
    except Exception as e:
        log("Column", "delete_column[verify]", "", "", "FAIL", str(e))

    # 3-D  set_column_width
    r = _cmd("set column width A to 25", F)
    ok_r = _ok(r, "width", "column", "done", "ok", "set", "success")
    try:
        wb = _wb(F); ws = wb.active
        w = ws.column_dimensions["A"].width
        wb.close()
        ok_v = w is not None and float(w) >= 20
        log("Column", "set_column_width", "set column width A to 25",
            "column A width >= 20",
            "PASS" if ok_v else "FAIL", f"width={w}")
    except Exception as e:
        log("Column", "set_column_width[verify]", "", "", "FAIL", str(e))

    cleanup(F)

# ══════════════════════════════════════════════════════════════════════════════
#  CATEGORY 4 — CELL OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════
def test_cells():
    section("4. CELL OPERATIONS")
    F = "qa_audit_cells.xlsx"
    _cmd(f"create excel file {F}", F)

    # 4-A  write_cell (text)
    r = _cmd("write cell A1 Hello", F)
    try:
        wb = _wb(F); ws = wb.active; v = ws["A1"].value; wb.close()
        ok_v = str(v).strip() == "Hello"
        log("Cell", "write_cell[text]", "write cell A1 Hello",
            "A1 = 'Hello'", "PASS" if ok_v else "FAIL", f"A1={v!r}")
    except Exception as e:
        log("Cell", "write_cell[text][verify]", "", "", "FAIL", str(e))

    # 4-B  write_cell (number)
    r = _cmd("write cell B1 42", F)
    try:
        wb = _wb(F); ws = wb.active; v = ws["B1"].value; wb.close()
        ok_v = str(v).strip() == "42" or v == 42
        log("Cell", "write_cell[number]", "write cell B1 42",
            "B1 = 42", "PASS" if ok_v else "FAIL", f"B1={v!r}")
    except Exception as e:
        log("Cell", "write_cell[number][verify]", "", "", "FAIL", str(e))

    # 4-C  write_cell (multi-word value)
    r = _cmd("write cell C1 John Smith", F)
    try:
        wb = _wb(F); ws = wb.active; v = ws["C1"].value; wb.close()
        ok_v = "John" in str(v)
        log("Cell", "write_cell[multiword]", "write cell C1 John Smith",
            "C1 contains 'John'", "PASS" if ok_v else "FAIL", f"C1={v!r}")
    except Exception as e:
        log("Cell", "write_cell[multiword][verify]", "", "", "FAIL", str(e))

    # 4-D  read_cell
    r = _cmd("read cell A1", F)
    ok_r = _ok(r, "hello", "a1", "value", "cell")
    log("Cell", "read_cell", "read cell A1",
        "response mentions Hello/A1",
        "PASS" if ok_r else "FAIL", r)

    # 4-E  delete_cell
    _cmd("write cell D1 TEMP", F)
    r = _cmd("delete cell D1", F)
    ok_r = _ok(r, "delet", "clear", "done", "ok", "success", "remov")
    try:
        wb = _wb(F); ws = wb.active; v = ws["D1"].value; wb.close()
        ok_v = v is None
        log("Cell", "delete_cell", "delete cell D1",
            "D1 is None after delete",
            "PASS" if ok_v else "FAIL", f"D1={v!r}")
    except Exception as e:
        log("Cell", "delete_cell[verify]", "", "", "FAIL", str(e))

    cleanup(F)

# ══════════════════════════════════════════════════════════════════════════════
#  CATEGORY 5 — RANGE OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════
def test_ranges():
    section("5. RANGE OPERATIONS")
    F = "qa_audit_ranges.xlsx"
    _cmd(f"create excel file {F}", F)
    for cell, val in [("A1","Name"),("B1","Score"),("A2","Alice"),("B2","85"),
                      ("A3","Bob"),("B3","72"),("A4","Charlie"),("B4","90")]:
        _cmd(f"write cell {cell} {val}", F)

    # 5-A  read_range
    r = _cmd("read range A1:B3", F)
    ok_r = _ok(r, "name", "score", "alice", "bob", "range", "a1", "b1")
    log("Range", "read_range", "read range A1:B3",
        "response includes data from range",
        "PASS" if ok_r else "FAIL", r[:80])

    # 5-B  delete_range
    _cmd("write cell E1 X", F)
    _cmd("write cell E2 Y", F)
    r = _cmd("delete range E1:E2", F)
    ok_r = _ok(r, "delet", "clear", "done", "ok", "success", "remov", "rang")
    try:
        wb = _wb(F); ws = wb.active
        v1 = ws["E1"].value; v2 = ws["E2"].value
        wb.close()
        ok_v = v1 is None and v2 is None
        log("Range", "delete_range", "delete range E1:E2",
            "E1 and E2 are None",
            "PASS" if ok_v else "FAIL", f"E1={v1!r} E2={v2!r}")
    except Exception as e:
        log("Range", "delete_range[verify]", "", "", "FAIL", str(e))

    cleanup(F)

# ══════════════════════════════════════════════════════════════════════════════
#  CATEGORY 6 — FORMULA OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════
def test_formulas():
    section("6. FORMULA OPERATIONS")
    F = "qa_audit_formulas.xlsx"
    _cmd(f"create excel file {F}", F)
    for cell, val in [("A1","Name"),("B1","Score"),
                      ("A2","Alice"),("B2","85"),
                      ("A3","Bob"),("B3","72"),
                      ("A4","Charlie"),("B4","90")]:
        _cmd(f"write cell {cell} {val}", F)

    # 6-A  write_formula  (=SUM)
    r = _cmd("write formula SUM(B2:B4) in C1", F)
    ok_r = _ok(r, "formula", "write", "done", "ok", "success", "sum", "c1")
    try:
        wb = _wb(F); ws = wb.active; v = ws["C1"].value; wb.close()
        ok_v = v is not None and "SUM" in str(v).upper()
        log("Formula", "write_formula[SUM]", "write formula SUM(B2:B4) in C1",
            "C1 contains =SUM formula",
            "PASS" if ok_v else "FAIL", f"C1={v!r}")
    except Exception as e:
        log("Formula", "write_formula[SUM][verify]", "", "", "FAIL", str(e))

    # 6-B  write_formula  (=AVERAGE)
    r = _cmd("write formula AVERAGE(B2:B4) in C2", F)
    try:
        wb = _wb(F); ws = wb.active; v = ws["C2"].value; wb.close()
        ok_v = v is not None and "AVERAGE" in str(v).upper()
        log("Formula", "write_formula[AVG]", "write formula AVERAGE(B2:B4) in C2",
            "C2 contains =AVERAGE formula",
            "PASS" if ok_v else "FAIL", f"C2={v!r}")
    except Exception as e:
        log("Formula", "write_formula[AVG][verify]", "", "", "FAIL", str(e))

    # 6-C  write_sum_formula  (via "calculate sum … in …")
    r = _cmd("calculate sum B2:B4 in D1", F)
    ok_r = _ok(r, "sum", "formula", "done", "ok", "success", "calcul", "d1")
    try:
        wb = _wb(F); ws = wb.active; v = ws["D1"].value; wb.close()
        ok_v = v is not None and "SUM" in str(v).upper()
        log("Formula", "write_sum_formula", "calculate sum B2:B4 in D1",
            "D1 contains =SUM formula",
            "PASS" if ok_v else "FAIL", f"D1={v!r}")
    except Exception as e:
        log("Formula", "write_sum_formula[verify]", "", "", "FAIL", str(e))

    # 6-D  write_average_formula  (via "calculate average … in …")
    r = _cmd("calculate average B2:B4 in D2", F)
    ok_r = _ok(r, "average", "formula", "done", "ok", "success", "calcul", "d2")
    try:
        wb = _wb(F); ws = wb.active; v = ws["D2"].value; wb.close()
        ok_v = v is not None and "AVERAGE" in str(v).upper()
        log("Formula", "write_average_formula", "calculate average B2:B4 in D2",
            "D2 contains =AVERAGE formula",
            "PASS" if ok_v else "FAIL", f"D2={v!r}")
    except Exception as e:
        log("Formula", "write_average_formula[verify]", "", "", "FAIL", str(e))

    # 6-E  calculate_row_sum
    r = _cmd("calculate row sum for row 2", F)
    ok_r = _ok(r, "sum", "row", "done", "ok", "success", "calcul", "total")
    log("Formula", "calculate_row_sum", "calculate row sum for row 2",
        "response mentions sum/row/done",
        "PASS" if ok_r else "FAIL", r[:80])

    cleanup(F)

# ══════════════════════════════════════════════════════════════════════════════
#  CATEGORY 7 — FORMATTING
# ══════════════════════════════════════════════════════════════════════════════
def test_formatting():
    section("7. FORMATTING")
    F = "qa_audit_fmt.xlsx"
    _cmd(f"create excel file {F}", F)
    for cell, val in [("A1","Name"),("B1","Score"),("A2","Alice"),("B2","85")]:
        _cmd(f"write cell {cell} {val}", F)

    # 7-A  format_cells  bold
    r = _cmd("format cells A1:B1 bold", F)
    ok_r = _ok(r, "format", "bold", "done", "ok", "success", "appl")
    try:
        wb = _wb(F); ws = wb.active; bold = ws["A1"].font.bold; wb.close()
        ok_v = bold is True
        log("Format", "format_cells[bold]", "format cells A1:B1 bold",
            "A1 font.bold = True",
            "PASS" if ok_v else "FAIL", f"bold={bold}")
    except Exception as e:
        log("Format", "format_cells[bold][verify]", "", "", "FAIL", str(e))

    # 7-B  format_cells  italic
    r = _cmd("format cells A1:B1 italic", F)
    ok_r = _ok(r, "format", "italic", "done", "ok", "success", "appl")
    try:
        wb = _wb(F); ws = wb.active; ital = ws["A1"].font.italic; wb.close()
        ok_v = ital is True
        log("Format", "format_cells[italic]", "format cells A1:B1 italic",
            "A1 font.italic = True",
            "PASS" if ok_v else "FAIL", f"italic={ital}")
    except Exception as e:
        log("Format", "format_cells[italic][verify]", "", "", "FAIL", str(e))

    # 7-C  format_cells  font_size
    r = _cmd("format cells A1:B1 font size 16", F)
    ok_r = _ok(r, "format", "font", "size", "done", "ok", "success", "appl")
    try:
        wb = _wb(F); ws = wb.active; sz = ws["A1"].font.size; wb.close()
        ok_v = sz is not None and float(sz) >= 14
        log("Format", "format_cells[font_size]", "format cells A1:B1 font size 16",
            "A1 font.size >= 14",
            "PASS" if ok_v else "FAIL", f"size={sz}")
    except Exception as e:
        log("Format", "format_cells[font_size][verify]", "", "", "FAIL", str(e))

    # 7-D  format_cells  background color
    r = _cmd("format cells A1:B1 background color yellow", F)
    ok_r = _ok(r, "format", "color", "background", "done", "ok", "success", "appl", "yellow")
    log("Format", "format_cells[bg_color]", "format cells A1:B1 background color yellow",
        "response ok",
        "PASS" if ok_r else "FAIL", r[:80])

    cleanup(F)

# ══════════════════════════════════════════════════════════════════════════════
#  CATEGORY 8 — MERGE / UNMERGE
# ══════════════════════════════════════════════════════════════════════════════
def test_merge():
    section("8. MERGE / UNMERGE")
    F = "qa_audit_merge.xlsx"
    _cmd(f"create excel file {F}", F)
    for cell, val in [("A1","Title"),("B1","Sub"),("C1","Extra")]:
        _cmd(f"write cell {cell} {val}", F)

    # 8-A  merge_cells
    r = _cmd("merge cells A1:C1", F)
    ok_r = _ok(r, "merge", "cell", "done", "ok", "success")
    try:
        wb = _wb(F); ws = wb.active
        merged = [str(m) for m in ws.merged_cells.ranges]
        wb.close()
        ok_v = any("A1" in m and "C1" in m for m in merged)
        log("Merge", "merge_cells", "merge cells A1:C1",
            "A1:C1 in merged ranges",
            "PASS" if ok_v else "FAIL", str(merged))
    except Exception as e:
        log("Merge", "merge_cells[verify]", "", "", "FAIL", str(e))

    # 8-B  unmerge_cells
    r = _cmd("unmerge cells A1:C1", F)
    ok_r = _ok(r, "unmerge", "merge", "cell", "done", "ok", "success")
    try:
        wb = _wb(F); ws = wb.active
        merged = [str(m) for m in ws.merged_cells.ranges]
        wb.close()
        ok_v = not any("A1" in m and "C1" in m for m in merged)
        log("Merge", "unmerge_cells", "unmerge cells A1:C1",
            "A1:C1 NOT in merged ranges",
            "PASS" if ok_v else "FAIL", str(merged))
    except Exception as e:
        log("Merge", "unmerge_cells[verify]", "", "", "FAIL", str(e))

    cleanup(F)

# ══════════════════════════════════════════════════════════════════════════════
#  CATEGORY 9 — FREEZE / UNFREEZE
# ══════════════════════════════════════════════════════════════════════════════
def test_freeze():
    section("9. FREEZE / UNFREEZE PANES")
    F = "qa_audit_freeze.xlsx"
    _cmd(f"create excel file {F}", F)
    for cell, val in [("A1","Name"),("B1","Score"),("A2","Alice"),("B2","85")]:
        _cmd(f"write cell {cell} {val}", F)

    # 9-A  freeze_panes
    r = _cmd("freeze panes B2", F)
    ok_r = _ok(r, "freeze", "pane", "done", "ok", "success")
    try:
        wb = _wb(F); ws = wb.active; fp = ws.freeze_panes; wb.close()
        ok_v = fp is not None
        log("Freeze", "freeze_panes", "freeze panes B2",
            "freeze_panes is not None",
            "PASS" if ok_v else "FAIL", f"freeze_panes={fp!r}")
    except Exception as e:
        log("Freeze", "freeze_panes[verify]", "", "", "FAIL", str(e))

    # 9-B  unfreeze_panes
    r = _cmd("unfreeze panes", F)
    ok_r = _ok(r, "unfreeze", "freeze", "pane", "done", "ok", "success", "remov")
    try:
        wb = _wb(F); ws = wb.active; fp = ws.freeze_panes; wb.close()
        ok_v = fp is None
        log("Freeze", "unfreeze_panes", "unfreeze panes",
            "freeze_panes is None",
            "PASS" if ok_v else "FAIL", f"freeze_panes={fp!r}")
    except Exception as e:
        log("Freeze", "unfreeze_panes[verify]", "", "", "FAIL", str(e))

    cleanup(F)

# ══════════════════════════════════════════════════════════════════════════════
#  CATEGORY 10 — CHART
# ══════════════════════════════════════════════════════════════════════════════
def test_chart():
    section("10. CHART")
    F = "qa_audit_chart.xlsx"
    _cmd(f"create excel file {F}", F)
    for cell, val in [("A1","Name"),("B1","Score"),
                      ("A2","Alice"),("B2","85"),
                      ("A3","Bob"),("B3","72")]:
        _cmd(f"write cell {cell} {val}", F)

    # 10-A  create_chart
    r = _cmd("create bar chart from B2:B3 titled Scores", F)
    ok_r = _ok(r, "chart", "creat", "done", "ok", "success", "bar", "add")
    try:
        wb = _wb(F); ws = wb.active
        has_chart = len(ws._charts) > 0
        wb.close()
        log("Chart", "create_chart", "create bar chart from B2:B3 titled Scores",
            "at least one chart in worksheet",
            "PASS" if has_chart else "FAIL",
            f"charts={len(ws._charts) if not has_chart else 'yes'}")
    except Exception as e:
        log("Chart", "create_chart[verify]", "", "", "FAIL", str(e))

    cleanup(F)

# ══════════════════════════════════════════════════════════════════════════════
#  CATEGORY 11 — SHEET OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════
def test_sheets():
    section("11. SHEET OPERATIONS")
    F = "qa_audit_sheets.xlsx"
    _cmd(f"create excel file {F}", F)
    _cmd("write cell A1 Name", F)

    # 11-A  create_sheet
    r = _cmd("create sheet Results", F)
    ok_r = _ok(r, "creat", "sheet", "done", "ok", "success", "results")
    try:
        wb = _wb(F)
        ok_v = "Results" in wb.sheetnames; wb.close()
        log("Sheet", "create_sheet", "create sheet Results",
            "Results in sheetnames",
            "PASS" if ok_v else "FAIL", str(wb.sheetnames if not ok_v else "ok"))
    except Exception as e:
        log("Sheet", "create_sheet[verify]", "", "", "FAIL", str(e))

    # 11-B  list_sheets
    r = _cmd("list sheets", F)
    ok_r = _ok(r, "sheet", "results", "sheet1", "available", "list")
    log("Sheet", "list_sheets", "list sheets",
        "response lists sheet names",
        "PASS" if ok_r else "FAIL", r[:80])

    # 11-C  switch_sheet
    r = _cmd("switch sheet Results", F)
    ok_r = _ok(r, "switch", "sheet", "results", "done", "ok", "activ", "now on")
    log("Sheet", "switch_sheet", "switch sheet Results",
        "response mentions Results/active",
        "PASS" if ok_r else "FAIL", r[:80])

    # 11-D  rename_sheet
    r = _cmd("rename sheet Results to Summary", F)
    ok_r = _ok(r, "renam", "sheet", "done", "ok", "success", "summary")
    try:
        wb = _wb(F)
        ok_v = "Summary" in wb.sheetnames and "Results" not in wb.sheetnames
        wb.close()
        log("Sheet", "rename_sheet", "rename sheet Results to Summary",
            "Summary exists, Results gone",
            "PASS" if ok_v else "FAIL",
            f"sheets={wb.sheetnames}" if not ok_v else "ok")
    except Exception as e:
        log("Sheet", "rename_sheet[verify]", "", "", "FAIL", str(e))

    # switch back to Sheet1 before delete
    _cmd("switch sheet Sheet1", F)

    # 11-E  delete_sheet
    r = _cmd("delete sheet Summary", F)
    ok_r = _ok(r, "delet", "sheet", "done", "ok", "success", "remov")
    try:
        wb = _wb(F)
        ok_v = "Summary" not in wb.sheetnames; wb.close()
        log("Sheet", "delete_sheet", "delete sheet Summary",
            "Summary not in sheetnames",
            "PASS" if ok_v else "FAIL",
            f"sheets={wb.sheetnames}" if not ok_v else "ok")
    except Exception as e:
        log("Sheet", "delete_sheet[verify]", "", "", "FAIL", str(e))

    cleanup(F)

# ══════════════════════════════════════════════════════════════════════════════
#  CATEGORY 12 — SORT / FIND-REPLACE / FILTER
# ══════════════════════════════════════════════════════════════════════════════
def test_data_ops():
    section("12. SORT / FIND-REPLACE / FILTER")
    F = "qa_audit_data.xlsx"
    _cmd(f"create excel file {F}", F)
    for cell, val in [("A1","Name"),("B1","Score"),
                      ("A2","Alice"),("B2","85"),
                      ("A3","Bob"),  ("B3","72"),
                      ("A4","Charlie"),("B4","90"),
                      ("A5","Dave"), ("B5","60")]:
        _cmd(f"write cell {cell} {val}", F)

    # 12-A  sort_range ascending
    r = _cmd("sort range by column B ascending", F)
    ok_r = _ok(r, "sort", "done", "ok", "success", "ascend", "order")
    try:
        wb = _wb(F); ws = wb.active
        scores = [ws.cell(i,2).value for i in range(2,6)]
        wb.close()
        numeric = [float(s) for s in scores if s is not None]
        ok_v = numeric == sorted(numeric)
        log("Data", "sort_range[asc]", "sort range by column B ascending",
            "column B ascending order",
            "PASS" if ok_v else "FAIL", str(scores))
    except Exception as e:
        log("Data", "sort_range[asc][verify]", "", "", "FAIL", str(e))

    # 12-B  sort_range descending
    r = _cmd("sort range by column B descending", F)
    try:
        wb = _wb(F); ws = wb.active
        scores = [ws.cell(i,2).value for i in range(2,6)]
        wb.close()
        numeric = [float(s) for s in scores if s is not None]
        ok_v = numeric == sorted(numeric, reverse=True)
        log("Data", "sort_range[desc]", "sort range by column B descending",
            "column B descending order",
            "PASS" if ok_v else "FAIL", str(scores))
    except Exception as e:
        log("Data", "sort_range[desc][verify]", "", "", "FAIL", str(e))

    # 12-C  find_replace
    r = _cmd("find Alice replace with Alicia", F)
    ok_r = _ok(r, "find", "replac", "done", "ok", "success", "alicia")
    try:
        wb = _wb(F); ws = wb.active
        vals = [ws.cell(i,1).value for i in range(1, ws.max_row+1)]
        wb.close()
        ok_v = "Alicia" in vals and "Alice" not in vals
        log("Data", "find_replace", "find Alice replace with Alicia",
            "Alicia present, Alice gone",
            "PASS" if ok_v else "FAIL", str(vals))
    except Exception as e:
        log("Data", "find_replace[verify]", "", "", "FAIL", str(e))

    # 12-D  filter_table
    r = _cmd("filter table column B greater than 75", F)
    ok_r = _ok(r, "filter", "column", "done", "ok", "success",
               "hidden", "row", "match", "found", "result")
    log("Data", "filter_table", "filter table column B greater than 75",
        "response mentions filter/rows/ok",
        "PASS" if ok_r else "FAIL", r[:80])

    # 12-E  clear_filter
    r = _cmd("clear filter", F)
    ok_r = _ok(r, "clear", "filter", "done", "ok", "success", "remov", "show", "reset")
    log("Data", "clear_filter", "clear filter",
        "response mentions clear/filter/ok",
        "PASS" if ok_r else "FAIL", r[:80])

    cleanup(F)

# ══════════════════════════════════════════════════════════════════════════════
#  CATEGORY 13 — UNDO
# ══════════════════════════════════════════════════════════════════════════════
def test_undo():
    section("13. UNDO")
    F = "qa_audit_undo.xlsx"
    _cmd(f"create excel file {F}", F)
    _cmd("write cell A1 Original", F)
    _cmd("save", F)                       # snapshot saved

    # Make a known change
    _cmd("write cell A1 CHANGED", F)
    try:
        wb = _wb(F); v_before = ws["A1"].value if False else wb.active["A1"].value; wb.close()
    except Exception:
        v_before = "?"

    r = _cmd("undo", F)
    ok_r = _ok(r, "undo", "revert", "done", "ok", "success", "restor", "previous")

    try:
        wb = _wb(F); v_after = wb.active["A1"].value; wb.close()
        ok_v = v_after != "CHANGED"
        log("Undo", "undo", "undo",
            "A1 reverted from 'CHANGED'",
            "PASS" if ok_v else "FAIL",
            f"before={v_before!r} after={v_after!r}")
    except Exception as e:
        log("Undo", "undo[verify]", "", "", "FAIL", str(e))

    cleanup(F)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — run all categories, print report
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"\n{'━'*68}")
    print(f"  Sara Non-Voice Command Audit  —  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Files dir: {_FILES_DIR}")
    print(f"{'━'*68}")

    test_file()
    test_rows()
    test_columns()
    test_cells()
    test_ranges()
    test_formulas()
    test_formatting()
    test_merge()
    test_freeze()
    test_chart()
    test_sheets()
    test_data_ops()
    test_undo()

    # ── summary ───────────────────────────────────────────────────────────────
    passed  = sum(1 for r in _results if r.status == "PASS")
    failed  = sum(1 for r in _results if r.status == "FAIL")
    skipped = sum(1 for r in _results if r.status == "SKIP")
    total   = len(_results)

    print(f"\n{'━'*68}")
    print(f"  OVERALL: {passed}/{total} PASS   {failed} FAIL   {skipped} SKIP")
    print(f"{'━'*68}")

    # Section 4 — full per-command table
    print("\n### Section 4: Test Results\n")
    print(f"  {'Category':<10} {'Command':<32} {'Status'} {'Actual / Note'}")
    print("  " + "─" * 90)
    for r in _results:
        icon = "✔" if r.status=="PASS" else ("⚠" if r.status=="SKIP" else "✘")
        detail = (r.note or r.actual)[:46]
        print(f"  {icon} {r.category:<10} {r.cmd_id:<32} {r.status:<6} {detail}")

    # Section 5 — failure report
    failures = [r for r in _results if r.status == "FAIL"]
    print(f"\n### Section 5: Failure Report  ({len(failures)} failure(s))\n")
    if not failures:
        print("  ✔ No failures — all tested commands stable.")
    else:
        for r in failures:
            print(f"\n  ✘ COMMAND  : {r.cmd_id}")
            print(f"    PHRASE   : {r.phrase}")
            print(f"    EXPECTED : {r.expected}")
            print(f"    ACTUAL   : {r.actual[:120]}")
            if r.note:
                print(f"    NOTE     : {r.note}")
            # Classify failure type
            if "[verify]" in r.cmd_id:
                kind = "WORKBOOK-STATE — command ran but Excel state was not as expected"
            elif not r.actual or r.actual.strip() == "":
                kind = "DISPATCH — no response returned (possible crash or routing failure)"
            else:
                kind = "RESPONSE-MESSAGE — command ran but response text was unexpected"
            print(f"    TYPE     : {kind}")

    print(f"\n{'━'*68}\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
