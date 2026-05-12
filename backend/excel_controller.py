"""Excel automation helpers using openpyxl.

This module provides simple wrappers to create workbooks and modify rows/columns.

Undo System
-----------
A module-level rollback stack (_UNDO_STACK) records a file-level snapshot
before every destructive operation (delete_row, delete_column, find_replace).
``undo_last(path)`` restores the most recent snapshot for that file.

Stack is bounded to _UNDO_STACK_LIMIT entries per file to cap memory usage.
Snapshots are stored in-memory as raw bytes — no temp files needed.
"""
from openpyxl import Workbook, load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.chart import BarChart, LineChart, PieChart, ScatterChart, Reference
import collections
import io
import os
import re
import string
import logging
import sys
import tempfile
import threading
import time

from pipeline_sanitizer import sanitize_cell_value, is_formula_safe, get_formula_error

log = logging.getLogger('excel_controller')

_CELL_RE = re.compile(r'^([A-Z]+)(\d+)$')
_MAX_PATH_LEN = 260

# ---------------------------------------------------------------------------
# Persistent workbook session — one file at a time
# ---------------------------------------------------------------------------
# Instead of calling load_workbook() on every command, Sara keeps a single
# _WorkbookSession in memory.  Disk writes happen only when:
#   (a) a modification occurs and _flush_session() / close_excel() is called, or
#   (b) a new file is opened/created (initial save so the path exists on disk).
#
# Thread safety: _session_lock guards every read/write of _session.  The lock
# is held briefly to copy refs, then released before I/O so slow disk ops
# never block unrelated threads.

class _WorkbookSession:
    """In-memory workbook state for one active file."""

    __slots__ = ('abs_path', 'workbook', 'dirty')

    def __init__(self, abs_path: str, wb: Workbook) -> None:
        self.abs_path: str = abs_path      # normalised absolute path
        self.workbook: Workbook = wb       # live openpyxl Workbook object
        self.dirty: bool = False           # True = has unsaved in-memory changes


_session_lock = threading.Lock()
_session: _WorkbookSession | None = None  # single active session

# Default file created automatically when a modifying command is issued with
# no file open (e.g. "add row John 90" before any open/create command).
_DEFAULT_FILENAME = 'sara_default.xlsx'

# Retry budget for _atomic_save when another process (e.g. Excel.exe) holds a
# write lock on the file (Windows WinError 32 / PermissionError).
_SAVE_RETRIES = 5
_SAVE_RETRY_DELAY = 0.5                  # seconds between retries
# Time to wait after a COM close before calling os.replace.  Excel releases
# its write lock asynchronously; skipping this sleep causes WinError 32 on
# the very first os.replace attempt even though the workbook was just closed.
# Increased from 0.12 → 0.25 → 0.5 s: logs showed first add-row failing when
# Excel holds a lock longer than 0.25 s on some Windows 11 configurations.
_LOCK_RELEASE_DELAY = 0.25               # seconds to wait after COM close
_LOCK_POLL_INTERVAL = 0.05               # seconds between _poll_for_unlock checks
_LOCK_RELEASE_TIMEOUT = 4.0             # max seconds to wait before giving up on lock release

# ---------------------------------------------------------------------------
# Undo / rollback stack
# ---------------------------------------------------------------------------
# Maps absolute file path → deque of (description, workbook_bytes) snapshots.
# Storing raw bytes keeps the snapshot self-contained and independent of any
# external backup file.  Each entry is typically 20–200 KB for a small sheet.

_UNDO_STACK_LIMIT = 10   # max snapshots per file

# {abs_path: deque[(description, bytes)]}
_UNDO_STACK: dict[str, collections.deque] = {}

# COM workbook reference cache — keyed by normcase(normpath(abs_path)).
# Populated when Sara opens a workbook via _launch_excel_com so that
# _com_find_workbook can return it directly without relying on GetActiveObject
# (which fails when Excel was opened via os.startfile and is not in the COM ROT).
_com_wb_cache: dict[str, object] = {}


def _undo_snapshot(path: str, description: str) -> None:
    """Save a workbook snapshot to the undo stack BEFORE a destructive write.

    Prefers serialising the live in-memory workbook (which is always current)
    over reading from disk (which may be stale when the session is dirty).
    Falls back to disk when no matching session is found.
    """
    abs_path = os.path.abspath(path)
    raw: bytes | None = None

    # Serialize from the in-memory session when available — it is authoritative.
    with _session_lock:
        if (_session is not None
                and os.path.normcase(_session.abs_path) == os.path.normcase(abs_path)):
            try:
                buf = io.BytesIO()
                _session.workbook.save(buf)
                raw = buf.getvalue()
            except Exception as exc:
                log.warning('[Undo] in-memory snapshot failed, falling back to disk: %s', exc)

    if raw is None:
        # Fallback: read from disk (no session, or serialisation failed).
        if not os.path.exists(abs_path):
            return
        try:
            with open(abs_path, 'rb') as fh:
                raw = fh.read()
        except Exception as exc:
            log.warning('[Undo] snapshot failed (non-fatal): %s', exc)
            return

    if abs_path not in _UNDO_STACK:
        _UNDO_STACK[abs_path] = collections.deque(maxlen=_UNDO_STACK_LIMIT)
    _UNDO_STACK[abs_path].append((description, raw))
    log.debug('[Undo] snapshot saved: %r  stack_depth=%d',
              description, len(_UNDO_STACK[abs_path]))


def undo_last(path: str) -> dict:
    """Restore the most recent undo snapshot for *path*.

    Returns a result dict with 'speech_text' describing what was restored.
    Returns an error dict if no snapshot is available.

    Usage::

        result = undo_last("backend/files/sales.xlsx")
        # → {"message": "undo_complete", "description": "delete row 3",
        #    "speech_text": "Done! I've restored the previous state. Row 3 is back."}
    """
    global _session          # required: else branch assigns to _session
    abs_path = os.path.abspath(path)
    stack = _UNDO_STACK.get(abs_path)

    if not stack:
        log.info('[Undo] no snapshot available for %s', path)
        return {
            "error": "no_undo_available",
            "speech_text": "There's nothing to undo right now.",
        }

    description, raw = stack.pop()
    log.info('[Undo] restoring snapshot: %r  (%d bytes)', description, len(raw))

    try:
        # Restore in-memory: rebuild the workbook from snapshot bytes and
        # update the session directly.  We do NOT rename files on disk here
        # because Excel holds a write lock on the open workbook on Windows,
        # causing os.replace to fail with WinError 5 / PermissionError.
        # _flush_session (called below) already handles the lock safely via
        # _com_close_workbook before writing.
        reloaded = load_workbook(io.BytesIO(raw))
        with _session_lock:
            if (_session is not None
                    and os.path.normcase(_session.abs_path) == os.path.normcase(abs_path)):
                _session.workbook = reloaded
                _session.dirty = True   # needs flushing to disk + Excel UI refresh
            else:
                # Session points at a different file — create a new session for
                # the undo target so subsequent commands operate on it.
                _session = _WorkbookSession(abs_path, reloaded)
                _session.dirty = True

        # Write to disk and refresh the Excel UI.  _flush_session closes Excel's
        # copy before writing (releasing the lock) then reopens the fresh file.
        _flush_session(abs_path)

        remaining = len(stack)
        return {
            "message": "undo_complete",
            "description": description,
            "remaining_undos": remaining,
            "speech_text": (
                f"Done! I've undone the last action ({description}). "
                + (f"You can undo {remaining} more time{'s' if remaining != 1 else ''}."
                   if remaining else "That was the last undo available.")
            ),
        }
    except Exception as exc:
        log.error('[Undo] restore failed: %s', exc)
        return {
            "error": str(exc),
            "speech_text": "I couldn't undo that action. Your file may not have changed.",
        }


def undo_stack_depth(path: str) -> int:
    """Return how many undo steps are available for *path*."""
    abs_path = os.path.abspath(path)
    stack = _UNDO_STACK.get(abs_path)
    return len(stack) if stack else 0


def _col_letters_to_index(letters: str) -> int:
    """Convert column letter(s) like 'A', 'B', 'AA' to a 1-based index."""
    idx = 0
    for c in letters:
        idx = idx * 26 + (ord(c) - ord('A') + 1)
    return idx


def _atomic_save(wb, path):
    """Save workbook to a temp file then atomically replace the target.

    On Windows, another process (typically Excel.exe) may hold a write lock
    on the file, causing ``os.replace`` to raise a ``PermissionError``
    (WinError 32).  We retry up to ``_SAVE_RETRIES`` times before re-raising,
    which handles the common case where the user has just closed Excel.
    """
    dir_name = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(suffix='.xlsx', dir=dir_name)
    os.close(fd)
    try:
        wb.save(tmp)
        for attempt in range(_SAVE_RETRIES):
            try:
                os.replace(tmp, path)
                return                   # success — exit retry loop
            except PermissionError as exc:
                if attempt < _SAVE_RETRIES - 1:
                    log.warning(
                        '_atomic_save: file locked by another process '
                        '(attempt %d/%d) — retrying in %.1fs.  '
                        'Close the file in Excel and retry if this persists.  '
                        'Error: %s',
                        attempt + 1, _SAVE_RETRIES, _SAVE_RETRY_DELAY, exc,
                    )
                    time.sleep(_SAVE_RETRY_DELAY)
                else:
                    log.error(
                        '_atomic_save: could not save %s after %d attempts — '
                        'file is locked.  Close it in Excel and try again.',
                        path, _SAVE_RETRIES,
                    )
                    raise
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _auto_open_default() -> None:
    """Create or load the default workbook and set it as the active session.

    MUST be called while ``_session_lock`` is held by the caller.
    Saves the workbook to disk immediately so the path exists on disk before
    any dependent code (COM automation, undo snapshots) tries to access it.
    """
    global _session
    files_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files')
    os.makedirs(files_dir, exist_ok=True)
    default_path = os.path.abspath(os.path.join(files_dir, _DEFAULT_FILENAME))

    if os.path.exists(default_path):
        try:
            wb = load_workbook(default_path)
        except Exception:
            wb = Workbook()
    else:
        wb = Workbook()

    _session = _WorkbookSession(default_path, wb)

    # Ensure the file exists on disk so COM and precondition checks work.
    if not os.path.exists(default_path):
        try:
            _atomic_save(wb, default_path)
        except Exception as exc:
            log.warning('[Session] could not pre-save default file: %s', exc)
            _session.dirty = True   # will be saved on next _flush_session call

    log.info('[Session] auto-opened default: %s', default_path)


def _mark_dirty() -> None:
    """Mark the active session as having unsaved in-memory changes.

    Called by every function that modifies the workbook.  The actual disk
    write is deferred until ``_flush_session()`` or ``close_excel()``.
    """
    with _session_lock:
        if _session is not None:
            _session.dirty = True


def _flush_session(path: str | None = None) -> bool:
    """Write the in-memory workbook to disk if it has unsaved changes.

    Parameters
    ----------
    path : When given, only flushes if the active session path matches.
           When omitted, always flushes the active session.

    Returns ``True`` when a save was performed, ``False`` when skipped
    (nothing to save or path mismatch).
    """
    wb_to_save: Workbook | None = None
    save_path: str | None = None

    with _session_lock:
        if _session is None or not _session.dirty:
            return False
        if path:
            abs_path = os.path.abspath(path)
            if os.path.normcase(_session.abs_path) != os.path.normcase(abs_path):
                return False
        wb_to_save = _session.workbook
        save_path = _session.abs_path

    # Close the workbook in the running Excel instance before writing.
    # This releases Excel's write lock so _atomic_save can replace the file
    # without hitting WinError 5 / PermissionError.  No-op when Excel is
    # not running or the file is not open in Excel.
    was_open = _com_close_workbook(save_path)

    # Poll until the OS releases the write lock, or proceed immediately when
    # Excel is not running (_is_file_locked returns False right away).
    # Polling exits the instant the handle drops instead of a fixed worst-case
    # sleep.  Also catches the case where _com_close_workbook returned False
    # due to a path-representation mismatch yet the file IS still locked.
    if was_open or _is_file_locked(save_path):
        _poll_for_unlock(save_path)

    # I/O outside the lock so concurrent reads are not blocked.
    _atomic_save(wb_to_save, save_path)

    with _session_lock:
        if _session is not None and _session.abs_path == save_path:
            _session.dirty = False

    # Reopen the saved file in the running Excel instance so the UI shows
    # the latest data immediately.  No-op when Excel is not running.
    _com_open_workbook(save_path)

    log.info('[Session] flushed: %s', save_path)
    return True


def _launch_excel_com(abs_path: str) -> None:
    """Open *abs_path* in the Excel application via COM automation.

    Behaviour
    ---------
    * **Reuse**: if Excel is already running, the existing process is used;
      a new one is started only when none is found.
    * **Fresh open**: if the workbook is already loaded in Excel (stale copy
      from a previous session), it is closed without saving before the
      current on-disk version is opened.  openpyxl's file is always the
      authoritative source — Excel's in-memory copy may be out of date.
    * **Visible + foreground**: ``xl.Visible = True`` is set unconditionally
      and ``ActiveWindow.Activate()`` brings the window to the front.

    Platform / availability
    -----------------------
    Windows-only.  When ``pywin32`` is not installed the function returns
    silently so that non-Windows environments (CI, Linux servers) are
    unaffected.  Every COM exception is caught, logged at WARNING level,
    and swallowed — openpyxl file operations must never be blocked by an
    Excel automation failure.
    """
    if sys.platform != 'win32':
        log.debug('[COM] _launch_excel_com: skipped on non-Windows platform')
        return

    try:
        import pythoncom          # type: ignore[import]
        import win32com.client    # type: ignore[import]
    except ImportError:
        log.debug('[COM] pywin32 not installed — Excel launch skipped')
        return

    try:
        # COM requires initialisation on every thread that uses it.
        # CoInitialize returns S_FALSE (not an error) when already initialised
        # on this thread, and raises if the thread is in an incompatible
        # apartment model — both cases are safe to ignore.
        try:
            pythoncom.CoInitialize()
        except Exception:
            pass

        # Prefer reusing the running Excel instance to avoid spawning
        # duplicate processes for the same user session.
        try:
            xl = win32com.client.GetActiveObject('Excel.Application')
            log.debug('[COM] reusing running Excel instance')
        except Exception:
            xl = win32com.client.Dispatch('Excel.Application')
            log.debug('[COM] started new Excel instance')

        xl.Visible = True
        xl.DisplayAlerts = False

        # If this exact file is already open in Excel, close the stale copy
        # first so the fresh on-disk version (just written by openpyxl) is
        # loaded cleanly.  Iterate safely — the collection may shrink.
        norm = os.path.normcase(abs_path)
        for _wb in list(xl.Workbooks):
            try:
                if os.path.normcase(_wb.FullName) == norm:
                    _wb.Close(SaveChanges=False)
                    log.debug('[COM] closed stale workbook: %s', abs_path)
                    break
            except Exception:
                continue

        # Open the authoritative version from disk and cache the COM workbook so
        # _com_find_workbook can return it directly without GetActiveObject.
        _opened_wb = xl.Workbooks.Open(abs_path)
        _com_wb_cache[os.path.normcase(os.path.normpath(abs_path))] = _opened_wb

        # Bring the Excel window to the foreground.
        try:
            xl.ActiveWindow.Activate()
        except Exception:
            pass  # no window yet (headless / terminal server)

        log.info('[COM] displayed in Excel: %s', abs_path)

    except Exception as exc:
        log.warning('[COM] Excel COM automation failed (non-fatal): %s', exc)


def _com_close_workbook(abs_path: str) -> bool:
    """Close *abs_path* in the running Excel instance (without saving).

    Used before an atomic disk write to release Excel's write lock on the
    file.  If Excel is not running or the file is not open, this is a no-op.

    Returns True when the workbook was actually closed, False otherwise.
    All COM exceptions are caught and logged so callers are never blocked.
    """
    if sys.platform != 'win32':
        return False
    try:
        import pythoncom          # type: ignore[import]
        import win32com.client    # type: ignore[import]
    except ImportError:
        return False
    try:
        try:
            pythoncom.CoInitialize()
        except Exception:
            pass
        try:
            xl = win32com.client.GetActiveObject('Excel.Application')
        except Exception:
            return False   # Excel not running — nothing to close
        xl.DisplayAlerts = False   # suppress any pending dialog so Close() doesn't block or raise
        norm = os.path.normcase(abs_path)
        # Pass 1 — exact full-path match (preferred).
        for _wb in list(xl.Workbooks):
            try:
                if os.path.normcase(_wb.FullName) == norm:
                    _wb.Close(SaveChanges=False)
                    _com_wb_cache.pop(norm, None)
                    log.debug('[COM] _com_close_workbook: closed %s', abs_path)
                    return True
            except Exception:
                continue
        # Pass 2 — filename-only fallback.  Handles path-representation mismatches
        # (short 8.3 names, UNC vs drive-letter, OneDrive virtual paths) where
        # FullName doesn't normalise to the same string as abs_path.
        basename_lower = os.path.basename(abs_path).lower()
        for _wb in list(xl.Workbooks):
            try:
                if os.path.basename(_wb.FullName).lower() == basename_lower:
                    _wb.Close(SaveChanges=False)
                    _com_wb_cache.pop(norm, None)
                    log.debug('[COM] _com_close_workbook: closed by name fallback: %s', abs_path)
                    return True
            except Exception:
                continue
    except Exception as exc:
        log.warning('[COM] _com_close_workbook failed (non-fatal): %s', exc)
    return False


def _com_open_workbook(abs_path: str) -> None:
    """Open (or refresh) *abs_path* in the Excel application after a disk write.

    Improvements over the original single-liner:

    1. **Stale-copy guard** — closes any existing open copy before reopening so
       Excel never shows a "file already open" dialog or a duplicate window.
       This handles the edge case where ``_com_close_workbook`` silently failed
       (its inner COM close raised and was swallowed by ``continue``).

    2. **Dispatch fallback** — if ``GetActiveObject`` fails (Excel auto-quit
       when the last workbook was closed), a new instance is started so the
       user's visible session is not silently lost.

    3. **DisplayAlerts = False** — suppresses any interactive dialogs Excel
       might raise (e.g. "file already open", "read-only" warnings) that would
       block the COM call indefinitely.

    4. **Windows(1).Activate()** on the just-opened workbook object — more
       reliable than ``xl.ActiveWindow.Activate()`` which targets whatever
       Excel's current active window happens to be, not the file we just wrote.
    """
    if sys.platform != 'win32':
        return
    try:
        import pythoncom          # type: ignore[import]
        import win32com.client    # type: ignore[import]
    except ImportError:
        return
    try:
        try:
            pythoncom.CoInitialize()
        except Exception:
            pass

        # Prefer reusing the running instance; fall back to starting a fresh one
        # when Excel auto-quit after _com_close_workbook removed the last workbook.
        try:
            xl = win32com.client.GetActiveObject('Excel.Application')
        except Exception:
            try:
                xl = win32com.client.Dispatch('Excel.Application')
                xl.Visible = True
                log.info('[COM] _com_open_workbook: restarted Excel to show %s', abs_path)
            except Exception:
                return   # cannot contact or start Excel — give up silently

        xl.DisplayAlerts = False

        # Close stale copy if it is still open — guards against the case where
        # _com_close_workbook silently failed (exception swallowed inside its loop).
        # Without this guard, Workbooks.Open below would raise a COM error or
        # open a duplicate window.
        norm = os.path.normcase(abs_path)
        for _wb in list(xl.Workbooks):
            try:
                if os.path.normcase(_wb.FullName) == norm:
                    _wb.Close(SaveChanges=False)
                    log.debug('[COM] _com_open_workbook: closed stale copy of %s', abs_path)
                    break
            except Exception:
                continue

        opened_wb = xl.Workbooks.Open(abs_path)
        xl.Visible = True

        # Activate the specific workbook window we just opened.  Using
        # opened_wb.Windows(1) is more reliable than xl.ActiveWindow because
        # it targets this workbook even if another one has focus.
        try:
            opened_wb.Windows(1).Activate()
        except Exception:
            try:
                xl.ActiveWindow.Activate()
            except Exception:
                pass

        log.debug('[COM] _com_open_workbook: opened %s', abs_path)
    except Exception as exc:
        log.warning('[COM] _com_open_workbook failed (non-fatal): %s', exc)


def _com_close_and_maybe_quit(abs_path: str) -> None:
    """Close *abs_path* in Excel; quit the application if no workbooks remain.

    Called from ``close_excel`` so the Excel window disappears when Sara closes
    a file.  If other workbooks are still open in Excel, only this one is closed
    and the application stays running.

    All COM exceptions are swallowed — Python-side session management must
    never be blocked by a COM failure.
    """
    if sys.platform != 'win32':
        return
    try:
        import pythoncom          # type: ignore[import]
        import win32com.client    # type: ignore[import]
    except ImportError:
        return
    try:
        try:
            pythoncom.CoInitialize()
        except Exception:
            pass
        try:
            xl = win32com.client.GetActiveObject('Excel.Application')
        except Exception:
            return   # Excel not running — nothing to close
        xl.DisplayAlerts = False
        # Close this specific workbook (already saved by _atomic_save above).
        norm = os.path.normcase(abs_path)
        for _wb in list(xl.Workbooks):
            try:
                if os.path.normcase(_wb.FullName) == norm:
                    _wb.Close(SaveChanges=False)
                    _com_wb_cache.pop(norm, None)
                    log.debug('[COM] _com_close_and_maybe_quit: closed %s', abs_path)
                    break
            except Exception:
                continue
        # Quit Excel when this was the last open workbook.
        if xl.Workbooks.Count == 0:
            xl.Quit()
            log.info('[COM] _com_close_and_maybe_quit: Excel application quit')
    except Exception as exc:
        log.warning('[COM] _com_close_and_maybe_quit failed (non-fatal): %s', exc)


# ── COM live-write helpers ────────────────────────────────────────────────────
# Used when the target workbook is already open in the running Excel instance.
# Instead of the close→openpyxl-atomic-save→reopen cycle (which races against
# Windows' async file-lock release and causes WinError 32 / 5 "Access is
# denied"), these helpers write changes directly into the live COM session and
# call xl_wb.Save() — Excel owns the file handle, so there is no lock conflict.

def _com_find_workbook(abs_path: str):
    """Return the COM Workbook object for *abs_path* if open in Excel, else None.

    Two-pass path matching (full path, then filename-only) mirrors
    _com_close_workbook so OneDrive / UNC / 8.3-name mismatches are handled.
    Returns None when pywin32 is absent, Excel is not running, or the file is
    not currently open in Excel.
    """
    if sys.platform != 'win32':
        return None
    try:
        import pythoncom          # type: ignore[import]
        import win32com.client    # type: ignore[import]
        try:
            pythoncom.CoInitialize()
        except Exception:
            pass

        norm = os.path.normcase(os.path.normpath(os.path.abspath(abs_path)))
        basename_lower = os.path.basename(norm).lower()

        # Pass 0 — module-level cache: Sara opened the workbook herself via
        # _launch_excel_com which stores the returned Workbook COM object.
        _cached = _com_wb_cache.get(norm)
        if _cached is not None:
            try:
                _ = _cached.FullName   # COM liveness probe
                log.debug('[COM] _com_find_workbook: cache hit %s', abs_path)
                return _cached
            except Exception:
                log.debug('[COM] _com_find_workbook: stale cache removed %s', abs_path)
                _com_wb_cache.pop(norm, None)

        # Pass 1 — ROT scan: covers ALL running Excel instances.
        # Each open workbook registers a moniker:
        #   !Microsoft.Office.Excel.Workbook12|D:\path\to\file.xlsx
        try:
            context = pythoncom.CreateBindCtx(0)
            rot = pythoncom.GetRunningObjectTable()
            for moniker in rot:
                try:
                    display_name = moniker.GetDisplayName(context, None)
                except Exception:
                    continue
                if 'Excel' not in display_name or '|' not in display_name:
                    continue
                wb_path = display_name.split('|', 1)[1]
                wb_norm = os.path.normcase(os.path.normpath(wb_path))
                log.debug('[COM] _com_find_workbook: ROT entry full=%s base=%s',
                          wb_norm, os.path.basename(wb_path).lower())
                if wb_norm == norm or os.path.basename(wb_path).lower() == basename_lower:
                    try:
                        com_obj = rot.GetObject(moniker)
                        idisp = com_obj.QueryInterface(pythoncom.IID_IDispatch)
                        xl_wb = win32com.client.Dispatch(idisp)
                        log.debug('[COM] _com_find_workbook: found via ROT: %s', wb_path)
                        return xl_wb
                    except Exception as exc:
                        log.debug('[COM] _com_find_workbook: ROT dispatch failed: %s', exc)
        except Exception as exc:
            log.debug('[COM] _com_find_workbook: ROT scan error: %s', exc)

        # Pass 2 — GetActiveObject: single-instance fallback.
        # Note: does NOT return None on failure — falls through to Pass 3.
        try:
            xl = win32com.client.GetActiveObject('Excel.Application')
            for _wb in list(xl.Workbooks):
                try:
                    wb_norm = os.path.normcase(os.path.normpath(_wb.FullName))
                    log.debug('[COM] _com_find_workbook: GAO FullName=%s vs norm=%s',
                              wb_norm, norm)
                    if wb_norm == norm:
                        return _wb
                except Exception:
                    continue
            for _wb in list(xl.Workbooks):
                try:
                    wb_base = os.path.basename(_wb.FullName).lower()
                    log.debug('[COM] _com_find_workbook: GAO name-only: %s vs %s',
                              wb_base, basename_lower)
                    if wb_base == basename_lower:
                        return _wb
                except Exception:
                    continue
        except Exception as exc:
            log.warning('[COM] _com_find_workbook: GetActiveObject failed: %s', exc)
            # Fall through to HWND scan — Excel may be open but not in the ROT.

        # Pass 3 — HWND + AccessibleObjectFromWindow.
        # Works when Excel was opened via os.startfile / shell association and
        # never registered in the COM ROT.  Standard approach used by xlwings.
        try:
            import ctypes
            import win32gui  # type: ignore[import]

            class _GUID(ctypes.Structure):
                _fields_ = [('Data1', ctypes.c_uint32), ('Data2', ctypes.c_uint16),
                             ('Data3', ctypes.c_uint16), ('Data4', ctypes.c_ubyte * 8)]

            # IID_IDispatch = {00020400-0000-0000-C000-000000000046}
            _iid = _GUID(0x00020400, 0, 0,
                         (ctypes.c_ubyte * 8)(0xC0, 0, 0, 0, 0, 0, 0, 0x46))
            _OBJID_NATIVEOM = ctypes.c_ulong(0xFFFFFFF0)

            _xl_hwnds: list = []

            def _enum_top(hwnd, _):
                try:
                    if win32gui.GetClassName(hwnd) == 'XLMAIN':
                        _xl_hwnds.append(hwnd)
                except Exception:
                    pass

            win32gui.EnumWindows(_enum_top, None)
            log.debug('[COM] _com_find_workbook: HWND found %d XLMAIN windows', len(_xl_hwnds))

            for _xl_hwnd in _xl_hwnds:
                try:
                    _desk = win32gui.FindWindowEx(_xl_hwnd, 0, 'XLDESK', None)
                    if not _desk:
                        continue
                    _e7 = win32gui.FindWindowEx(_desk, 0, 'EXCEL7', None)
                    while _e7:
                        try:
                            _ptr = ctypes.c_void_p()
                            _hr = ctypes.windll.oleacc.AccessibleObjectFromWindow(
                                _e7, _OBJID_NATIVEOM,
                                ctypes.byref(_iid), ctypes.byref(_ptr))
                            if _hr == 0 and _ptr.value:
                                _xl_sheet = win32com.client.Dispatch(
                                    pythoncom.ObjectFromAddress(
                                        _ptr.value, pythoncom.IID_IDispatch))
                                _xl_wb = _xl_sheet.Parent   # Sheet → Workbook
                                _wb_norm = os.path.normcase(
                                    os.path.normpath(_xl_wb.FullName))
                                log.debug('[COM] _com_find_workbook: HWND wb=%s', _wb_norm)
                                if (_wb_norm == norm or
                                        os.path.basename(_wb_norm).lower() == basename_lower):
                                    _com_wb_cache[norm] = _xl_wb
                                    log.debug('[COM] _com_find_workbook: HWND match+cached %s',
                                              abs_path)
                                    return _xl_wb
                        except Exception as _e:
                            log.debug('[COM] _com_find_workbook: HWND e7=%d err=%s', _e7, _e)
                        _e7 = win32gui.FindWindowEx(_desk, _e7, 'EXCEL7', None)
                except Exception as _e:
                    log.debug('[COM] _com_find_workbook: HWND xlmain=%d err=%s',
                              _xl_hwnd, _e)
        except Exception as exc:
            log.debug('[COM] _com_find_workbook: HWND scan error: %s', exc)

    except Exception as exc:
        log.warning('[COM] _com_find_workbook: unexpected error: %s', exc)
    return None


def _com_live_append_row(abs_path: str, values: list,
                          sheet_name: str | None = None) -> dict | None:
    """Append *values* as a new row in the live Excel workbook via COM and save.

    Returns a result dict on success, or None when COM is unavailable or the
    workbook is not open in Excel.  Callers fall through to the openpyxl path
    on None so closed-workbook behaviour is unchanged.
    """
    try:
        import pythoncom          # type: ignore[import]
        try:
            pythoncom.CoInitialize()
        except Exception:
            pass
        xl_wb = _com_find_workbook(abs_path)
        if xl_wb is None:
            return None

        xl = xl_wb.Application
        xl.DisplayAlerts = False

        xl_ws = xl_wb.Worksheets(sheet_name) if sheet_name else xl_wb.ActiveSheet

        # Locate the last used row so the new row is appended immediately after.
        # UsedRange.Row is the 1-based start of the used area; adding Rows.Count
        # and subtracting 1 gives the absolute last row with data.
        used = xl_ws.UsedRange
        last_row = used.Row + used.Rows.Count - 1
        next_row = last_row + 1

        if values:
            ncols = len(values)
            rng = xl_ws.Range(xl_ws.Cells(next_row, 1), xl_ws.Cells(next_row, ncols))
            rng.Value = [tuple(values)]

        xl_wb.Save()
        log.info('[COM] live append: row=%d  values=%s  sheet=%s  path=%s',
                 next_row, values, sheet_name or 'active', abs_path)

        summary = ', '.join(str(v) for v in values[:4])
        if len(values) > 4:
            summary += f', and {len(values) - 4} more'
        return {
            'message': 'row_added', 'values': values,
            'speech_text': f"Done! I've added row {next_row} with: {summary}.",
        }
    except Exception as exc:
        log.warning('[COM] _com_live_append_row failed — falling back to openpyxl path: %s', exc)
        return None


def _com_live_write_cell(abs_path: str, row: int, col: int, value,
                          sheet_name: str | None = None) -> bool:
    """Write *value* to cell (row, col) in the live Excel workbook via COM and save.

    Formula strings (starting with '=') are written via .Formula so Excel
    evaluates them; all other values are written via .Value.

    Returns True on success, False when COM is unavailable or the workbook is
    not open in Excel (caller falls through to openpyxl path).
    """
    try:
        import pythoncom          # type: ignore[import]
        try:
            pythoncom.CoInitialize()
        except Exception:
            pass
        xl_wb = _com_find_workbook(abs_path)
        if xl_wb is None:
            return False

        xl = xl_wb.Application
        xl.DisplayAlerts = False

        xl_ws = xl_wb.Worksheets(sheet_name) if sheet_name else xl_wb.ActiveSheet
        cell_obj = xl_ws.Cells(row, col)
        if isinstance(value, str) and value.startswith('='):
            cell_obj.Formula = value
        else:
            cell_obj.Value = value
        xl_wb.Save()

        log.info('[COM] live write: (%d,%d)=%r  sheet=%s  path=%s',
                 row, col, value, sheet_name or 'active', abs_path)
        return True
    except Exception as exc:
        log.warning('[COM] _com_live_write_cell failed — falling back to openpyxl path: %s', exc)
        return False


def _com_live_format_cells(abs_path: str, cell_range: str,
                            bold=None, italic=None, font_size=None,
                            font_color=None, fill_color=None,
                            num_format=None,
                            align=None, font_size_change=None,
                            sheet_name=None) -> dict | None:
    """Apply cell formatting in the live Excel workbook via COM and save.

    Returns a result dict on success, or None when COM is unavailable or the
    workbook is not open in Excel (caller falls through to openpyxl path).
    Never raises — all COM exceptions are caught and logged.
    """
    try:
        import pythoncom          # type: ignore[import]
        try:
            pythoncom.CoInitialize()
        except Exception:
            pass
        xl_wb = _com_find_workbook(abs_path)
        if xl_wb is None:
            return None

        xl = xl_wb.Application
        xl.DisplayAlerts = False
        xl_ws = xl_wb.Worksheets(sheet_name) if sheet_name else xl_wb.ActiveSheet
        rng = xl_ws.Range(cell_range)

        if bold is not None:
            rng.Font.Bold = bool(bold)
        if italic is not None:
            rng.Font.Italic = bool(italic)
        if font_size is not None:
            try:
                rng.Font.Size = int(font_size)
            except (TypeError, ValueError):
                pass
        if font_size_change is not None:
            try:
                cur = rng.Font.Size or 11
                rng.Font.Size = max(6, int(cur) + int(font_size_change))
            except (TypeError, ValueError):
                pass
        if font_color is not None:
            c = str(font_color).lstrip('#')
            if len(c) == 6:
                r_, g_, b_ = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
                rng.Font.Color = r_ | (g_ << 8) | (b_ << 16)
        if fill_color is not None:
            c = str(fill_color).lstrip('#')
            if len(c) == 6:
                r_, g_, b_ = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
                rng.Interior.Color = r_ | (g_ << 8) | (b_ << 16)
        if align is not None:
            _HALIGN_COM = {'left': -4131, 'center': -4108, 'centre': -4108, 'right': -4152}
            _hv = _HALIGN_COM.get(str(align).lower().strip())
            if _hv is not None:
                rng.HorizontalAlignment = _hv
        if num_format is not None:
            rng.NumberFormat = str(num_format)

        xl_wb.Save()
        log.info('[COM] live format: range=%s bold=%s italic=%s align=%s num_format=%s path=%s',
                 cell_range, bold, italic, align, num_format, abs_path)
        return {'_com_used': True}
    except Exception as exc:
        log.warning('[COM] _com_live_format_cells failed: %s', exc)
        return None


def _com_live_unmerge_cells(abs_path: str, cell_range: str, sheet_name=None) -> dict | None:
    """Unmerge *cell_range* in the live Excel workbook via COM and save.

    Returns a result dict on success, or None when COM is unavailable or the
    workbook is not open in Excel (caller falls through to openpyxl path).
    Never raises — all COM exceptions are caught and logged.
    """
    try:
        import pythoncom          # type: ignore[import]
        try:
            pythoncom.CoInitialize()
        except Exception:
            pass
        xl_wb = _com_find_workbook(abs_path)
        if xl_wb is None:
            return None

        xl = xl_wb.Application
        xl.DisplayAlerts = False
        xl_ws = xl_wb.Worksheets(sheet_name) if sheet_name else xl_wb.ActiveSheet
        xl_ws.Range(cell_range).UnMerge()

        xl_wb.Save()
        log.info('[COM] live unmerge: range=%s  sheet=%s  path=%s',
                 cell_range, sheet_name, abs_path)
        return {'_com_used': True}
    except Exception as exc:
        log.warning('[COM] _com_live_unmerge_cells failed: %s', exc)
        return None


def _com_live_filter_table(abs_path: str, col_idx: int, operator: str, value,
                           sheet_name=None) -> dict | None:
    """Apply an Excel AutoFilter on the live workbook via COM.

    Returns a result dict on success, or None when COM is unavailable or the
    workbook is not open in Excel (caller falls through to openpyxl path).
    Never raises — all COM exceptions are caught and logged.
    """
    try:
        import pythoncom          # type: ignore[import]
        try:
            pythoncom.CoInitialize()
        except Exception:
            pass
        xl_wb = _com_find_workbook(abs_path)
        if xl_wb is None:
            return None

        xl = xl_wb.Application
        xl.DisplayAlerts = False
        xl_ws = xl_wb.Worksheets(sheet_name) if sheet_name else xl_wb.ActiveSheet

        # Build AutoFilter criteria string.
        # Use value directly (no '=' prefix) for equality — avoids type mismatch
        # when matching numeric cells.  Other operators use their symbol prefix.
        op = (operator or '=').lower().strip()
        val_str = str(value) if value is not None else ''
        if op == '=':
            criteria = val_str           # exact match: "90" matches numeric 90
        elif op in ('!=', '<>'):
            criteria = f'<>{val_str}'
        elif op == 'contains':
            criteria = f'*{val_str}*'
        else:
            criteria = f'{op}{val_str}'  # >, <, >=, <=

        # Apply AutoFilter directly — enables it if not active, updates criteria
        # if already active.  Do NOT reset AutoFilterMode first: clearing it then
        # re-applying in two steps can raise COM errors on some Excel versions.
        xl_ws.Range('A1').AutoFilter(Field=col_idx, Criteria1=criteria)

        xl_wb.Save()
        log.info('[COM] live filter: col_idx=%s op=%s val=%s sheet=%s path=%s',
                 col_idx, operator, value, sheet_name, abs_path)
        return {'_com_used': True}
    except Exception as exc:
        log.warning('[COM] _com_live_filter_table failed: %s', exc)
        return None


def _com_live_merge_cells(abs_path: str, cell_range: str, sheet_name=None) -> dict | None:
    """Merge *cell_range* in the live Excel workbook via COM and save.

    Returns a result dict on success, or None when COM is unavailable or the
    workbook is not open in Excel (caller falls through to openpyxl path).
    Never raises — all COM exceptions are caught and logged.
    """
    try:
        import pythoncom          # type: ignore[import]
        try:
            pythoncom.CoInitialize()
        except Exception:
            pass
        xl_wb = _com_find_workbook(abs_path)
        if xl_wb is None:
            return None

        xl = xl_wb.Application
        xl.DisplayAlerts = False
        xl_ws = xl_wb.Worksheets(sheet_name) if sheet_name else xl_wb.ActiveSheet
        xl_ws.Range(cell_range).Merge()

        xl_wb.Save()
        log.info('[COM] live merge: range=%s  sheet=%s  path=%s',
                 cell_range, sheet_name, abs_path)
        return {'_com_used': True}
    except Exception as exc:
        log.warning('[COM] _com_live_merge_cells failed: %s', exc)
        return None


def _com_live_sort_range(abs_path: str, col_idx: int, ascending: bool,
                         header_row: bool = True, sheet_name=None) -> dict | None:
    """Sort the used range by column *col_idx* in the live Excel workbook via COM.

    Returns a result dict on success, or None when COM is unavailable or the
    workbook is not open in Excel (caller falls through to openpyxl path).
    Never raises — all COM exceptions are caught and logged.
    """
    try:
        import pythoncom          # type: ignore[import]
        try:
            pythoncom.CoInitialize()
        except Exception:
            pass
        xl_wb = _com_find_workbook(abs_path)
        if xl_wb is None:
            return None

        xl = xl_wb.Application
        xl.DisplayAlerts = False
        xl_ws = xl_wb.Worksheets(sheet_name) if sheet_name else xl_wb.ActiveSheet

        sort_order = 1 if ascending else 2  # xlAscending=1, xlDescending=2
        header    = 1 if header_row else 2  # xlYes=1, xlNo=2

        xl_ws.UsedRange.Sort(
            Key1=xl_ws.Cells(1, col_idx),
            Order1=sort_order,
            Header=header,
        )
        xl_wb.Save()
        log.info('[COM] live sort: col_idx=%s asc=%s sheet=%s path=%s',
                 col_idx, ascending, sheet_name, abs_path)
        return {'_com_used': True}
    except Exception as exc:
        log.warning('[COM] _com_live_sort_range failed: %s', exc)
        return None


def _session_reload_from_disk(abs_path: str) -> None:
    """Reload the openpyxl session from disk after a COM save.

    Keeps the in-memory workbook in sync with what Excel wrote so the next
    undo snapshot captures the correct post-change state and subsequent
    openpyxl reads are not stale.
    """
    global _session
    with _session_lock:
        if (_session is None or
                os.path.normcase(_session.abs_path) != os.path.normcase(abs_path)):
            return
        try:
            wb = load_workbook(abs_path)
            _session.workbook = wb
            _session.dirty = False
            log.info('[Session] reloaded from disk after COM save: %s', abs_path)
        except Exception as exc:
            log.warning('[Session] _session_reload_from_disk failed: %s', exc)
            _session.dirty = False   # prevent stale in-memory data from clobbering disk


def ensure_workbook(path, sheet_name=None):
    """Return the active workbook and worksheet, loading from disk only when needed.

    Cache behaviour
    ---------------
    * **Cache hit**: when *path* matches the current session, the in-memory
      ``Workbook`` object is returned directly — no ``load_workbook()`` call.
    * **Cache miss**: when *path* differs, the current session is auto-flushed
      (if dirty) before the new file is loaded and stored as the new session.
    * **No path + no session**: ``_auto_open_default()`` creates/loads
      ``sara_default.xlsx`` automatically so modifying commands never fail
      silently for lack of an open file.

    Sheet behaviour
    ---------------
    If *sheet_name* is given the named sheet is returned instead of the
    active sheet.  An error dict is returned when the sheet does not exist;
    callers guard with ``if isinstance(ws, dict): return ws``.
    """
    global _session

    with _session_lock:
        resolved = os.path.abspath(path) if path else None

        # No explicit path — fall back to active session or auto-create default.
        if not resolved:
            if _session is None:
                _auto_open_default()   # sets _session; called inside lock
            if _session is None:
                return None, {
                    "error": "no_active_file",
                    "speech_text": "No Excel file is open. Please open or create a file first.",
                }
            resolved = _session.abs_path

        norm = os.path.normcase(resolved)

        if _session is not None and os.path.normcase(_session.abs_path) == norm:
            # Cache hit — reuse the live workbook object.
            log.debug('[Session] cache hit: %s', resolved)
            wb = _session.workbook
        else:
            # Cache miss — flush old session (if dirty) and load new one.
            if _session is not None and _session.dirty:
                try:
                    _atomic_save(_session.workbook, _session.abs_path)
                    _session.dirty = False
                    log.info('[Session] auto-flushed before switch: %s', _session.abs_path)
                except Exception as exc:
                    log.warning('[Session] auto-flush failed: %s', exc)

            if os.path.exists(resolved):
                wb = load_workbook(resolved)
            else:
                wb = Workbook()
            _session = _WorkbookSession(resolved, wb)
            log.info('[Session] loaded: %s', resolved)

    # Sheet selection is done outside the lock (wb is now a local reference).
    if sheet_name:
        resolved_sn = _resolve_sheet_name(wb, sheet_name)
        if resolved_sn is None:
            return wb, {"error": "sheet_not_found", "sheet": sheet_name,
                        "available": wb.sheetnames}
        ws = wb[resolved_sn]
    else:
        ws = wb.active
    return wb, ws


def open_file_session(path: str) -> dict:
    """Point the session at *path* without loading the workbook.

    Used by ``state_manager`` and other callers that need to record a file
    switch before the first actual command arrives.  The workbook is loaded
    lazily on the first ``ensure_workbook`` call.
    """
    global _session
    abs_path = os.path.abspath(path)
    with _session_lock:
        prev = _session.abs_path if _session else None
        # Only replace the session when the path actually changes.
        if prev != abs_path:
            if _session is not None and _session.dirty:
                try:
                    _atomic_save(_session.workbook, _session.abs_path)
                except Exception as exc:
                    log.warning('[Session] flush on switch failed: %s', exc)
            # Load from disk if the file exists, otherwise create in memory.
            if os.path.exists(abs_path):
                try:
                    wb = load_workbook(abs_path)
                except Exception:
                    wb = Workbook()
            else:
                wb = Workbook()
            _session = _WorkbookSession(abs_path, wb)
            log.info('[Session] %s → %s', prev or '(none)', abs_path)

    exists = os.path.exists(abs_path)
    return {
        "message": "session_opened",
        "path": abs_path,
        "exists": exists,
        "speech_text": (
            f"{'Opened' if exists else 'Ready to create'} {os.path.basename(abs_path)}."
        ),
    }


def close_file_session() -> dict:
    """Flush any dirty changes and clear the active session.

    After this call, path-less commands will trigger ``_auto_open_default``.
    """
    global _session
    with _session_lock:
        if _session is None:
            return {"message": "session_already_closed"}
        prev = _session.abs_path
        if _session.dirty:
            try:
                _atomic_save(_session.workbook, _session.abs_path)
            except Exception as exc:
                log.warning('[Session] flush on close failed: %s', exc)
        _session = None
    log.info('[Session] closed: %s', prev)
    return {"message": "session_closed", "path": prev}


def get_active_path() -> str | None:
    """Return the absolute path of the currently active file, or ``None``."""
    with _session_lock:
        return _session.abs_path if _session else None


def _ws_headers(ws) -> list[str]:
    """Return row-1 header values from *ws* as strings.

    Stops at the first empty cell so trailing None columns are excluded.
    Used internally by open_excel, switch_sheet, and get_sheet_headers to
    embed the current column list in result dicts — enabling _update_memory
    to keep memory accurate without a second workbook read.
    """
    headers: list[str] = []
    for c in range(1, ws.max_column + 1):
        val = ws.cell(row=1, column=c).value
        if val is None:
            break
        headers.append(str(val))
    return headers


def get_sheet_headers(path: str, sheet_name: str | None = None) -> list[str]:
    """Return row-1 header values from the active or named sheet.

    Uses the in-memory session when available (no extra disk I/O on a cache
    hit).  Returns an empty list on any error so callers never need to guard.

    Used by agent._update_memory to reload headers after undo/undo_last.
    """
    try:
        wb, ws = ensure_workbook(path, sheet_name)
        if isinstance(ws, dict):
            return []
        return _ws_headers(ws)
    except Exception:
        return []


def _is_file_locked(path):
    try:
        with open(path, "a"):
            return False
    except PermissionError:
        return True


def _poll_for_unlock(path: str) -> bool:
    """Block until *path* is writable or ``_LOCK_RELEASE_TIMEOUT`` expires.

    Polls at ``_LOCK_POLL_INTERVAL`` intervals so the caller proceeds the
    instant the OS releases the handle — no wasted time from a fixed sleep.
    Returns True when the file is confirmed unlocked, False on timeout.
    """
    deadline = time.monotonic() + _LOCK_RELEASE_TIMEOUT
    while time.monotonic() < deadline:
        if not _is_file_locked(path):
            return True
        time.sleep(_LOCK_POLL_INTERVAL)
    log.warning('[Session] _poll_for_unlock: %s still locked after %.1f s — '
                'proceeding anyway', path, _LOCK_RELEASE_TIMEOUT)
    return False


def create_excel(path, headers=None, sheet_name=None):
    """Create a new Excel file optionally with headers.

    If *sheet_name* is provided the default sheet is renamed to that name.
    """
    global _session
    log.info('create_excel: path=%s  headers=%s  sheet=%s', path, headers, sheet_name)
    filename = os.path.basename(path)
    if os.path.exists(path):
        if _is_file_locked(path):
            return {
                "speech_text": f"{filename} is currently open in Excel. Please close it before creating a new workbook with the same name."
            }
        else:
            return {
                "speech_text": f"{filename} already exists. Please choose another file name."
            }
    if headers is not None and not isinstance(headers, (list, tuple)):
        return {"error": "headers must be a list"}
    wb = Workbook()
    ws = wb.active
    if sheet_name:
        ws.title = sheet_name
    if headers:
        ws.append(list(headers))
    abs_path = os.path.abspath(path)
    # Write to disk once so the file exists for COM automation.
    _atomic_save(wb, abs_path)
    # Seed the session with the workbook we just built — no reload needed.
    with _session_lock:
        _session = _WorkbookSession(abs_path, wb)
        # dirty=False: the on-disk file is already current
    # Open in Excel so the user sees the new file immediately.
    _launch_excel_com(abs_path)
    col_count = len(headers) if headers else 0
    if col_count:
        col_names = ', '.join(str(h) for h in headers)
        speech = f"All set! I've created a new spreadsheet with {col_count} column{'s' if col_count != 1 else ''}: {col_names}."
    else:
        speech = "All set! I've created a new blank spreadsheet."
    return {"message": "created", "path": path, "active_sheet": ws.title, "speech_text": speech}


def add_row(path, values, sheet_name=None):
    """Append a row of values to the workbook."""
    log.info('add_row: path=%s  values=%s  sheet=%s', path, values, sheet_name)
    if not isinstance(values, (list, tuple)):
        return {"error": "values must be a list"}
    # Auto-convert numeric strings so Excel stores them as numbers
    coerced = []
    for v in values:
        if isinstance(v, str):
            try:
                coerced.append(int(v))
                continue
            except ValueError:
                pass
            try:
                coerced.append(float(v))
                continue
            except ValueError:
                pass
        coerced.append(v)
    # Prevent formula injection: neutralise any string that starts with =, +, -, @
    safe = [sanitize_cell_value(v) for v in coerced]
    _undo_snapshot(path, f"add row {safe[:2]}")
    wb, ws = ensure_workbook(path, sheet_name)
    if isinstance(ws, dict):
        return ws

    # ── COM live-write: workbook open in Excel ────────────────────────────
    # Bypasses the close→atomic-replace→reopen path that fails with
    # WinError 32/5 when Excel's file lock does not release within the poll
    # window.  On success, reload the openpyxl session from what COM saved.
    abs_path = os.path.abspath(path)
    _com_result = _com_live_append_row(abs_path, safe, sheet_name)
    if _com_result is not None:
        log.info('[Controller] add_row: COM live path used for %s', abs_path)
        _session_reload_from_disk(abs_path)
        return _com_result

    # Safety gate: if the file is still locked by Excel, openpyxl + _atomic_save
    # will crash with WinError 5 / "Access is denied".  Return a controlled error
    # instead of falling through.  This catches the case where _com_find_workbook
    # failed to detect the workbook (e.g. an Office version that does not register
    # in the ROT) but Excel still holds the write lock.
    if _is_file_locked(abs_path):
        log.error('[Controller] add_row: file locked by Excel but COM write failed for %s — '
                  'refusing openpyxl fallback', abs_path)
        return {
            'error': 'com_write_failed',
            'message': 'row_not_added',
            'speech_text': 'The file is open in Excel but I could not write to it. Please try again.',
        }

    # ── Openpyxl path: workbook not open in Excel ─────────────────────────
    log.info('[Controller] add_row: openpyxl path for %s', path)
    ws.append(safe)
    _mark_dirty()
    _flush_session(path)
    row_num = ws.max_row
    summary = ', '.join(str(v) for v in safe[:4])
    if len(safe) > 4:
        summary += f', and {len(safe) - 4} more'
    return {"message": "row_added", "values": safe,
            "speech_text": f"Done! I've added row {row_num} with: {summary}."}


def delete_row(path, index, sheet_name=None):
    """Delete a row by 1-based index."""
    log.info('delete_row: path=%s  index=%s  sheet=%s', path, index, sheet_name)
    try:
        index = int(index)
    except (TypeError, ValueError):
        return {"error": "index must be an integer"}
    wb, ws = ensure_workbook(path, sheet_name)
    if isinstance(ws, dict):
        return ws
    max_row = ws.max_row
    if index < 1 or index > max_row:
        return {"error": "index_out_of_range", "max_row": max_row}
    if index == 1:
        return {
            "error": "header_row_protected",
            "speech_text": "Row 1 is the header row and can't be deleted. Did you mean a different row?",
        }
    # Snapshot BEFORE deleting so the row can be restored by undo_last()
    _undo_snapshot(path, f"delete row {index}")

    # ── COM live path: avoid WinError 5 when file is open in Excel ───────────
    abs_path = os.path.abspath(path)
    xl_wb = _com_find_workbook(abs_path)
    if xl_wb is not None:
        try:
            xl_wb.Application.DisplayAlerts = False
            xl_ws = xl_wb.Worksheets(sheet_name) if sheet_name else xl_wb.ActiveSheet
            xl_ws.Rows(index).Delete()
            xl_wb.Save()
            log.info('[COM] delete_row: deleted row %d via COM', index)
        except Exception as exc:
            log.warning('[COM] delete_row failed: %s', exc)
            return {"error": "delete_row_com_failed", "detail": str(exc),
                    "speech_text": "The file is open in Excel but I could not delete the row. Please try again."}
        _session_reload_from_disk(abs_path)
        return {
            "message": "row_deleted",
            "index": index,
            "speech_text": f"Done! Row {index} has been deleted. Say 'undo' if you want to restore it.",
        }

    if _is_file_locked(abs_path):
        return {
            'error': 'com_write_failed',
            'speech_text': 'The file is open in Excel but I could not write to it. Please try again.',
        }

    # ── openpyxl path (file NOT open in Excel) ────────────────────────────
    ws.delete_rows(index)
    _mark_dirty()
    _flush_session(path)
    return {
        "message": "row_deleted",
        "index": index,
        "speech_text": f"Done! Row {index} has been deleted. Say 'undo' if you want to restore it.",
    }


def insert_row_at(path, index, sheet_name=None):
    """Insert a single blank row at 1-based *index*, shifting existing rows down.

    Row 1 (header row) is protected. Inserting at max_row+1 is allowed.
    Inserting inside a merged range is rejected to prevent corrupt merge metadata.
    """
    log.info('insert_row_at: path=%s  index=%s  sheet=%s', path, index, sheet_name)
    try:
        index = int(index)
    except (TypeError, ValueError):
        return {"error": "index must be an integer"}
    if index < 2:
        return {
            "error": "header_row_protected",
            "speech_text": "Row 1 is the header row. Use 'insert row 2' or higher to insert below it.",
        }
    wb, ws = ensure_workbook(path, sheet_name)
    if isinstance(ws, dict):
        return ws
    max_row = ws.max_row
    if index > max_row + 1:
        return {"error": "index_out_of_range",
                "speech_text": f"Row {index} is beyond the sheet (max {max_row + 1})."}
    # Reject if the insertion point cuts through a merged range.
    for mr in ws.merged_cells.ranges:
        if mr.min_row < index <= mr.max_row:
            return {
                "error": "merge_conflict",
                "speech_text": f"Cannot insert row {index} — it falls inside merged range {mr}. Unmerge first.",
            }
    _undo_snapshot(path, f"insert row at {index}")
    ws.insert_rows(index)
    _mark_dirty()
    _flush_session(path)
    return {
        "message": "row_inserted",
        "index": index,
        "speech_text": f"Done! A blank row has been inserted at row {index}. Existing rows shifted down.",
    }


def insert_column_at(path, column, sheet_name=None):
    """Insert a single blank column at position *column* (letter or 1-based int),
    shifting existing columns right.

    Inserting inside a merged range is rejected to prevent corrupt merge metadata.
    """
    log.info('insert_column_at: path=%s  col=%s  sheet=%s', path, column, sheet_name)
    if not column:
        return {"error": "column identifier is required"}
    col_str = str(column).strip().upper()
    try:
        if col_str.isalpha():
            from openpyxl.utils import column_index_from_string
            col_idx = column_index_from_string(col_str)
        else:
            col_idx = int(col_str)
    except Exception:
        return {"error": "invalid_column", "speech_text": f"Cannot resolve column '{column}'."}
    if col_idx < 1:
        return {"error": "invalid_column", "speech_text": "Column index must be >= 1."}
    wb, ws = ensure_workbook(path, sheet_name)
    if isinstance(ws, dict):
        return ws
    max_col = ws.max_column
    if col_idx > max_col + 1:
        return {"error": "column_out_of_range",
                "speech_text": f"Column {column} is beyond the sheet (max {get_column_letter(max_col + 1)})."}
    # Reject if the insertion point cuts through a merged range.
    for mr in ws.merged_cells.ranges:
        if mr.min_col < col_idx <= mr.max_col:
            return {
                "error": "merge_conflict",
                "speech_text": f"Cannot insert column {column} — it falls inside merged range {mr}. Unmerge first.",
            }
    _undo_snapshot(path, f"insert column at {col_str}")
    ws.insert_cols(col_idx)
    _mark_dirty()
    _flush_session(path)
    col_letter = get_column_letter(col_idx)
    return {
        "message": "column_inserted",
        "column": col_letter,
        "speech_text": f"Done! A blank column has been inserted at column {col_letter}. Existing columns shifted right.",
    }


def delete_column(path, col, sheet_name=None):
    """Delete a column by letter (e.g., 'B') or 1-based index.

    Returns info including number of rows affected and a speech_text summary.
    """
    log.info('delete_column: path=%s  col=%s  sheet=%s', path, col, sheet_name)
    if col is None:
        return {"error": "column identifier is required"}
    wb, ws = ensure_workbook(path, sheet_name)
    if isinstance(ws, dict):
        return ws
    # accept letter or integer
    if isinstance(col, str):
        col = col.strip().upper()
        if col.isalpha():
            col_idx = _col_letters_to_index(col)
        else:
            try:
                col_idx = int(col)
            except Exception:
                return {"error": "invalid_column", "col": col}
    else:
        col_idx = int(col)

    max_col = ws.max_column
    if col_idx < 1 or col_idx > max_col:
        return {"error": "column_out_of_range", "max_col": max_col}

    # Count non-empty cells in column (below header)
    rows = ws.max_row
    non_empty = 0
    for r in range(1, rows + 1):
        val = ws.cell(row=r, column=col_idx).value
        if val is not None and str(val).strip() != '':
            non_empty += 1

    # Snapshot BEFORE deleting so the column can be restored by undo_last()
    _undo_snapshot(path, f"delete column {get_column_letter(col_idx)}")

    # ── COM live path: avoid WinError 5 when file is open in Excel ───────────
    abs_path = os.path.abspath(path)
    xl_wb = _com_find_workbook(abs_path)
    if xl_wb is not None:
        try:
            xl_wb.Application.DisplayAlerts = False
            xl_ws = xl_wb.Worksheets(sheet_name) if sheet_name else xl_wb.ActiveSheet
            xl_ws.Columns(col_idx).Delete()
            xl_wb.Save()
            col_letter = get_column_letter(col_idx)
            log.info('[COM] delete_column: deleted column %s via COM', col_letter)
        except Exception as exc:
            log.warning('[COM] delete_column failed: %s', exc)
            return {"error": "delete_column_com_failed", "detail": str(exc),
                    "speech_text": "The file is open in Excel but I could not delete the column. Please try again."}
        _session_reload_from_disk(abs_path)
        col_letter = get_column_letter(col_idx)
        return {
            "message": "column_deleted",
            "column": col_letter,
            "rows_affected": non_empty,
            "speech_text": (
                f"Done! Column {col_letter} has been removed ({non_empty} cells cleared). "
                "Say 'undo' if you want to restore it."
            ),
        }

    if _is_file_locked(abs_path):
        return {
            'error': 'com_write_failed',
            'speech_text': 'The file is open in Excel but I could not write to it. Please try again.',
        }

    # ── openpyxl path (file NOT open in Excel) ────────────────────────────
    ws.delete_cols(col_idx)
    _mark_dirty()
    _flush_session(path)
    col_letter = get_column_letter(col_idx)
    return {
        "message": "column_deleted",
        "column": col_letter,
        "rows_affected": non_empty,
        "speech_text": (
            f"Done! Column {col_letter} has been removed ({non_empty} cells cleared). "
            "Say 'undo' if you want to restore it."
        ),
    }


def delete_cell(path, cell, sheet_name=None):
    """Clear the value of a single cell (e.g. 'A1') without deleting the row or column."""
    log.info('delete_cell: path=%s  cell=%s  sheet=%s', path, cell, sheet_name)
    if not cell:
        return {"error": "cell reference is required"}
    cell = str(cell).upper().strip()
    m = _CELL_RE.match(cell)
    if not m:
        return {"error": "invalid_cell", "cell": cell}
    col_letters, row = m.group(1), int(m.group(2))
    col_idx = _col_letters_to_index(col_letters)
    _undo_snapshot(path, f"delete cell {cell}")
    wb, ws = ensure_workbook(path, sheet_name)
    if isinstance(ws, dict):
        return ws
    # ── COM live path: avoid WinError 5 when file is open in Excel ───────────
    abs_path = os.path.abspath(path)
    xl_wb = _com_find_workbook(abs_path)
    if xl_wb is not None:
        try:
            xl_wb.Application.DisplayAlerts = False
            xl_ws = xl_wb.Worksheets(sheet_name) if sheet_name else xl_wb.ActiveSheet
            xl_ws.Cells(row, col_idx).ClearContents()
            xl_wb.Save()
            log.info('[COM] delete_cell: cleared %s via COM', cell)
        except Exception as exc:
            log.warning('[COM] delete_cell failed: %s', exc)
            return {"error": "delete_cell_com_failed", "detail": str(exc),
                    "speech_text": "The file is open in Excel but I could not clear the cell. Please try again."}
        _session_reload_from_disk(abs_path)
        return {
            "message": "cell_deleted",
            "cell": cell,
            "speech_text": f"Done! I've cleared cell {cell}. Say 'undo' to restore it.",
        }

    if _is_file_locked(abs_path):
        return {
            'error': 'com_write_failed',
            'speech_text': 'The file is open in Excel but I could not write to it. Please try again.',
        }

    # ── openpyxl path (file NOT open in Excel) ────────────────────────────
    ws.cell(row=row, column=col_idx).value = None
    _mark_dirty()
    _flush_session(path)
    return {
        "message": "cell_deleted",
        "cell": cell,
        "speech_text": f"Done! I've cleared cell {cell}. Say 'undo' to restore it.",
    }


def delete_range(path, cell_range, sheet_name=None):
    """Clear the values of all cells in *cell_range* (e.g. 'A2:C4')."""
    log.info('delete_range: path=%s  range=%s  sheet=%s', path, cell_range, sheet_name)
    if not cell_range:
        return {"error": "cell_range is required"}
    cell_range = str(cell_range).upper().strip()
    _undo_snapshot(path, f"delete range {cell_range}")
    wb, ws = ensure_workbook(path, sheet_name)
    if isinstance(ws, dict):
        return ws
    try:
        target = ws[cell_range]
    except Exception as exc:
        return {"error": "invalid_range", "detail": str(exc)}
    # ws[range] returns Cell, tuple-of-Cell, or tuple-of-tuples
    if not isinstance(target, tuple):
        target = ((target,),)
    cells_cleared = sum(
        1 for row in target for cell in (row if isinstance(row, tuple) else (row,))
    )

    # ── COM live path: avoid WinError 5 when file is open in Excel ───────────
    abs_path = os.path.abspath(path)
    xl_wb = _com_find_workbook(abs_path)
    if xl_wb is not None:
        try:
            xl_wb.Application.DisplayAlerts = False
            xl_ws = xl_wb.Worksheets(sheet_name) if sheet_name else xl_wb.ActiveSheet
            xl_ws.Range(cell_range).ClearContents()
            xl_wb.Save()
            log.info('[COM] delete_range: cleared %s via COM', cell_range)
        except Exception as exc:
            log.warning('[COM] delete_range failed: %s', exc)
            return {"error": "delete_range_com_failed", "detail": str(exc),
                    "speech_text": "The file is open in Excel but I could not clear the range. Please try again."}
        _session_reload_from_disk(abs_path)
        return {
            "message": "range_deleted",
            "range": cell_range,
            "cells_cleared": cells_cleared,
            "speech_text": f"Done! I've cleared {cells_cleared} cell(s) in {cell_range}. Say 'undo' to restore.",
        }

    if _is_file_locked(abs_path):
        return {
            'error': 'com_write_failed',
            'speech_text': 'The file is open in Excel but I could not write to it. Please try again.',
        }

    # ── openpyxl path (file NOT open in Excel) ────────────────────────────
    for row in target:
        for cell in (row if isinstance(row, tuple) else (row,)):
            cell.value = None
    _mark_dirty()
    _flush_session(path)
    return {
        "message": "range_deleted",
        "range": cell_range,
        "cells_cleared": cells_cleared,
        "speech_text": f"Done! I've cleared {cells_cleared} cell(s) in {cell_range}. Say 'undo' to restore.",
    }


def add_column(path, column_name, default_value=None, sheet_name=None):
    """Add a new column at the end and set its header to column_name."""
    log.info('add_column: path=%s  name=%s  sheet=%s', path, column_name, sheet_name)
    if not column_name or not isinstance(column_name, str):
        return {"error": "column_name must be a non-empty string"}
    _undo_snapshot(path, f"add column {column_name}")
    wb, ws = ensure_workbook(path, sheet_name)
    if isinstance(ws, dict):
        return ws
    # openpyxl reports max_column=1 even on a blank sheet, so we cannot rely
    # on it directly.  Scan row 1 for the last occupied header cell instead;
    # an empty sheet yields last_header_col=0 → new_col=1 (column A).
    last_header_col = 0
    for _c in range(1, ws.max_column + 1):
        if ws.cell(row=1, column=_c).value is not None:
            last_header_col = _c
    new_col = last_header_col + 1
    col_letter = get_column_letter(new_col)
    # ensure header row exists
    if ws.max_row < 1:
        ws.append([])
    ws.cell(row=1, column=new_col, value=column_name)
    # Set default value for existing rows (optional)
    if default_value is not None:
        for r in range(2, ws.max_row + 1):
            ws.cell(row=r, column=new_col, value=default_value)
    _mark_dirty()
    _flush_session(path)
    return {"message": "column_added", "column": column_name,
            "speech_text": f"Done! I've added the '{column_name}' column."}


def read_cell(path, cell, sheet_name=None):
    """Read the value of a single cell and return it with a speech response.

    Usage::

        result = read_cell("sales.xlsx", "B3")
        # → {"message": "cell_read", "cell": "B3", "value": 1500,
        #    "speech_text": "Cell B3 contains 1500."}
    """
    log.info('read_cell: path=%s  cell=%s  sheet=%s', path, cell, sheet_name)
    if not cell:
        return {"error": "cell reference is required",
                "speech_text": "Please tell me which cell to read, like A1 or B3."}
    cell = str(cell).upper().strip()
    m = _CELL_RE.match(cell)
    if not m:
        return {"error": "invalid_cell", "cell": cell,
                "speech_text": f"'{cell}' doesn't look like a valid cell. Try something like A1 or B3."}
    wb, ws = ensure_workbook(path, sheet_name)
    if isinstance(ws, dict):
        return ws
    col_letters, row = m.group(1), int(m.group(2))
    col_idx = _col_letters_to_index(col_letters)
    value = ws.cell(row=row, column=col_idx).value
    if value is None:
        speech = f"Cell {cell} is empty."
    else:
        speech = f"Cell {cell} contains {value}."
    return {"message": "cell_read", "cell": cell, "value": value, "speech_text": speech}


def write_cell(path, cell, value, sheet_name=None):
    log.info('write_cell: path=%s  cell=%s  value=%s  sheet=%s', path, cell, value, sheet_name)
    if not cell:
        return {"error": "cell reference is required"}
    wb, ws = ensure_workbook(path, sheet_name)
    if isinstance(ws, dict):
        return ws
    # accept A1 style
    cell = str(cell).upper().strip()
    m = _CELL_RE.match(cell)
    if not m:
        log.warning('write_cell: invalid cell reference %r', cell)
        return {"error": "invalid_cell", "cell": cell}
    col_letters, row = m.group(1), int(m.group(2))
    col_idx = _col_letters_to_index(col_letters)
    safe_value = sanitize_cell_value(value)
    _undo_snapshot(path, f"write cell {cell}")

    # ── COM live-write: workbook open in Excel ────────────────────────────
    abs_path = os.path.abspath(path)
    if _com_live_write_cell(abs_path, row, col_idx, safe_value, sheet_name):
        log.info('[Controller] write_cell: COM live path: %s %s=%r', abs_path, cell, safe_value)
        _session_reload_from_disk(abs_path)
        return {"message": "cell_written", "cell": cell, "value": safe_value,
                "speech_text": f"Done! I've set cell {cell} to {safe_value}."}

    # Safety gate: file still locked → openpyxl + _atomic_save would crash.
    if _is_file_locked(abs_path):
        log.error('[Controller] write_cell: file locked by Excel but COM write failed %s',
                  abs_path)
        return {
            'error': 'com_write_failed',
            'message': 'cell_not_written',
            'speech_text': 'The file is open in Excel but I could not write to it. Please try again.',
        }

    # ── Openpyxl path: workbook not open in Excel ─────────────────────────
    log.info('[Controller] write_cell: openpyxl path for %s %s', path, cell)
    ws.cell(row=row, column=col_idx, value=safe_value)
    _mark_dirty()
    _flush_session(path)
    return {"message": "cell_written", "cell": cell, "value": safe_value,
            "speech_text": f"Done! I've set cell {cell} to {safe_value}."}


def write_formula(path, cell, formula, sheet_name=None):
    """Write an Excel formula (e.g. =SUM(A1:A10)) to a specific cell."""
    log.info('write_formula: path=%s  cell=%s  formula=%s  sheet=%s', path, cell, formula, sheet_name)
    if not cell:
        return {"error": "cell reference is required"}
    if not formula or not isinstance(formula, str):
        return {"error": "formula is required and must be a string"}
    formula = formula.strip()
    if not formula.startswith('='):
        formula = '=' + formula
    # Hard-block dangerous keywords regardless of function-whitelist status.
    # These are caught by substring scan so they trigger even without a call
    # syntax (e.g. CMD embedded inside a string literal).
    _upper = formula.upper()
    _BLOCKED = ('CMD', 'EXEC', 'SYSTEM', 'HYPERLINK(', 'WEBSERVICE(')
    for blocked in _BLOCKED:
        if blocked in _upper:
            return {
                "error": "formula_blocked",
                "detail": f"Formulas containing '{blocked}' are not allowed.",
                "speech_text": (
                    f"I can't write that formula — it contains a restricted keyword ({blocked}). "
                    "Please use a standard function like SUM or AVERAGE instead."
                ),
            }
    # Structural + whitelist validation — returns a human-friendly message on failure.
    _formula_err = get_formula_error(formula)
    if _formula_err:
        return {
            "error": "formula_invalid",
            "detail": _formula_err,
            "speech_text": _formula_err,
        }
    cell = str(cell).upper().strip()
    m = _CELL_RE.match(cell)
    if not m:
        return {"error": "invalid_cell", "cell": cell}
    col_letters, row = m.group(1), int(m.group(2))
    col_idx = _col_letters_to_index(col_letters)
    _undo_snapshot(path, f"write formula {cell}")
    wb, ws = ensure_workbook(path, sheet_name)
    if isinstance(ws, dict):
        return ws

    # ── COM live-write: workbook open in Excel ────────────────────────────
    abs_path = os.path.abspath(path)
    if _com_live_write_cell(abs_path, row, col_idx, formula, sheet_name):
        log.info('[Controller] write_formula: COM live path: %s %s=%r', abs_path, cell, formula)
        _session_reload_from_disk(abs_path)
        return {"message": "formula_written", "cell": cell, "formula": formula,
                "speech_text": f"Done! I've added the formula to cell {cell}."}

    # Safety gate: file still locked → openpyxl + _atomic_save would crash.
    if _is_file_locked(abs_path):
        log.error('[Controller] write_formula: file locked by Excel but COM write failed %s',
                  abs_path)
        return {
            'error': 'com_write_failed',
            'message': 'formula_not_written',
            'speech_text': 'The file is open in Excel but I could not write to it. Please try again.',
        }

    # ── Openpyxl path: workbook not open in Excel ─────────────────────────
    log.info('[Controller] write_formula: openpyxl path for %s %s', path, cell)
    ws.cell(row=row, column=col_idx, value=formula)
    _mark_dirty()
    _flush_session(path)
    return {"message": "formula_written", "cell": cell, "formula": formula,
            "speech_text": f"Done! I've added the formula to cell {cell}."}


def write_sum_formula(path, range_ref, cell, sheet_name=None):
    """Write =SUM(range_ref) into *cell*.

    Delegates entirely to :func:`write_formula` so all existing behaviour
    (undo snapshot, dirty-flag, flush, formula whitelist) is inherited
    without duplication.

    Example::

        write_sum_formula("data.xlsx", "B2:B10", "C2")
        # → writes =SUM(B2:B10) into C2
    """
    log.info('write_sum_formula: path=%s  range=%s  cell=%s', path, range_ref, cell)
    if not range_ref or not isinstance(range_ref, str):
        return {"error": "range is required", "speech_text": "Please specify a range like B2:B10."}
    if not cell or not isinstance(cell, str):
        return {"error": "cell is required", "speech_text": "Please specify a target cell like C2."}
    formula = f"=SUM({range_ref.strip().upper()})"
    result = write_formula(path, cell, formula, sheet_name=sheet_name)
    # Rewrite the speech_text to be specific to summing
    if isinstance(result, dict) and result.get("message") == "formula_written":
        result = dict(result,
                      speech_text=f"Done! I've written the SUM formula for {range_ref} into cell {cell}.")
    return result


def write_average_formula(path, range_ref, cell, sheet_name=None):
    """Write =AVERAGE(range_ref) into *cell*.

    Delegates entirely to :func:`write_formula` so undo snapshot, dirty-flag,
    flush, and formula-whitelist checks are all inherited without duplication.

    Example::

        write_average_formula("data.xlsx", "B2:B10", "C2")
        # → writes =AVERAGE(B2:B10) into C2
    """
    log.info('write_average_formula: path=%s  range=%s  cell=%s', path, range_ref, cell)
    if not range_ref or not isinstance(range_ref, str):
        return {"error": "range is required", "speech_text": "Please specify a range like B2:B10."}
    if not cell or not isinstance(cell, str):
        return {"error": "cell is required", "speech_text": "Please specify a target cell like C2."}
    formula = f"=AVERAGE({range_ref.strip().upper()})"
    result = write_formula(path, cell, formula, sheet_name=sheet_name)
    if isinstance(result, dict) and result.get("message") == "formula_written":
        result = dict(result,
                      speech_text=f"Done! I've written the AVERAGE formula for {range_ref} into cell {cell}.")
    return result


def calculate_row_sum(path, index, sheet_name=None):
    log.info('calculate_row_sum: path=%s  index=%s  sheet=%s', path, index, sheet_name)
    try:
        idx = int(index)
    except (TypeError, ValueError):
        return {"error": "index must be an integer"}
    wb, ws = ensure_workbook(path, sheet_name)
    if isinstance(ws, dict):
        return ws
    if idx < 1 or idx > ws.max_row:
        return {"error": "row_out_of_range", "max_row": ws.max_row}
    total = 0.0
    found = False
    for c in range(1, ws.max_column + 1):
        v = ws.cell(row=idx, column=c).value
        try:
            if v is not None and v != '':
                total += float(v)
                found = True
        except Exception:
            continue
    if not found:
        return {"message": "row_sum", "sum": 0.0, "speech_text": f"Row {idx} contains no numeric values."}
    _undo_snapshot(path, f"row sum row {idx}")
    # Write the result to the next column so it is visible in Excel, then flush.
    result_col = ws.max_column + 1
    ws.cell(row=idx, column=result_col, value=total)
    _mark_dirty()
    _flush_session(path)
    return {"message": "row_sum", "sum": total, "speech_text": f"The sum of row {idx} is {total}."}


def open_excel(path):
    """Open an existing Excel file and return its metadata.

    Registers *path* as the active session so subsequent commands that
    omit the filename (add row, write cell, …) automatically target this file.
    Only one file is active at a time; calling this replaces any prior session.
    """
    global _session
    log.info('open_excel: path=%s', path)
    if not os.path.exists(path):
        return {"error": "file_not_found", "path": path}
    try:
        wb = load_workbook(path)
    except Exception as exc:
        return {"error": "invalid_workbook", "detail": str(exc)}
    sheets = wb.sheetnames
    ws = wb.active
    abs_path = os.path.abspath(path)
    # Seed the session with the workbook we just loaded — no re-read needed.
    with _session_lock:
        _session = _WorkbookSession(abs_path, wb)
        # dirty=False: matches what is currently on disk
    # Display in Excel and bring the window to the front.
    _launch_excel_com(abs_path)
    msg = f"Opened {os.path.basename(path)} with {len(sheets)} sheet(s) and {ws.max_row} rows."
    return {
        "message": "file_opened",
        "path": path,
        "sheets": sheets,
        "active_sheet": ws.title,
        "rows": ws.max_row,
        "columns": ws.max_column,
        "headers": _ws_headers(ws),   # row-1 values so _update_memory can sync headers
        "speech_text": msg,
    }


def close_excel(path: str | None = None) -> dict:
    """Save and close the active Excel session.

    Flushes any pending in-memory changes to disk, then clears the session
    so subsequent pathless commands (add_row, write_cell, …) trigger
    ``_auto_open_default`` rather than silently targeting the old file.

    1. Resolves the target file (explicit *path* or active session).
    2. Writes the workbook to disk if ``dirty=True``.
    3. Clears ``_session``.

    Parameters
    ----------
    path : When given, only closes the session if it matches the currently
           active path — guards against accidentally closing a different file
           (e.g. if two code paths race).  When omitted, closes whatever is
           currently active.

    Returns
    -------
    dict with ``"message"``, ``"path"``, and ``"speech_text"`` on success, or
    an ``"error"`` key on failure.
    """
    global _session

    with _session_lock:
        current = _session.abs_path if _session else None

        if path:
            abs_path = os.path.abspath(path)
            if current and os.path.normcase(current) != os.path.normcase(abs_path):
                return {
                    "error": "file_not_active",
                    "requested": abs_path,
                    "active": current,
                    "speech_text": (
                        f"{os.path.basename(abs_path)} is not the active file. "
                        f"The currently open file is {os.path.basename(current)}."
                    ),
                }
            target = abs_path
        else:
            target = current

        if not target:
            return {
                "message": "no_session_to_close",
                "speech_text": "No Excel file is currently open.",
            }

        # Capture dirty state and workbook before clearing the session.
        wb_to_flush = _session.workbook if (_session and _session.dirty) else None
        _session = None   # clear inside the lock

    # Close the workbook in Excel FIRST so its write lock is released before
    # we attempt _atomic_save.  If we saved first (old order), Excel's lock on
    # the open file would cause WinError 5 / PermissionError on Windows.
    # _com_close_and_maybe_quit is a no-op when Excel is not running.
    _com_close_and_maybe_quit(target)
    # Poll until the OS releases the write lock (async after COM close/xl.Quit).
    if os.path.exists(target) and _is_file_locked(target):
        _poll_for_unlock(target)

    # Flush any unsaved in-memory changes — outside the lock so I/O doesn't block.
    # Excel's write lock is now gone, so _atomic_save can replace the file safely.
    save_ok = True
    if wb_to_flush is not None:
        try:
            _atomic_save(wb_to_flush, target)
            log.info('[Session] flushed on close: %s', target)
        except Exception as exc:
            log.error('[Session] flush on close failed: %s', exc)
            save_ok = False

    fname = os.path.basename(target)
    log.info('[Session] close_excel: cleared session for %s (save_ok=%s)', target, save_ok)
    if save_ok:
        return {
            "message": "excel_closed",
            "path": target,
            "speech_text": f"Closed {fname}. All changes have been saved.",
        }
    return {
        "error": "save_failed_on_close",
        "message": "excel_closed_save_failed",
        "path": target,
        "speech_text": (
            f"Closed {fname}, but the file could not be saved — "
            "it may be locked by another process. Please save manually."
        ),
    }


def calculate_column_average(path, col, sheet_name=None):
    """Calculate the average of numeric values in a column (letter or 1-based index)."""
    log.info('calculate_column_average: path=%s  col=%s  sheet=%s', path, col, sheet_name)
    if col is None:
        # Auto-detect: use the first column that contains numeric values
        # (skip header row 1). Lets "calculate average" work without specifying a column.
        _wb_tmp, _ws_tmp = ensure_workbook(path, sheet_name)
        if isinstance(_ws_tmp, dict):
            return _ws_tmp
        _detected = None
        for _c in range(1, _ws_tmp.max_column + 1):
            for _r in range(2, _ws_tmp.max_row + 1):
                _v = _ws_tmp.cell(row=_r, column=_c).value
                try:
                    if _v is not None:
                        float(_v)
                        _detected = _c
                        break
                except (TypeError, ValueError):
                    continue
            if _detected is not None:
                break
        if _detected is None:
            return {"error": "no numeric column found — add numeric data first"}
        log.info('calculate_column_average: auto-detected numeric column index=%d', _detected)
        col = _detected
    wb, ws = ensure_workbook(path, sheet_name)
    if isinstance(ws, dict):
        return ws
    # resolve column identifier
    if isinstance(col, str):
        col = col.strip().upper()
        if col.isalpha():
            col_idx = _col_letters_to_index(col)
        else:
            try:
                col_idx = int(col)
            except Exception:
                return {"error": "invalid_column", "col": col}
    else:
        try:
            col_idx = int(col)
        except (TypeError, ValueError):
            return {"error": "invalid_column", "col": col}
    max_col = ws.max_column
    if col_idx < 1 or col_idx > max_col:
        return {"error": "column_out_of_range", "max_col": max_col}
    total = 0.0
    count = 0
    # skip header row (row 1)
    for r in range(2, ws.max_row + 1):
        v = ws.cell(row=r, column=col_idx).value
        try:
            if v is not None and v != '':
                total += float(v)
                count += 1
        except (TypeError, ValueError):
            continue
    if count == 0:
        col_letter = get_column_letter(col_idx)
        return {"message": "column_average", "average": 0.0, "count": 0,
                "speech_text": f"Column {col_letter} contains no numeric values."}
    avg = total / count
    col_letter = get_column_letter(col_idx)
    msg = f"The average of column {col_letter} is {avg:.2f} across {count} values."
    _undo_snapshot(path, f"column average {col_letter}")
    # Write the result to the next row so it is visible in Excel, then flush.
    result_row = ws.max_row + 1
    ws.cell(row=result_row, column=col_idx, value=round(avg, 2))
    _mark_dirty()
    _flush_session(path)
    return {"message": "column_average", "average": avg, "count": count, "speech_text": msg}


_COLOR_NAMES = {
    'red':        'FF0000', 'green':      '00B050', 'blue':       '0070C0',
    'yellow':     'FFFF00', 'orange':     'FFA500', 'purple':     '7030A0',
    'pink':       'FFC0CB', 'black':      '000000', 'white':      'FFFFFF',
    'grey':       '808080', 'gray':       '808080', 'cyan':       '00FFFF',
    'brown':      'A52A2A', 'gold':       'FFD700', 'light blue': 'ADD8E6',
    'light_blue': 'ADD8E6', 'light green':'90EE90', 'dark blue':  '00008B',
}

_NUM_FORMAT_NAMES = {
    'number':     '#,##0.00',
    'integer':    '#,##0',
    'currency':   '$#,##0.00',
    'percentage': '0.00%',
    'percent':    '0.00%',
    'decimal':    '0.00',
    'date':       'YYYY-MM-DD',
    'text':       '@',
}


def format_cells(path, cell_range, bold=None, italic=None, font_size=None,
                 font_color=None, fill_color=None, num_format=None,
                 align=None, font_size_change=None,
                 sheet_name=None):
    """Apply formatting to a range of cells.

    Parameters:
        cell_range: A1-style range string, e.g. 'A1:E1' or 'B2'.
        bold/italic: bool
        font_size: int (points)
        font_color/fill_color: hex string without '#', or a named color like 'red'
        num_format: Excel number-format string or a friendly name like 'number'
    """
    log.info('format_cells: path=%s range=%s bold=%s fill=%s', path, cell_range, bold, fill_color)
    if not cell_range:
        return {"error": "cell range is required"}
    # Resolve named colors to hex
    if font_color is not None:
        font_color = _COLOR_NAMES.get(str(font_color).lower(), str(font_color).lstrip('#'))
    if fill_color is not None:
        fill_color = _COLOR_NAMES.get(str(fill_color).lower(), str(fill_color).lstrip('#'))
    # Resolve friendly number-format names to Excel format strings
    if num_format is not None:
        num_format = _NUM_FORMAT_NAMES.get(str(num_format).lower(), str(num_format))
    _undo_snapshot(path, f"format cells {cell_range}")
    wb, ws = ensure_workbook(path, sheet_name)
    if isinstance(ws, dict):
        return ws
    # Build font/fill objects from the provided options
    font_kwargs = {}
    if bold is not None:
        font_kwargs['bold'] = bool(bold)
    if italic is not None:
        font_kwargs['italic'] = bool(italic)
    if font_size is not None:
        try:
            font_kwargs['size'] = int(font_size)
        except (TypeError, ValueError):
            return {"error": "font_size must be an integer"}
    if font_color is not None:
        font_kwargs['color'] = str(font_color).lstrip('#')
    font = Font(**font_kwargs) if font_kwargs else None
    fill = None
    if fill_color is not None:
        fc = str(fill_color).lstrip('#')
        fill = PatternFill(start_color=fc, end_color=fc, fill_type='solid')
    alignment = Alignment(horizontal=align.strip().lower()) if align else None
    # Apply to every cell in range
    cells_affected = 0
    target = ws[cell_range]
    # ws[range] returns a Cell for a single cell, a tuple of Cells for a row,
    # or a tuple-of-tuples for a multi-row range.
    if not isinstance(target, tuple):
        target = ((target,),)
    for row in target:
        if not isinstance(row, tuple):
            row = (row,)
        for cell in row:
            if font:
                cell.font = font
            elif font_size_change is not None:
                cur = cell.font if cell.font else Font()
                cur_size = cur.size if cur.size else 11
                new_size = max(6, int(cur_size) + int(font_size_change))
                cell.font = Font(name=cur.name, bold=cur.bold, italic=cur.italic,
                                 color=cur.color, size=new_size)
            if fill:
                cell.fill = fill
            if num_format is not None:
                cell.number_format = str(num_format)
            if alignment:
                cell.alignment = alignment
            cells_affected += 1
    _mark_dirty()
    # If the workbook is open in Excel, apply formatting live via COM to avoid
    # WinError 5 / Access Denied that _atomic_save raises on a locked file.
    # On success COM has already saved; skip _flush_session (close/reopen cycle).
    # On COM failure (unavailable or workbook not open) fall through to the
    # normal openpyxl flush path.
    abs_path = os.path.abspath(path)
    _com_fmt = _com_live_format_cells(
        abs_path, cell_range,
        bold=bold, italic=italic, font_size=font_size,
        font_color=font_color, fill_color=fill_color,
        num_format=num_format,
        align=align, font_size_change=font_size_change,
        sheet_name=sheet_name,
    )
    if _com_fmt is not None:
        # COM saved the file; mark session clean so next flush is a no-op.
        with _session_lock:
            if _session is not None and _session.abs_path == abs_path:
                _session.dirty = False
    else:
        _flush_session(path)
    # Build a descriptive speech response
    applied = []
    if bold:
        applied.append("bold")
    if italic:
        applied.append("italic")
    if fill_color is not None:
        applied.append("fill color")
    if font_color is not None:
        applied.append("font color")
    if num_format is not None:
        applied.append("number format")
    if font_size is not None or font_size_change is not None:
        applied.append("font size")
    if align is not None:
        applied.append(f"aligned {align}")
    desc = " and ".join(applied) if applied else "formatted"
    msg = f"Done! Made {cell_range} {desc}."
    return {"message": "cells_formatted", "range": cell_range,
            "cells_affected": cells_affected, "speech_text": msg}


_CHART_TYPES = {
    'bar': BarChart,
    'line': LineChart,
    'pie': PieChart,
    'scatter': ScatterChart,
}


def create_chart(path, chart_type='bar', data_range=None, title=None,
                 x_axis=None, y_axis=None, position='E2',
                 sheet_name=None):
    """Create a chart and embed it in the active sheet.

    Parameters:
        chart_type: 'bar', 'line', 'pie', or 'scatter'.
        data_range: A1-style range like 'A1:C10'.  The first column is used as
                    category labels and remaining columns become data series.
        title: optional chart title.
        x_axis/y_axis: optional axis labels (ignored for pie).
        position: cell where the top-left corner of the chart is anchored.
    """
    log.info('create_chart: path=%s type=%s range=%s', path, chart_type, data_range)
    chart_cls = _CHART_TYPES.get(str(chart_type).lower())
    if chart_cls is None:
        _allowed = ', '.join(_CHART_TYPES.keys())
        return {
            "error": "unsupported_chart_type",
            "allowed": list(_CHART_TYPES.keys()),
            "speech_text": (
                f"I don't know how to create a '{chart_type}' chart. "
                f"Try one of: {_allowed}."
            ),
        }

    wb, ws = ensure_workbook(path, sheet_name)
    if isinstance(ws, dict):
        return ws

    # Auto-detect data_range from the sheet when the caller did not specify one.
    # Covers "create a bar chart" voice commands where no range was given.
    if not data_range:
        if ws.max_row < 2 or ws.max_column < 1:
            return {
                "error": "chart_no_data",
                "speech_text": (
                    "I can't create a chart because the sheet has no data yet. "
                    "Please add at least a header row and one data row first."
                ),
            }
        data_range = f"A1:{get_column_letter(ws.max_column)}{ws.max_row}"
        log.info('create_chart: auto-detected data_range=%s', data_range)

    # Parse the A1-style range into min/max row/col
    range_match = re.match(r'^([A-Z]+)(\d+):([A-Z]+)(\d+)$', data_range.upper().strip())
    if not range_match:
        return {
            "error": "invalid_data_range",
            "detail": "Expected format like A1:C10",
            "speech_text": (
                f"The data range '{data_range}' doesn't look right. "
                "Please use a format like A1:C10."
            ),
        }
    min_col = _col_letters_to_index(range_match.group(1))
    min_row = int(range_match.group(2))
    max_col = _col_letters_to_index(range_match.group(3))
    max_row = int(range_match.group(4))

    # ---- Pre-execution data validation (before modifying the workbook) ----

    # Need at least one data row below the header row.
    if max_row - min_row < 1:
        return {
            "error": "chart_insufficient_rows",
            "speech_text": (
                f"The range {data_range} only has one row. "
                "A chart needs a header row and at least one data row."
            ),
        }

    # Need at least two columns: one for labels, one for data.
    if max_col <= min_col:
        return {
            "error": "chart_insufficient_columns",
            "speech_text": (
                f"The range {data_range} only has one column. "
                "A chart needs at least two columns — one for labels and one for values."
            ),
        }

    # Scatter charts require at least two data columns (X and Y).
    if str(chart_type).lower() == 'scatter' and (max_col - min_col) < 2:
        return {
            "error": "chart_scatter_needs_two_columns",
            "speech_text": (
                "A scatter chart needs at least two data columns — "
                "one for X values and one for Y values."
            ),
        }

    # Check that the range isn't entirely outside the worksheet's data area.
    actual_max_row = ws.max_row
    actual_max_col = ws.max_column
    if min_row > actual_max_row or min_col > actual_max_col:
        return {
            "error": "chart_range_outside_data",
            "speech_text": (
                f"The range {data_range} is outside the data in this sheet. "
                f"The sheet currently has {actual_max_row} row(s) and "
                f"{actual_max_col} column(s)."
            ),
        }

    # Check that the data columns contain at least one numeric value.
    # Scan only the data rows (skip header row at min_row) up to actual data bounds.
    _scan_max_row = min(max_row, actual_max_row)
    _scan_max_col = min(max_col, actual_max_col)
    has_numeric = False
    for _r in range(min_row + 1, _scan_max_row + 1):
        for _c in range(min_col + 1, _scan_max_col + 1):
            _v = ws.cell(row=_r, column=_c).value
            if _v is not None:
                try:
                    float(_v)
                    has_numeric = True
                    break
                except (TypeError, ValueError):
                    pass
        if has_numeric:
            break
    if not has_numeric:
        return {
            "error": "chart_no_numeric_data",
            "speech_text": (
                f"I couldn't create the chart because the range {data_range} "
                "has no numeric values. Please add numbers to your data first."
            ),
        }

    chart = chart_cls()
    if title:
        chart.title = str(title)
    if x_axis and hasattr(chart, 'x_axis'):
        chart.x_axis.title = str(x_axis)
    if y_axis and hasattr(chart, 'y_axis'):
        chart.y_axis.title = str(y_axis)

    # Build series — ScatterChart uses a different openpyxl API than the others.
    if str(chart_type).lower() == 'scatter':
        # ScatterChart requires Series(yvalues, xvalues) — add_data/set_categories
        # are not supported and raise AttributeError on openpyxl ScatterChart.
        from openpyxl.chart import Series as _ChartSeries
        xvals = Reference(ws, min_col=min_col, min_row=min_row + 1, max_row=max_row)
        for c in range(min_col + 1, max_col + 1):
            # Include header row so title_from_data picks up the column label.
            yvals = Reference(ws, min_col=c, min_row=min_row, max_row=max_row)
            series = _ChartSeries(yvals, xvals, title_from_data=True)
            chart.series.append(series)
    else:
        # Bar, Line, Pie: standard category + data references.
        cats = Reference(ws, min_col=min_col, min_row=min_row + 1, max_row=max_row)
        for c in range(min_col + 1, max_col + 1):
            data_ref = Reference(ws, min_col=c, min_row=min_row, max_row=max_row)
            chart.add_data(data_ref, titles_from_data=True)
        chart.set_categories(cats)

    ws.add_chart(chart, str(position).upper())
    _mark_dirty()
    _flush_session(path)
    msg = f"{chart_type.capitalize()} chart created from {data_range}."
    return {"message": "chart_created", "chart_type": chart_type,
            "data_range": data_range, "position": position, "speech_text": msg}


def save(path):
    """Flush any pending in-memory changes to disk."""
    _flush_session(path)
    return {"message": "saved", "path": path, "speech_text": "Your spreadsheet has been saved."}


# --------------- Multi-sheet management ---------------


def create_sheet(path, sheet_name):
    """Create a new sheet in the workbook.  Returns an error if the name
    already exists or is empty."""
    log.info('create_sheet: path=%s  sheet=%s', path, sheet_name)
    if not sheet_name or not isinstance(sheet_name, str):
        return {"error": "sheet_name must be a non-empty string"}
    wb, ws = ensure_workbook(path)
    if isinstance(ws, dict):
        return ws
    if sheet_name in wb.sheetnames:
        return {"error": "sheet_already_exists", "sheet": sheet_name,
                "available": wb.sheetnames}
    # ── COM live path: avoid WinError 5 when file is open in Excel ───────────
    abs_path = os.path.abspath(path)
    xl_wb = _com_find_workbook(abs_path)
    if xl_wb is not None:
        try:
            xl_wb.Application.DisplayAlerts = False
            xl_wb.Worksheets.Add(After=xl_wb.Worksheets(xl_wb.Worksheets.Count))
            xl_wb.ActiveSheet.Name = sheet_name
            xl_wb.Save()
            log.info('[COM] create_sheet: created %r via COM', sheet_name)
        except Exception as exc:
            log.warning('[COM] create_sheet failed: %s', exc)
            return {"error": "create_sheet_com_failed", "detail": str(exc),
                    "speech_text": "The file is open in Excel but I could not create the sheet. Please try again."}
        wb.create_sheet(title=sheet_name)
        msg = f"Done! I've created a new sheet called '{sheet_name}'."
        return {"message": "sheet_created", "sheet": sheet_name,
                "sheets": wb.sheetnames, "speech_text": msg}

    if _is_file_locked(abs_path):
        return {
            'error': 'com_write_failed',
            'speech_text': 'The file is open in Excel but I could not write to it. Please try again.',
        }

    # ── openpyxl path (file NOT open in Excel) ────────────────────────────
    wb.create_sheet(title=sheet_name)
    _mark_dirty()
    _flush_session(path)
    msg = f"Done! I've created a new sheet called '{sheet_name}'."
    return {"message": "sheet_created", "sheet": sheet_name,
            "sheets": wb.sheetnames, "speech_text": msg}


def _resolve_sheet_name(wb, name: str):
    """Return the actual sheet title in *wb* that matches *name* case-insensitively.

    STT transcripts arrive in lowercase, so 'sheet' must find 'Sheet', 'style'
    must find 'Style', and '18' must find '18'.  Returns None when no tab matches.
    """
    target = name.lower()
    for sn in wb.sheetnames:
        if sn.lower() == target:
            return sn
    return None


def _resolve_sheet_name_merged(wb, name: str) -> str | None:
    """Try to split an STT-merged token like 'sheetbook' into 'sheet' + sheet_name.

    When the STT engine concatenates 'sheet' directly with the sheet name
    (e.g. 'sheetbook' instead of 'sheet book'), this extracts the suffix and
    checks whether it matches an existing sheet.  Returns the resolved canonical
    name on success, else None.  Only splits when the suffix is non-empty so
    'sheet' alone does not produce a spurious match.
    """
    lower = name.lower()
    if not lower.startswith('sheet') or len(lower) <= 5:
        return None
    return _resolve_sheet_name(wb, lower[5:])


def switch_sheet(path, sheet_name):
    """Set the active sheet in the workbook.  Fails if the sheet does not exist."""
    log.info('switch_sheet: path=%s  sheet=%s', path, sheet_name)
    if not sheet_name or not isinstance(sheet_name, str):
        return {"error": "sheet_name must be a non-empty string"}
    wb, ws = ensure_workbook(path)
    if isinstance(ws, dict):
        return ws
    resolved = _resolve_sheet_name(wb, sheet_name)
    if resolved is None:
        return {"error": "sheet_not_found", "sheet": sheet_name,
                "available": wb.sheetnames}
    sheet_name = resolved

    # ── COM live-activation path ──────────────────────────────────────────────
    # When the workbook is already open in Excel, activate the worksheet directly
    # via COM.  Going through _flush_session here would close the file, call
    # _atomic_save (WinError 5 / Access Denied — Excel still holds the write
    # lock), then reopen — all unnecessary work for a simple sheet switch.
    xl_wb = _com_find_workbook(os.path.abspath(path))
    if xl_wb is not None:
        try:
            xl_wb.Worksheets(sheet_name).Activate()
            log.info('[COM] switch_sheet: activated %r in live Excel session', sheet_name)
        except Exception as exc:
            log.warning('[COM] switch_sheet: Activate failed for %r: %s', sheet_name, exc)
            return {"error": "switch_sheet_com_failed",
                    "sheet": sheet_name, "detail": str(exc)}
        # Keep openpyxl's active-sheet pointer in sync without writing to disk.
        wb.active = wb.sheetnames.index(sheet_name)
        ws = wb[sheet_name]
        msg = f"Switched to '{sheet_name}'. It has {ws.max_row} rows."
        return {"message": "sheet_switched", "sheet": sheet_name,
                "rows": ws.max_row, "columns": ws.max_column,
                "headers": _ws_headers(ws),
                "speech_text": msg}

    # ── openpyxl path (workbook NOT open in Excel) ────────────────────────────
    wb.active = wb.sheetnames.index(sheet_name)
    _mark_dirty()
    _flush_session(path)
    ws = wb[sheet_name]
    msg = f"Switched to '{sheet_name}'. It has {ws.max_row} rows."
    return {"message": "sheet_switched", "sheet": sheet_name,
            "rows": ws.max_row, "columns": ws.max_column,
            "headers": _ws_headers(ws),
            "speech_text": msg}


def delete_sheet(path, sheet_name):
    """Delete a sheet from the workbook.  Cannot delete the last sheet."""
    log.info('delete_sheet: path=%s  sheet=%s', path, sheet_name)
    if not sheet_name or not isinstance(sheet_name, str):
        return {"error": "sheet_name must be a non-empty string"}
    wb, ws = ensure_workbook(path)
    if isinstance(ws, dict):
        return ws
    resolved = _resolve_sheet_name(wb, sheet_name)
    if resolved is None:
        resolved = _resolve_sheet_name_merged(wb, sheet_name)
    if resolved is None:
        return {"error": "sheet_not_found", "sheet": sheet_name,
                "available": wb.sheetnames}
    sheet_name = resolved
    if len(wb.sheetnames) <= 1:
        return {"error": "cannot_delete_last_sheet",
                "detail": "A workbook must have at least one sheet."}

    # ── COM live-deletion path ────────────────────────────────────────────────
    # When the workbook is open in Excel, delete via COM so we never call
    # _atomic_save against the write-locked file (WinError 5).
    xl_wb = _com_find_workbook(os.path.abspath(path))
    if xl_wb is not None:
        try:
            xl_wb.Worksheets(sheet_name).Delete()
            log.info('[COM] delete_sheet: deleted %r via COM', sheet_name)
        except Exception as exc:
            log.warning('[COM] delete_sheet: Delete failed for %r: %s', sheet_name, exc)
            return {"error": "delete_sheet_com_failed",
                    "sheet": sheet_name, "detail": str(exc)}
        del wb[sheet_name]
        active = wb.active
        new_active = active.title if active else wb.sheetnames[0]
        msg = f"Deleted sheet '{sheet_name}'. Active sheet is now '{new_active}'."
        return {"message": "sheet_deleted", "sheet": sheet_name,
                "active_sheet": new_active, "sheets": wb.sheetnames,
                "speech_text": msg}

    # ── openpyxl path (workbook NOT open in Excel) ────────────────────────────
    _undo_snapshot(path, f"delete sheet {sheet_name}")
    del wb[sheet_name]
    _mark_dirty()
    _flush_session(path)
    active = wb.active
    new_active = active.title if active else wb.sheetnames[0]
    msg = f"Deleted sheet '{sheet_name}'. Active sheet is now '{new_active}'."
    return {"message": "sheet_deleted", "sheet": sheet_name,
            "active_sheet": new_active, "sheets": wb.sheetnames,
            "speech_text": msg}


def list_sheets(path):
    """Return the list of sheet names in the workbook."""
    log.info('list_sheets: path=%s', path)
    wb, ws = ensure_workbook(path)
    if isinstance(ws, dict):
        return ws
    active = wb.active.title
    return {"message": "sheets_listed", "sheets": wb.sheetnames,
            "active_sheet": active,
            "speech_text": f"Workbook has {len(wb.sheetnames)} sheet(s): {', '.join(wb.sheetnames)}."}


def find_and_replace(path, find_value, replace_value, sheet_name=None):
    """Replace every occurrence of *find_value* with *replace_value* in the sheet.

    Comparison is case-insensitive for strings.  Returns the number of cells
    changed and a human-friendly speech_text.
    """
    log.info('find_and_replace: path=%s  find=%r  replace=%r  sheet=%s',
             path, find_value, replace_value, sheet_name)
    if find_value is None:
        return {"error": "find_value is required"}
    wb, ws = ensure_workbook(path, sheet_name)
    if isinstance(ws, dict):
        return ws
    # Snapshot before modification so the user can undo a mass replace
    _undo_snapshot(path, f"find_replace '{find_value}' → '{replace_value}'")
    find_str = str(find_value).lower()

    # ── COM live path: avoid WinError 5 when file is open in Excel ───────────
    abs_path = os.path.abspath(path)
    xl_wb = _com_find_workbook(abs_path)
    if xl_wb is not None:
        try:
            changed = sum(
                1 for r in ws.iter_rows() for c in r
                if c.value is not None and str(c.value).lower() == find_str
            )
            xl_wb.Application.DisplayAlerts = False
            xl_ws = xl_wb.Worksheets(sheet_name) if sheet_name else xl_wb.ActiveSheet
            repl = '' if replace_value is None else str(replace_value)
            xl_ws.UsedRange.Replace(str(find_value), repl, 1, 1, False)
            xl_wb.Save()
            log.info('[COM] find_and_replace: %d replacements via COM', changed)
        except Exception as exc:
            log.warning('[COM] find_and_replace failed: %s', exc)
            return {"error": "find_replace_com_failed", "detail": str(exc),
                    "speech_text": "The file is open in Excel but I could not complete the replacement. Please try again."}
        _session_reload_from_disk(abs_path)
        msg = (f"Replaced {changed} occurrence{'s' if changed != 1 else ''} "
               f"of '{find_value}' with '{replace_value}'.")
        return {"message": "find_replace_done", "changed": changed, "speech_text": msg}

    if _is_file_locked(abs_path):
        return {
            'error': 'com_write_failed',
            'speech_text': 'The file is open in Excel but I could not write to it. Please try again.',
        }

    # ── openpyxl path (file NOT open in Excel) ────────────────────────────
    changed = 0
    for row in ws.iter_rows():
        for cell in row:
            if cell.value is not None and str(cell.value).lower() == find_str:
                cell.value = replace_value
                changed += 1
    _mark_dirty()
    _flush_session(path)
    msg = (f"Replaced {changed} occurrence{'s' if changed != 1 else ''} "
           f"of '{find_value}' with '{replace_value}'.")
    return {"message": "find_replace_done", "changed": changed, "speech_text": msg}


def sort_range(path, column, ascending=True, header_row=True, sheet_name=None):
    """Sort the sheet by *column* (letter or 1-based index).

    Reads all data rows into memory, sorts by the target column, then writes
    them back — preserving the header row when *header_row* is True.
    """
    log.info('sort_range: path=%s  col=%s  asc=%s  sheet=%s', path, column, ascending, sheet_name)
    if column is None:
        return {"error": "column identifier is required"}
    wb, ws = ensure_workbook(path, sheet_name)
    if isinstance(ws, dict):
        return ws

    if isinstance(column, str):
        column = column.strip().upper()
        if column.isalpha() and len(column) > 2:
            # Looks like a header name (e.g. "Score", "Revenue") rather than a
            # column letter (A-Z, AA-ZZ).  Try to resolve against row-1 headers.
            header_row = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)]
            matched = next(
                (i + 1 for i, h in enumerate(header_row)
                 if h is not None and str(h).upper() == column),
                None,
            )
            col_idx = matched if matched is not None else _col_letters_to_index(column)
        elif column.isalpha():
            col_idx = _col_letters_to_index(column)
        else:
            try:
                col_idx = int(column)
            except (TypeError, ValueError):
                return {"error": "invalid_column", "col": column}
    else:
        col_idx = int(column)

    max_col = ws.max_column
    if col_idx < 1 or col_idx > max_col:
        return {"error": "column_out_of_range", "max_col": max_col}

    _undo_snapshot(path, f"sort by column {get_column_letter(col_idx)}")

    # ── COM live path: sort via Excel's native engine when the file is open ──
    # Avoids the WinError 5 race from _atomic_save on a locked file, and is
    # faster than the Python read/sort/write cycle below.
    abs_path = os.path.abspath(path)
    _com_result = _com_live_sort_range(abs_path, col_idx, ascending, header_row, sheet_name)
    if _com_result is not None:
        with _session_lock:
            if _session is not None and _session.abs_path == abs_path:
                _session.dirty = False
        col_letter = get_column_letter(col_idx)
        direction = "ascending" if ascending else "descending"
        msg = f"Sorted sheet by column {col_letter} in {direction} order."
        return {"message": "sort_done", "column": col_letter, "ascending": ascending, "speech_text": msg}

    # ── Fallback: openpyxl read/sort/write (when COM is unavailable) ─────────
    start_row = 2 if header_row else 1
    rows = [[ws.cell(row=r, column=c).value for c in range(1, max_col + 1)]
            for r in range(start_row, ws.max_row + 1)]

    def _sort_key(row):
        v = row[col_idx - 1]
        if v is None:
            return (0, 0.0, '')
        try:
            return (1, float(v), '')
        except (TypeError, ValueError):
            return (2, 0.0, str(v).lower())

    rows.sort(key=_sort_key, reverse=not ascending)

    # Unmerge any ranges that overlap the sort area so we can write cell values back.
    for mr in list(ws.merged_cells.ranges):
        ws.unmerge_cells(str(mr))

    for r_idx, row_data in enumerate(rows, start=start_row):
        for c_idx, val in enumerate(row_data, start=1):
            ws.cell(row=r_idx, column=c_idx, value=val)

    _mark_dirty()
    _flush_session(path)
    col_letter = get_column_letter(col_idx)
    direction = "ascending" if ascending else "descending"
    msg = f"Sorted sheet by column {col_letter} in {direction} order."
    return {"message": "sort_done", "column": col_letter, "ascending": ascending, "speech_text": msg}


def set_column_width(path, column, width, sheet_name=None):
    """Set the width of *column* (letter or 1-based index) to *width* characters."""
    log.info('set_column_width: path=%s  col=%s  width=%s  sheet=%s', path, column, width, sheet_name)
    if column is None:
        return {"error": "column identifier is required"}
    try:
        width = float(width)
    except (TypeError, ValueError):
        return {"error": "width must be a number"}
    # Reject noise words that STT/parser sometimes pass as column letters.
    _COL_WIDTH_GARBAGE = frozenset({'WITH', 'BE', 'TO', 'AS', 'AT', 'THE', 'AND', 'OR', 'IN', 'OF'})
    if isinstance(column, str):
        column = column.strip().upper()
        if column in _COL_WIDTH_GARBAGE:
            return {"error": "invalid_column", "column": column,
                    "speech_text": "I couldn't find a column letter. Please say something like 'set column width A 20'."}
        col_letter = column if column.isalpha() else get_column_letter(int(column))
    else:
        col_letter = get_column_letter(int(column))
    # ── COM live path: avoid WinError 32/5 when file is open in Excel ────────
    abs_path = os.path.abspath(path)
    xl_wb = _com_find_workbook(abs_path)
    if xl_wb is not None:
        try:
            xl_wb.Application.DisplayAlerts = False
            xl_ws = xl_wb.Worksheets(sheet_name) if sheet_name else xl_wb.ActiveSheet
            xl_ws.Columns(col_letter).ColumnWidth = width
            xl_wb.Save()
            log.info('[COM] set_column_width: col=%s width=%s via COM', col_letter, width)
        except Exception as exc:
            log.warning('[COM] set_column_width failed: %s', exc)
            return {'error': 'com_write_failed', 'detail': str(exc),
                    'speech_text': "Sorry, I couldn't set the column width."}
        _session_reload_from_disk(abs_path)
        msg = f"Done! Column {col_letter} is now {width} characters wide."
        return {"message": "column_width_set", "column": col_letter, "width": width, "speech_text": msg}

    if _is_file_locked(abs_path):
        return {'error': 'com_write_failed',
                'speech_text': 'The file is open in Excel but I could not write to it. Please try again.'}

    # ── openpyxl path (file NOT open in Excel) ────────────────────────────
    wb, ws = ensure_workbook(path, sheet_name)
    if isinstance(ws, dict):
        return ws
    _undo_snapshot(path, f"set column width {col_letter}")
    ws.column_dimensions[col_letter].width = width
    _mark_dirty()
    _flush_session(path)
    msg = f"Done! Column {col_letter} is now {width} characters wide."
    return {"message": "column_width_set", "column": col_letter, "width": width, "speech_text": msg}


def auto_fit_column(path, column, sheet_name=None):
    """Auto-fit one or all columns by measuring the longest cell value."""
    log.info('auto_fit_column: path=%s  col=%s  sheet=%s', path, column, sheet_name)
    wb, ws = ensure_workbook(path, sheet_name)
    if isinstance(ws, dict):
        return ws

    def _fit_one(col_letter):
        # ws[col_letter] returns a tuple-of-tuples (one inner tuple per row).
        # Flatten to iterate every cell in the column.
        max_len = 0
        for row_cells in ws[col_letter]:
            for cell in (row_cells if isinstance(row_cells, tuple) else (row_cells,)):
                if cell.value is not None:
                    max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[col_letter].width = max(8, max_len + 2)

    col_str = str(column).strip().upper()
    _undo_snapshot(path, f"auto fit column {col_str.lower()}")

    # ── COM live path: avoid WinError 5 when file is open in Excel ───────────
    abs_path = os.path.abspath(path)
    xl_wb = _com_find_workbook(abs_path)
    if xl_wb is not None:
        try:
            xl_wb.Application.DisplayAlerts = False
            xl_ws = xl_wb.Worksheets(sheet_name) if sheet_name else xl_wb.ActiveSheet
            if col_str == 'ALL':
                xl_ws.Columns.AutoFit()
                fitted = 'all'
                msg = "Done! Auto-fitted all columns."
            else:
                col_letter = col_str if col_str.isalpha() else get_column_letter(int(col_str))
                xl_ws.Columns(col_letter).AutoFit()
                fitted = col_letter
                msg = f"Done! Auto-fitted column {col_letter}."
            xl_wb.Save()
            log.info('[COM] auto_fit_column: fitted %s via COM', col_str)
        except Exception as exc:
            log.warning('[COM] auto_fit_column failed: %s', exc)
            return {"error": "auto_fit_com_failed", "detail": str(exc),
                    "speech_text": "The file is open in Excel but I could not auto-fit. Please try again."}
        _session_reload_from_disk(abs_path)
        return {"message": "column_auto_fitted", "column": fitted, "speech_text": msg}

    if _is_file_locked(abs_path):
        return {
            'error': 'com_write_failed',
            'speech_text': 'The file is open in Excel but I could not write to it. Please try again.',
        }

    # ── openpyxl path (file NOT open in Excel) ────────────────────────────
    if col_str == 'ALL':
        for i in range(1, ws.max_column + 1):
            _fit_one(get_column_letter(i))
        fitted = 'all'
        msg = "Done! Auto-fitted all columns."
    else:
        col_letter = col_str if col_str.isalpha() else get_column_letter(int(col_str))
        _fit_one(col_letter)
        fitted = col_letter
        msg = f"Done! Auto-fitted column {col_letter}."

    _mark_dirty()
    _flush_session(path)
    return {"message": "column_auto_fitted", "column": fitted, "speech_text": msg}


def freeze_panes(path, cell='B2', sheet_name=None):
    """Freeze rows/columns above and to the left of *cell* (e.g. 'B2' freezes row 1)."""
    log.info('freeze_panes: path=%s  cell=%s  sheet=%s', path, cell, sheet_name)
    cell = str(cell).upper().strip() if cell else 'B2'
    m = _CELL_RE.match(cell)
    if not m:
        return {"error": "invalid_cell", "cell": cell}
    # ── COM live path: avoid WinError 5 when file is open in Excel ───────────
    abs_path = os.path.abspath(path)
    xl_wb = _com_find_workbook(abs_path)
    if xl_wb is not None:
        try:
            xl_wb.Application.DisplayAlerts = False
            from openpyxl.utils import column_index_from_string
            win = xl_wb.Application.ActiveWindow
            # Combining logic using live COM freeze state
            if win.FreezePanes:
                _cur_com = ('A2' if win.SplitRow > 0 and win.SplitColumn == 0 else
                            'B1' if win.SplitRow == 0 and win.SplitColumn > 0 else
                            'B2' if win.SplitRow > 0 and win.SplitColumn > 0 else None)
                if cell == 'A2' and _cur_com == 'B1':
                    cell = 'B2'
                elif cell == 'B1' and _cur_com == 'A2':
                    cell = 'B2'
            _fc = _CELL_RE.match(cell)   # re-parse AFTER combining so B2 gives row=1,col=1
            split_col = column_index_from_string(_fc.group(1).upper()) - 1
            split_row = int(_fc.group(2)) - 1
            win.FreezePanes = False
            win.SplitRow = split_row
            win.SplitColumn = split_col
            win.FreezePanes = True
            xl_wb.Save()
            log.info('[COM] freeze_panes: cell=%s via COM', cell)
        except Exception as exc:
            log.warning('[COM] freeze_panes failed: %s', exc)
            return {'error': 'com_write_failed', 'detail': str(exc),
                    'speech_text': "Sorry, I couldn't freeze the panes."}
        _session_reload_from_disk(abs_path)
        msg = "Done! I've frozen the header row. Your column headers will stay visible while you scroll."
        return {"message": "panes_frozen", "cell": cell, "speech_text": msg}

    if _is_file_locked(abs_path):
        return {'error': 'com_write_failed',
                'speech_text': 'The file is open in Excel but I could not write to it. Please try again.'}

    # ── openpyxl path (file NOT open in Excel) ────────────────────────────
    wb, ws = ensure_workbook(path, sheet_name)
    if isinstance(ws, dict):
        return ws
    # ── Combine with existing freeze state ────────────────────────────────────
    # A2 = row-only freeze; B1 = column-only freeze; B2 = both.
    # Adding a column freeze on top of a row freeze (or vice versa) → B2.
    _cur = str(ws.freeze_panes).upper() if ws.freeze_panes else None
    if cell == 'A2' and _cur == 'B1':
        cell = 'B2'
    elif cell == 'B1' and _cur == 'A2':
        cell = 'B2'
    _undo_snapshot(path, f"freeze panes {cell}")
    ws.freeze_panes = cell
    _mark_dirty()
    _flush_session(path)
    msg = "Done! I've frozen the header row. Your column headers will stay visible while you scroll."
    return {"message": "panes_frozen", "cell": cell, "speech_text": msg}


def unfreeze_panes(path, sheet_name=None):
    """Remove all frozen panes from the active (or named) sheet."""
    log.info('unfreeze_panes: path=%s  sheet=%s', path, sheet_name)
    # ── COM live path: avoid WinError 5 when file is open in Excel ───────────
    abs_path = os.path.abspath(path)
    xl_wb = _com_find_workbook(abs_path)
    if xl_wb is not None:
        try:
            xl_wb.Application.DisplayAlerts = False
            win = xl_wb.Application.ActiveWindow
            win.FreezePanes = False
            win.SplitRow = 0
            win.SplitColumn = 0
            xl_wb.Save()
            log.info('[COM] unfreeze_panes: cleared via COM')
        except Exception as exc:
            log.warning('[COM] unfreeze_panes failed: %s', exc)
            return {'error': 'com_write_failed', 'detail': str(exc),
                    'speech_text': "Sorry, I couldn't unfreeze the panes."}
        _session_reload_from_disk(abs_path)
        return {"message": "panes_unfrozen",
                "speech_text": "Done! I've unfrozen the worksheet."}

    if _is_file_locked(abs_path):
        return {'error': 'com_write_failed',
                'speech_text': 'The file is open in Excel but I could not write to it. Please try again.'}

    # ── openpyxl path (file NOT open in Excel) ────────────────────────────
    wb, ws = ensure_workbook(path, sheet_name)
    if isinstance(ws, dict):
        return ws
    _undo_snapshot(path, "unfreeze panes")
    ws.freeze_panes = None
    # openpyxl leaves <selection pane="bottomLeft"> after removing freeze_panes;
    # Excel COM refuses to open a file that references a pane that no longer
    # exists (error 0x800A03EC / 1004).  Replace the stale selection with a
    # clean default — no pane attribute — so the XML is valid for an unfrozen
    # sheet and a subsequent freeze call can still find selection[0].
    from openpyxl.worksheet.views import Selection as _Selection
    ws.sheet_view.selection.clear()
    ws.sheet_view.selection.append(_Selection(activeCell='A1', sqref='A1'))
    _mark_dirty()
    _flush_session(path)
    return {"message": "panes_unfrozen",
            "speech_text": "Done! I've unfrozen the worksheet."}


def merge_cells(path, cell_range, sheet_name=None):
    """Merge the cells in *cell_range* (e.g. 'A1:C1')."""
    log.info('merge_cells: path=%s  range=%s  sheet=%s', path, cell_range, sheet_name)
    if not cell_range:
        return {"error": "cell_range is required"}
    wb, ws = ensure_workbook(path, sheet_name)
    if isinstance(ws, dict):
        return ws
    # Validate that cell_range contains exactly one colon and two valid cell refs
    _RANGE_RE = re.compile(
        r'^([A-Za-z]+)(\d+):([A-Za-z]+)(\d+)$', re.IGNORECASE
    )
    if not _RANGE_RE.match(cell_range.strip()):
        return {"error": "invalid_range", "detail": f"'{cell_range}' is not a valid merge range like A1:C1"}
    _undo_snapshot(path, f"merge cells {cell_range}")
    try:
        ws.merge_cells(cell_range.strip().upper())
    except Exception as exc:
        return {"error": "merge_failed", "detail": str(exc)}
    _mark_dirty()
    # If the workbook is open in Excel, merge live via COM to avoid
    # WinError 5 / Access Denied that _atomic_save raises on a locked file.
    # On success COM has already saved; skip _flush_session (close/reopen cycle).
    abs_path = os.path.abspath(path)
    _com_result = _com_live_merge_cells(abs_path, cell_range.strip().upper(),
                                        sheet_name=sheet_name)
    if _com_result is not None:
        with _session_lock:
            if _session is not None and _session.abs_path == abs_path:
                _session.dirty = False
    else:
        _flush_session(path)
    msg = f"Done! I've merged cells {cell_range} into one."
    return {"message": "cells_merged", "range": cell_range, "speech_text": msg}


def unmerge_cells(path, cell_range, sheet_name=None):
    """Unmerge previously merged cells in *cell_range* (e.g. 'A1:C1')."""
    log.info('unmerge_cells: path=%s  range=%s  sheet=%s', path, cell_range, sheet_name)
    if not cell_range:
        return {"error": "cell_range is required"}
    wb, ws = ensure_workbook(path, sheet_name=sheet_name)
    if isinstance(ws, dict):
        return ws
    _undo_snapshot(path, f"unmerge cells {cell_range}")
    try:
        ws.unmerge_cells(cell_range.strip().upper())
    except Exception as exc:
        return {"error": "unmerge_failed", "detail": str(exc)}
    _mark_dirty()
    # If the workbook is open in Excel, unmerge live via COM to avoid
    # WinError 5 / Access Denied that _atomic_save raises on a locked file.
    # On success COM has already saved; skip _flush_session (close/reopen cycle).
    # On COM failure (unavailable or workbook not open) fall through to the
    # normal openpyxl flush path.
    abs_path = os.path.abspath(path)
    _com_result = _com_live_unmerge_cells(abs_path, cell_range.strip().upper(),
                                          sheet_name=sheet_name)
    if _com_result is not None:
        # COM saved the file; mark session clean so next flush is a no-op.
        with _session_lock:
            if _session is not None and _session.abs_path == abs_path:
                _session.dirty = False
    else:
        _flush_session(path)
    msg = f"Done! I've unmerged cells {cell_range}."
    return {"message": "cells_unmerged", "range": cell_range, "speech_text": msg}


def rename_sheet(path, old_name, new_name):
    """Rename a sheet from *old_name* to *new_name*."""
    log.info('rename_sheet: path=%s  old=%s  new=%s', path, old_name, new_name)
    if not old_name or not new_name:
        return {"error": "both old_name and new_name are required"}
    wb, ws = ensure_workbook(path)
    if isinstance(ws, dict):
        return ws
    resolved_old = _resolve_sheet_name(wb, old_name)
    if resolved_old is None:
        return {"error": "sheet_not_found", "sheet": old_name, "available": wb.sheetnames}
    old_name = resolved_old
    if new_name in wb.sheetnames:
        return {"error": "sheet_name_taken", "sheet": new_name}
    _undo_snapshot(path, f"rename sheet {old_name} → {new_name}")

    # ── COM live path: avoid WinError 5 when file is open in Excel ───────────
    abs_path = os.path.abspath(path)
    xl_wb = _com_find_workbook(abs_path)
    if xl_wb is not None:
        try:
            xl_wb.Application.DisplayAlerts = False
            xl_wb.Worksheets(old_name).Name = new_name
            xl_wb.Save()
            log.info('[COM] rename_sheet: renamed %r → %r via COM', old_name, new_name)
        except Exception as exc:
            log.warning('[COM] rename_sheet failed: %s', exc)
            return {"error": "rename_sheet_com_failed", "detail": str(exc),
                    "speech_text": "The file is open in Excel but I could not rename the sheet. Please try again."}
        wb[old_name].title = new_name
        msg = f"Renamed sheet '{old_name}' to '{new_name}'."
        return {"message": "sheet_renamed", "old_name": old_name, "new_name": new_name, "speech_text": msg}

    if _is_file_locked(abs_path):
        return {
            'error': 'com_write_failed',
            'speech_text': 'The file is open in Excel but I could not write to it. Please try again.',
        }

    # ── openpyxl path (file NOT open in Excel) ────────────────────────────
    wb[old_name].title = new_name
    _mark_dirty()
    _flush_session(path)
    msg = f"Renamed sheet '{old_name}' to '{new_name}'."
    return {"message": "sheet_renamed", "old_name": old_name, "new_name": new_name, "speech_text": msg}


def rename_file(path, new_name):
    """Rename the current Excel file to *new_name* (same directory).

    Safely handles the Windows write lock by closing the Excel UI instance
    before renaming, then reopening the renamed file.  The in-memory session
    and undo stack keys are migrated to the new path so subsequent commands
    continue to work without a reload.
    """
    log.info('rename_file: path=%s  new_name=%s', path, new_name)
    if not new_name or not str(new_name).strip():
        return {"error": "new_name_required",
                "speech_text": "Please tell me what to rename the file to."}

    abs_old = os.path.abspath(path)
    if not os.path.exists(abs_old):
        return {"error": "file_not_found", "path": path,
                "speech_text": f"I can't find the file at {path}."}

    new_name = str(new_name).strip()
    if not new_name.lower().endswith('.xlsx'):
        new_name += '.xlsx'

    dir_path = os.path.dirname(abs_old)
    abs_new = os.path.join(dir_path, new_name)

    if os.path.normcase(abs_old) == os.path.normcase(abs_new):
        return {"error": "same_name", "name": new_name,
                "speech_text": f"The file is already named '{new_name}'."}
    if os.path.exists(abs_new):
        return {"error": "name_taken", "name": new_name,
                "speech_text": f"A file named '{new_name}' already exists."}

    # Always flush the in-memory workbook to disk before renaming — regardless of the
    # dirty flag.  COM-path operations (e.g. delete_sheet via xl_wb.Worksheets.Delete)
    # update the openpyxl workbook in memory but do NOT set _session.dirty=True, so
    # relying solely on the dirty flag would rename a stale disk file and lose changes
    # (deleted sheets would reappear, written cells would vanish).
    with _session_lock:
        wb_to_flush = (
            _session.workbook
            if _session is not None
            and os.path.normcase(_session.abs_path) == os.path.normcase(abs_old)
            else None
        )

    _com_close_workbook(abs_old)   # release Excel's write lock before saving

    if wb_to_flush is not None:
        try:
            _atomic_save(wb_to_flush, abs_old)
            with _session_lock:
                if (_session is not None
                        and os.path.normcase(_session.abs_path) == os.path.normcase(abs_old)):
                    _session.dirty = False
        except Exception as exc:
            log.warning('rename_file: pre-rename flush failed: %s', exc)

    try:
        os.rename(abs_old, abs_new)
    except OSError as exc:
        _com_open_workbook(abs_old)   # restore UI on failure
        return {"error": "rename_failed", "detail": str(exc),
                "speech_text": f"I couldn't rename the file: {exc}"}

    # Migrate in-memory session to new path.
    with _session_lock:
        if (_session is not None
                and os.path.normcase(_session.abs_path) == os.path.normcase(abs_old)):
            _session.abs_path = abs_new

    # Migrate undo stack key so existing snapshots remain accessible.
    if abs_old in _UNDO_STACK:
        _UNDO_STACK[abs_new] = _UNDO_STACK.pop(abs_old)

    _com_open_workbook(abs_new)   # show renamed file in Excel UI

    old_base = os.path.basename(abs_old)
    msg = f"Done! I've renamed '{old_base}' to '{new_name}'."
    return {"message": "file_renamed", "old_name": old_base, "new_name": new_name,
            "new_path": abs_new, "speech_text": msg}


def read_range(path, cell_range, sheet_name=None):
    """Read and return values from a cell range (e.g. 'A1:C3') as a 2-D list."""
    log.info('read_range: path=%s  range=%s  sheet=%s', path, cell_range, sheet_name)
    if not cell_range:
        return {"error": "cell_range is required"}
    wb, ws = ensure_workbook(path, sheet_name)
    if isinstance(ws, dict):
        return ws
    try:
        data = [[cell.value for cell in row] for row in ws[cell_range]]
    except Exception as exc:
        return {"error": "read_failed", "detail": str(exc)}
    rows = len(data)
    cols = len(data[0]) if data else 0
    msg = f"Read {rows} row{'s' if rows != 1 else ''} from {cell_range}."
    return {"message": "range_read", "range": cell_range, "data": data,
            "rows": rows, "columns": cols, "speech_text": msg}


def filter_table(path, column, operator='=', value=None, sheet_name=None):
    """Return rows where *column* matches *operator*/*value* and write them to a 'Filtered' sheet.

    Parameters:
        column:   Column header name (e.g. 'Status') or letter (e.g. 'A').
        operator: '=', '!=', '>', '<', '>=', '<=', or 'contains'.
        value:    The value to compare each cell against.

    The original sheet is left untouched.  A new sheet named 'Filtered' is
    created (replacing any previous one) containing the header row plus all
    matching data rows.
    """
    log.info('filter_table: path=%s col=%s op=%s val=%s', path, column, operator, value)
    if not column:
        return {"error": "column is required for filter_table"}
    wb, ws = ensure_workbook(path, sheet_name)
    if isinstance(ws, dict):
        return ws
    # If the active sheet is the "Filtered" results sheet from a prior run,
    # fall back to the first data (non-Filtered) sheet so we always filter source data.
    if ws.title == 'Filtered' and sheet_name is None:
        src = next((s for s in wb.sheetnames if s != 'Filtered'), None)
        if src:
            ws = wb[src]

    # Resolve the column index by header name first, then by letter.
    header_row = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    col_idx = None
    for i, h in enumerate(header_row, start=1):
        if str(h or '').lower() == str(column).lower():
            col_idx = i
            break
    if col_idx is None and re.fullmatch(r'[A-Za-z]{1,3}', str(column)):
        try:
            from openpyxl.utils import column_index_from_string
            col_idx = column_index_from_string(column.upper())
        except Exception:
            pass
    if col_idx is None:
        available = [h for h in header_row if h is not None]
        return {"error": "column_not_found", "column": column, "available": available}

    # Collect rows that satisfy the filter condition (needed for count and fallback).
    matching = []
    op = (operator or '=').lower().strip()
    for row in ws.iter_rows(min_row=2, values_only=True):
        cell_val = row[col_idx - 1]
        if _match_filter(cell_val, op, value):
            matching.append(list(row))

    if not matching:
        msg = f"No rows found where {column} {operator} {value}."
        return {"message": "filter_no_results", "count": 0, "speech_text": msg}

    count = len(matching)

    # ── COM live path: apply a visible Excel AutoFilter when the file is open ──
    # This avoids the _atomic_save / WinError 5 race and makes the filter
    # appear immediately in the Excel UI without a close/reopen cycle.
    try:
        abs_path = os.path.abspath(path)
    except Exception:
        abs_path = path
    com_result = _com_live_filter_table(abs_path, col_idx, operator, value, ws.title)
    if com_result:
        msg = (
            f"Filtered to {count} row{'s' if count != 1 else ''} "
            f"where {column} {operator} {value}."
        )
        return {
            "message": "filter_applied",
            "count": count,
            "_com_used": True,
            "speech_text": msg,
        }

    # ── Fallback: openpyxl Filtered sheet (when COM is unavailable) ──────────
    filtered_name = 'Filtered'
    if filtered_name in wb.sheetnames:
        del wb[filtered_name]
    ws_out = wb.create_sheet(title=filtered_name)
    ws_out.append(header_row)
    for row_data in matching:
        ws_out.append(row_data)

    # Switch active sheet to Filtered so Excel opens there after COM refresh.
    wb.active = wb.sheetnames.index(filtered_name)
    _mark_dirty()
    _flush_session(path)
    msg = (
        f"Found {count} row{'s' if count != 1 else ''} where {column} {operator} {value}. "
        "Results are in the Filtered sheet."
    )
    return {
        "message": "filter_applied",
        "count": count,
        "sheet": filtered_name,
        "speech_text": msg,
    }


def _match_filter(cell_val, operator: str, target) -> bool:
    """Return True when *cell_val* satisfies *operator* against *target*.

    Tries numeric comparison first; falls back to case-insensitive string
    comparison so commands like "filter Status = Active" work regardless of
    whether the column contains numbers or text.
    """
    if operator == 'contains':
        return target is not None and str(target).lower() in str(cell_val or '').lower()
    # Attempt numeric comparison; fall back to string if either side is non-numeric.
    try:
        cv = float(cell_val) if cell_val is not None else None
        tv = float(target) if target is not None else None
        if cv is None or tv is None:
            raise ValueError("non-numeric")
    except (ValueError, TypeError):
        cv = str(cell_val or '').lower().strip()
        tv = str(target or '').lower().strip()
    if operator == '=':
        return cv == tv
    if operator == '!=':
        return cv != tv
    if operator == '>':
        return cv > tv
    if operator == '<':
        return cv < tv
    if operator == '>=':
        return cv >= tv
    if operator == '<=':
        return cv <= tv
    # Unknown operator — fall back to equality.
    return str(cv) == str(tv)


def _com_live_clear_filter(abs_path: str, sheet_name=None) -> dict | None:
    """Clear the Excel AutoFilter on the live workbook via COM and save.

    Returns a result dict on success, or None when COM is unavailable or the
    workbook is not open in Excel (caller falls through to openpyxl path).
    Never raises — all COM exceptions are caught and logged.
    """
    try:
        import pythoncom          # type: ignore[import]
        try:
            pythoncom.CoInitialize()
        except Exception:
            pass
        xl_wb = _com_find_workbook(abs_path)
        if xl_wb is None:
            return None

        xl = xl_wb.Application
        xl.DisplayAlerts = False
        xl_ws = xl_wb.Worksheets(sheet_name) if sheet_name else xl_wb.ActiveSheet

        try:
            if xl_ws.AutoFilterMode:
                xl_ws.AutoFilterMode = False
        except Exception:
            pass  # Sheet may not support AutoFilter; treat as already clear

        xl_wb.Save()
        log.info('[COM] live clear_filter: sheet=%s  path=%s', sheet_name, abs_path)
        return {'_com_used': True}
    except Exception as exc:
        log.warning('[COM] _com_live_clear_filter failed: %s', exc)
        return None


def clear_filter(path, sheet_name=None):
    """Clear the active filter — either a COM AutoFilter or the openpyxl Filtered sheet."""
    log.info('clear_filter: path=%s', path)

    # ── COM live path: remove AutoFilter directly when Excel is open ─────────
    abs_path = os.path.abspath(path)
    com_result = _com_live_clear_filter(abs_path, sheet_name)
    if com_result:
        with _session_lock:
            if _session is not None and _session.abs_path == abs_path:
                _session.dirty = False
        return {"message": "filter_cleared", "speech_text": "Filter cleared. Showing all data."}

    # ── Fallback: delete the openpyxl 'Filtered' sheet (COM unavailable) ─────
    wb, ws = ensure_workbook(path, sheet_name)
    if isinstance(ws, dict):
        return ws
    filtered_name = 'Filtered'
    if filtered_name not in wb.sheetnames:
        return {"message": "no_filter_active", "speech_text": "There is no active filter to clear."}
    del wb[filtered_name]
    wb.active = 0
    _mark_dirty()
    _flush_session(path)
    return {"message": "filter_cleared", "speech_text": "Filter cleared. Showing all data."}


def execute_action(action, path):
    """Dispatch an action dict to the corresponding Excel function.

    Expected action format examples:
      {'type': 'create_excel', 'headers': ['Name','Score']}
      {'type': 'add_row', 'values': ['Tom', 90]}
      {'type': 'delete_row', 'index': 2}
      {'type': 'add_column', 'name': 'score'}
    """
    if not isinstance(action, dict) or not action.get('type'):
        log.error('execute_action: invalid action=%s', action)
        return {"error": "invalid_action", "detail": "action must be a dict with a 'type' key"}
    atype = action.get('type')
    log.info('execute_action: type=%s  path=%s', atype, path)
    if atype == 'create_excel':
        headers = action.get('headers')
        return create_excel(path, headers=headers)
    if atype == 'create_sheet_and_add_row':
        # Compound action: create workbook with headers, then immediately add a data row.
        headers = action.get('headers')
        values = action.get('values', [])
        result = create_excel(path, headers=headers)
        if 'error' in result:
            return result
        row_result = add_row(path, values)
        if 'error' in row_result:
            return row_result
        return {
            "message": "sheet_created_and_row_added",
            "headers": headers,
            "values": row_result.get("values", values),
            "speech_text": f"Created sheet with headers {headers} and added row {row_result.get('values', values)}.",
        }
    if atype == 'add_row':
        values = action.get('values', [])
        return add_row(path, values, sheet_name=action.get('sheet_name'))
    if atype == 'delete_row':
        index = int(action.get('index', 0))
        return delete_row(path, index, sheet_name=action.get('sheet_name'))
    if atype == 'insert_row_at':
        return insert_row_at(path, action.get('index'), sheet_name=action.get('sheet_name'))
    if atype == 'insert_column_at':
        return insert_column_at(path, action.get('column'), sheet_name=action.get('sheet_name'))
    if atype == 'add_column':
        name = action.get('name')
        default = action.get('default')
        return add_column(path, name, default, sheet_name=action.get('sheet_name'))
    if atype == 'save':
        return save(path)

    if atype == 'delete_column':
        col = action.get('column') or action.get('col')
        return delete_column(path, col)

    if atype == 'write_cell' or atype == 'insert_cell':
        cell = action.get('cell') or action.get('range')
        value = action.get('value') or action.get('text')
        return write_cell(path, cell, value)

    if atype == 'write_formula':
        cell = action.get('cell') or action.get('range')
        formula = action.get('formula') or action.get('value')
        return write_formula(path, cell, formula)

    if atype == 'calculate_row_sum':
        idx = action.get('index') or action.get('row')
        return calculate_row_sum(path, idx)

    if atype == 'calculate_average':
        col = action.get('column') or action.get('col')
        return calculate_column_average(path, col)

    if atype == 'open_file':
        return open_excel(path)

    if atype == 'close_excel_app':
        return close_excel(path)

    if atype == 'format_cells':
        # Accept both 'range' and 'cell_range' keys from the LLM output.
        cell_range_val = action.get('range') or action.get('cell_range')
        return format_cells(
            path,
            cell_range=cell_range_val,
            bold=action.get('bold'),
            italic=action.get('italic'),
            font_size=action.get('font_size'),
            font_color=action.get('font_color'),
            fill_color=action.get('fill_color'),
            num_format=action.get('num_format') or action.get('number_format'),
        )

    if atype == 'create_chart':
        return create_chart(
            path,
            chart_type=action.get('chart_type', 'bar'),
            data_range=action.get('data_range'),
            title=action.get('title'),
            x_axis=action.get('x_axis'),
            y_axis=action.get('y_axis'),
            position=action.get('position', 'E2'),
        )

    if atype == 'create_sheet':
        return create_sheet(path, action.get('sheet_name'))

    if atype == 'switch_sheet':
        return switch_sheet(path, action.get('sheet_name'))

    if atype == 'delete_sheet':
        return delete_sheet(path, action.get('sheet_name'))

    if atype == 'list_sheets':
        return list_sheets(path)

    if atype == 'find_replace':
        return find_and_replace(
            path,
            find_value=action.get('find_value') or action.get('find'),
            replace_value=action.get('replace_value') or action.get('replace'),
            sheet_name=action.get('sheet_name'),
        )

    if atype == 'sort_range':
        return sort_range(
            path,
            column=action.get('column') or action.get('col'),
            ascending=action.get('ascending', True),
            header_row=action.get('header_row', True),
            sheet_name=action.get('sheet_name'),
        )

    if atype == 'set_column_width':
        if action.get('auto'):
            return auto_fit_column(
                path,
                column=action.get('column') or action.get('col'),
                sheet_name=action.get('sheet_name'),
            )
        return set_column_width(
            path,
            column=action.get('column') or action.get('col'),
            width=action.get('width', 15),
            sheet_name=action.get('sheet_name'),
        )

    if atype == 'freeze_panes':
        return freeze_panes(
            path,
            cell=action.get('cell', 'B2'),
            sheet_name=action.get('sheet_name'),
        )

    if atype == 'merge_cells':
        return merge_cells(
            path,
            cell_range=action.get('range') or action.get('cell_range'),
            sheet_name=action.get('sheet_name'),
        )

    if atype == 'unmerge_cells':
        return unmerge_cells(
            path,
            cell_range=action.get('range') or action.get('cell_range'),
            sheet_name=action.get('sheet_name'),
        )

    if atype == 'rename_sheet':
        return rename_sheet(
            path,
            old_name=action.get('old_name'),
            new_name=action.get('new_name'),
        )

    if atype == 'read_range':
        return read_range(
            path,
            cell_range=action.get('range') or action.get('cell_range'),
            sheet_name=action.get('sheet_name'),
        )

    if atype == 'filter_table':
        return filter_table(
            path,
            column=action.get('column'),
            operator=action.get('operator', '='),
            value=action.get('value'),
            sheet_name=action.get('sheet_name'),
        )

    if atype == 'clear_filter':
        return clear_filter(path, sheet_name=action.get('sheet_name'))

    return {"error": "unknown_action", "action": action}
