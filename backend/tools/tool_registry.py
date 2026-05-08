"""Extensible tool registry for Sara.

The :class:`ToolRegistry` allows registering callable tools by name and
executing them uniformly via :meth:`execute`.  This decouples the Agent
from concrete tool implementations so that new tool families (Word,
PowerPoint, database, etc.) can be added without modifying Agent code.

Usage::

    from tools.tool_registry import registry, register_default_excel_tools

    register_default_excel_tools()          # one-time setup
    result = registry.execute("add_row", path="data.xlsx", values=["Tom", 90])

Each tool is simply a callable with the signature ``(path, **kwargs) -> dict``.
The *path* argument is always the target file path; additional keyword args
depend on the specific tool.
"""
from __future__ import annotations

import logging
import time
from typing import Callable

log = logging.getLogger('tool_registry')


class ToolNotRegisteredError(KeyError):
    """Raised when attempting to execute a tool that has not been registered."""


class ToolRegistry:
    """Central registry that maps tool names to callables.

    Tools are registered with :meth:`register` and invoked via
    :meth:`execute`.  The registry is intentionally generic so it can
    hold Excel tools, future Word/PowerPoint tools, database tools, etc.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., dict]] = {}

    # ---- registration ------------------------------------------------

    def register(self, name: str, fn: Callable[..., dict]) -> None:
        """Register *fn* under *name*.

        If *name* is already registered the previous entry is silently
        replaced (useful during hot-reload or tests).
        """
        if not name or not isinstance(name, str):
            raise ValueError("tool name must be a non-empty string")
        if not callable(fn):
            raise TypeError(f"fn must be callable, got {type(fn).__name__}")
        log.debug('register: %s -> %s', name, fn.__qualname__)
        self._tools[name] = fn

    def register_many(self, mapping: dict[str, Callable[..., dict]]) -> None:
        """Convenience: register several tools from a ``{name: fn}`` dict."""
        for name, fn in mapping.items():
            self.register(name, fn)

    # ---- execution ---------------------------------------------------

    def execute(self, tool_name: str, **kwargs) -> dict:
        """Execute the tool registered under *tool_name* with the given kwargs.

        Records per-tool latency at INFO level so thesis metrics can track
        individual tool performance without extra instrumentation.

        Raises :class:`ToolNotRegisteredError` if *tool_name* is unknown.
        All other exceptions from the tool implementation propagate unchanged
        so callers (Agent._dispatch_single) can handle them consistently.
        """
        fn = self._tools.get(tool_name)
        if fn is None:
            raise ToolNotRegisteredError(
                f"Tool '{tool_name}' is not registered. "
                f"Available tools: {', '.join(sorted(self._tools)) or '(none)'}"
            )
        _t0 = time.perf_counter()
        try:
            result = fn(**kwargs)
            _ms = (time.perf_counter() - _t0) * 1000
            _ok = isinstance(result, dict) and 'error' not in result
            log.info(
                'execute: tool=%-28s  latency=%.1f ms  success=%s',
                tool_name, _ms, _ok,
            )
            return result
        except Exception as exc:
            _ms = (time.perf_counter() - _t0) * 1000
            log.error(
                'execute: tool=%-28s  latency=%.1f ms  EXCEPTION: %s',
                tool_name, _ms, exc,
            )
            raise

    # ---- introspection -----------------------------------------------

    def list_tools(self) -> list[str]:
        """Return sorted list of registered tool names."""
        return sorted(self._tools)

    def is_registered(self, name: str) -> bool:
        """Return True if *name* is a registered tool."""
        return name in self._tools

    def assert_registered(self, name: str) -> None:
        """Raise :class:`ToolNotRegisteredError` if *name* is not registered.

        Use this as a strict guard when you want to explicitly reject hallucinated
        tool names rather than silently falling through to a legacy code path.
        """
        if name not in self._tools:
            raise ToolNotRegisteredError(
                f"Tool '{name}' is not registered. "
                f"Available tools: {', '.join(sorted(self._tools)) or '(none)'}"
            )

    def __contains__(self, name: str) -> bool:
        return self.is_registered(name)

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"<ToolRegistry tools={self.list_tools()}>"


# ---------- module-level singleton ----------

registry = ToolRegistry()


# ---------- Excel tool adapters ----------
# Each adapter has a uniform ``(path, **kw) -> dict`` signature so the
# registry can call them generically.  They delegate to the real
# functions in ``excel_controller``.

def _excel_create(path: str, *, headers=None, **_kw) -> dict:
    from excel_controller import create_excel
    return create_excel(path, headers=headers)


def _excel_add_row(path: str, *, values, **_kw) -> dict:
    from excel_controller import add_row
    _require('values', values, 'add_row')
    # Flatten one level of nesting: [['Samurai', 1], 0] → ['Samurai', 1, 0]
    flat = []
    for v in values:
        if isinstance(v, list):
            flat.extend(v)
        else:
            flat.append(v)
    return add_row(path, flat)


def _excel_delete_row(path: str, *, index: int, **_kw) -> dict:
    from excel_controller import delete_row
    if index is None:
        raise ValueError("Tool 'delete_row' requires parameter 'index'.")
    return delete_row(path, index)


def _excel_insert_row_at(path: str, *, index, **_kw) -> dict:
    from excel_controller import insert_row_at
    _require('index', index, 'insert_row_at')
    return insert_row_at(path, index)


def _excel_insert_column_at(path: str, *, column, **_kw) -> dict:
    from excel_controller import insert_column_at
    _require('column', column, 'insert_column_at')
    return insert_column_at(path, column)


def _excel_delete_column(path: str, *, column, **_kw) -> dict:
    from excel_controller import delete_column
    _require('column', column, 'delete_column')
    return delete_column(path, column)


def _excel_delete_cell(path: str, *, cell: str, **_kw) -> dict:
    from excel_controller import delete_cell
    _require('cell', cell, 'delete_cell')
    return delete_cell(path, cell)


def _excel_delete_range(path: str, *, range: str = None, cell_range: str = None, **_kw) -> dict:
    from excel_controller import delete_range
    rng = range or cell_range
    _require('range', rng, 'delete_range')
    return delete_range(path, rng)


def _excel_add_column(path: str, *, name: str, default=None, **_kw) -> dict:
    from excel_controller import add_column
    _require('name', name, 'add_column')
    return add_column(path, name, default)


def _excel_write_cell(path: str, *, cell: str, value, **_kw) -> dict:
    from excel_controller import write_cell
    _require('cell', cell, 'write_cell')
    if value is None:
        raise ValueError("Tool 'write_cell' requires parameter 'value'.")
    return write_cell(path, cell, value)


def _excel_write_formula(path: str, *, cell: str, formula: str, **_kw) -> dict:
    from excel_controller import write_formula
    from pipeline_sanitizer import is_formula_safe
    _require('cell', cell, 'write_formula')
    _require('formula', formula, 'write_formula')
    if not is_formula_safe(formula):
        return {"error": "formula_blocked",
                "detail": "Formula contains a function that is not on the allowed list.",
                "speech_text": "Sorry, that formula contains a function I'm not allowed to write."}
    return write_formula(path, cell, formula)


def _excel_write_sum_formula(path: str, *, range_ref: str, cell: str, **_kw) -> dict:
    from excel_controller import write_sum_formula
    _require('range_ref', range_ref, 'write_sum_formula')
    _require('cell', cell, 'write_sum_formula')
    return write_sum_formula(path, range_ref, cell)


def _excel_calculate_row_sum(path: str, *, index: int, **_kw) -> dict:
    from excel_controller import calculate_row_sum
    if index is None:
        raise ValueError("Tool 'calculate_row_sum' requires parameter 'index'.")
    return calculate_row_sum(path, index)


def _excel_write_average_formula(path: str, *, range_ref: str, cell: str, **_kw) -> dict:
    from excel_controller import write_average_formula
    _require('range_ref', range_ref, 'write_average_formula')
    _require('cell', cell, 'write_average_formula')
    return write_average_formula(path, range_ref, cell)


def _excel_calculate_average(path: str, *, column=None, **_kw) -> dict:
    from excel_controller import calculate_column_average
    # column is optional — auto-detects first numeric column when None
    return calculate_column_average(path, column)


def _excel_open_file(path: str, **_kw) -> dict:
    import os, sys, subprocess
    from excel_controller import open_excel
    result = open_excel(path)
    # Also open the file visually in the OS default application (Excel on Windows).
    if 'error' not in result and os.path.exists(path):
        try:
            if sys.platform == 'win32':
                os.startfile(os.path.abspath(path))
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', os.path.abspath(path)])
            else:
                subprocess.Popen(['xdg-open', os.path.abspath(path)])
        except Exception:
            pass  # Non-fatal: metadata result is still returned
    return result


def _excel_open_app(path: str = '', **_kw) -> dict:
    """Launch the Microsoft Excel application.

    If there is an active working file in memory, open that file directly so the
    user sees the same workbook that Sara is writing to.  This fixes the common
    issue where "open excel" launched a blank Excel window while Sara's writes
    were going to a different file on disk.
    """
    import os, sys, subprocess
    import logging
    _log = logging.getLogger('tool_registry')

    if sys.platform != 'win32':
        return {'error': 'open_excel_app is only supported on Windows',
                'speech_text': 'This command is only supported on Windows.'}

    # ── Prefer opening the active working file directly ───────────────────────
    # When Sara has an active file (set by create_excel, open_file, etc.) open
    # that specific .xlsx so the user sees the data that Sara is modifying.
    _active_path = path or ''
    if not _active_path:
        try:
            from memory import memory as _mem
            _active_path = _mem.snapshot().get('active_file') or ''
        except Exception:
            pass

    if _active_path and os.path.exists(_active_path):
        try:
            abs_active = os.path.abspath(_active_path)
            os.startfile(abs_active)
            basename = os.path.basename(abs_active)
            _log.info('[OpenApp] opened active file in Excel: %s', abs_active)
            return {'message': 'excel_file_opened', 'path': abs_active,
                    'speech_text': f'Opening {basename} in Excel.'}
        except Exception as exc:
            _log.warning('[OpenApp] os.startfile failed for %s: %s — falling back to blank Excel', _active_path, exc)

    # ── Fallback: launch blank Excel when no active file is known ─────────────
    _candidates = [
        r'C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE',
        r'C:\Program Files (x86)\Microsoft Office\root\Office16\EXCEL.EXE',
        r'C:\Program Files\Microsoft Office\Office16\EXCEL.EXE',
        r'C:\Program Files (x86)\Microsoft Office\Office16\EXCEL.EXE',
        r'C:\Program Files\Microsoft Office\root\Office15\EXCEL.EXE',
        r'C:\Program Files (x86)\Microsoft Office\root\Office15\EXCEL.EXE',
    ]

    for _path in _candidates:
        if os.path.exists(_path):
            try:
                subprocess.Popen([_path])
                return {'message': 'excel_app_opened', 'path': _path,
                        'speech_text': 'Opening Excel.'}
            except Exception as exc:
                return {'error': str(exc), 'speech_text': f'Could not open Excel: {exc}'}

    try:
        subprocess.Popen('excel', shell=True)
        return {'message': 'excel_app_opened', 'speech_text': 'Opening Excel.'}
    except Exception:
        pass

    try:
        os.startfile('excel')
        return {'message': 'excel_app_opened', 'speech_text': 'Opening Excel.'}
    except Exception as exc:
        return {'error': f'Excel not found: {exc}',
                'speech_text': 'I could not find Microsoft Excel on this computer.'}


def _excel_format_cells(path: str, *, cell_range: str, bold=None, italic=None,
                        font_size=None, font_color=None, fill_color=None,
                        num_format=None, align=None, font_size_change=None,
                        **_kw) -> dict:
    from excel_controller import format_cells
    _require('cell_range', cell_range, 'format_cells')
    return format_cells(path, cell_range, bold=bold, italic=italic,
                        font_size=font_size, font_color=font_color,
                        fill_color=fill_color, num_format=num_format,
                        align=align, font_size_change=font_size_change)


def _excel_create_chart(path: str, *, chart_type: str = 'bar', data_range: str = None,
                        title=None, x_axis=None, y_axis=None,
                        position: str = 'E2', **_kw) -> dict:
    from excel_controller import create_chart
    # data_range is optional: create_chart auto-detects sheet extent when None.
    return create_chart(path, chart_type=chart_type, data_range=data_range,
                        title=title, x_axis=x_axis, y_axis=y_axis,
                        position=position)


def _excel_save(path: str, **_kw) -> dict:
    from excel_controller import save
    return save(path)


def _excel_create_sheet(path: str, *, sheet_name: str, **_kw) -> dict:
    from excel_controller import create_sheet
    return create_sheet(path, sheet_name)


def _excel_switch_sheet(path: str, *, sheet_name: str, **_kw) -> dict:
    from excel_controller import switch_sheet
    return switch_sheet(path, sheet_name)


def _excel_delete_sheet(path: str, *, sheet_name: str, **_kw) -> dict:
    from excel_controller import delete_sheet
    return delete_sheet(path, sheet_name)


def _excel_list_sheets(path: str, **_kw) -> dict:
    from excel_controller import list_sheets
    return list_sheets(path)


def _require(name: str, value, tool: str) -> None:
    """Raise ValueError if *value* is None or empty, with a clear message."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"Tool '{tool}' requires parameter '{name}' but it was not provided.")


def _excel_find_replace(path: str, *, find_value, replace_value, **_kw) -> dict:
    from excel_controller import find_and_replace
    _require('find_value', find_value, 'find_replace')
    return find_and_replace(path, find_value=find_value, replace_value=replace_value)


def _excel_sort_range(path: str, *, column, ascending=True, **_kw) -> dict:
    from excel_controller import sort_range
    _require('column', column, 'sort_range')
    return sort_range(path, column=column, ascending=ascending)


def _excel_set_column_width(path: str, *, column, width=15, auto=False, **_kw) -> dict:
    from excel_controller import set_column_width, auto_fit_column
    _require('column', column, 'set_column_width')
    if auto:
        return auto_fit_column(path, column=column)
    return set_column_width(path, column=column, width=width)


def _excel_freeze_panes(path: str, *, cell='B2', **_kw) -> dict:
    from excel_controller import freeze_panes
    return freeze_panes(path, cell=cell or 'B2')


def _excel_unfreeze_panes(path: str, **_kw) -> dict:
    from excel_controller import unfreeze_panes
    return unfreeze_panes(path)


def _excel_merge_cells(path: str, *, cell_range, **_kw) -> dict:
    from excel_controller import merge_cells
    _require('cell_range', cell_range, 'merge_cells')
    return merge_cells(path, cell_range=cell_range)


def _excel_unmerge_cells(path: str, *, cell_range, **_kw) -> dict:
    from excel_controller import unmerge_cells
    _require('cell_range', cell_range, 'unmerge_cells')
    return unmerge_cells(path, cell_range=cell_range)


def _excel_rename_sheet(path: str, *, old_name, new_name, **_kw) -> dict:
    from excel_controller import rename_sheet
    _require('old_name', old_name, 'rename_sheet')
    _require('new_name', new_name, 'rename_sheet')
    return rename_sheet(path, old_name=old_name, new_name=new_name)


def _excel_rename_file(path: str, *, new_name, **_kw) -> dict:
    from excel_controller import rename_file
    _require('new_name', new_name, 'rename_file')
    return rename_file(path, new_name=new_name)


def _excel_read_range(path: str, *, cell_range, **_kw) -> dict:
    from excel_controller import read_range
    _require('cell_range', cell_range, 'read_range')
    return read_range(path, cell_range=cell_range)


def _excel_read_cell(path: str, *, cell: str, **_kw) -> dict:
    from excel_controller import read_cell
    _require('cell', cell, 'read_cell')
    return read_cell(path, cell=cell)


def _excel_undo_last(path: str, **_kw) -> dict:
    from excel_controller import undo_last
    return undo_last(path)


def _excel_close(path: str, **_kw) -> dict:
    from excel_controller import close_excel
    return close_excel(path)


def _excel_filter_table(path: str, *, column, operator='=', value=None, **_kw) -> dict:
    """Apply a filter to the active sheet and write matching rows to a 'Filtered' sheet."""
    from excel_controller import filter_table
    _require('column', column, 'filter_table')
    return filter_table(path, column=column, operator=operator, value=value)


def _excel_clear_filter(path: str, **_kw) -> dict:
    """Delete the Filtered sheet and return Excel focus to the source sheet."""
    from excel_controller import clear_filter
    return clear_filter(path)


def _excel_create_sheet_and_add_row(path: str, *, headers, values,
                                    sheet_name=None, **_kw) -> dict:
    from excel_controller import create_excel, add_row
    result = create_excel(path, headers=headers, sheet_name=sheet_name)
    if isinstance(result, dict) and 'error' in result:
        return result
    row_result = add_row(path, values)
    if isinstance(row_result, dict) and 'error' in row_result:
        return row_result
    final_values = row_result.get("values", values)
    # Build natural speech: "All set! I've created the sheet with columns Name, Score
    # and added the first row: Tom, 90."
    col_str = ', '.join(str(h) for h in headers) if headers else '(none)'
    val_str = ', '.join(str(v) for v in final_values[:4])
    if len(final_values) > 4:
        val_str += f' and {len(final_values) - 4} more'
    return {
        "message": "sheet_created_and_row_added",
        "headers": headers,
        "values": final_values,
        "speech_text": (
            f"All set! I've created the spreadsheet with columns {col_str} "
            f"and added the first row: {val_str}."
        ),
    }


# ---------- Default registration ----------

def register_default_excel_tools(target: ToolRegistry | None = None) -> None:
    """Register all built-in Excel tools on *target* (default: module singleton).

    Call this once at application startup (e.g. in ``app.py`` or
    ``assistant_runtime.py``) before the first ``registry.execute()`` call.
    """
    reg = target or registry
    reg.register_many({
        'create_excel':             _excel_create,
        'create_sheet_and_add_row': _excel_create_sheet_and_add_row,
        'add_row':                  _excel_add_row,
        'delete_row':               _excel_delete_row,
        'insert_row_at':            _excel_insert_row_at,
        'insert_column_at':         _excel_insert_column_at,
        'delete_column':            _excel_delete_column,
        'delete_cell':              _excel_delete_cell,
        'delete_range':             _excel_delete_range,
        'add_column':               _excel_add_column,
        'write_cell':               _excel_write_cell,
        'insert_cell':              _excel_write_cell,       # alias
        'write_formula':            _excel_write_formula,
        'write_sum_formula':        _excel_write_sum_formula,
        'write_average_formula':    _excel_write_average_formula,
        'calculate_row_sum':        _excel_calculate_row_sum,
        'calculate_average':        _excel_calculate_average,
        'open_file':                _excel_open_file,
        'open_excel_app':           _excel_open_app,
        'format_cells':             _excel_format_cells,
        'create_chart':             _excel_create_chart,
        'save':                     _excel_save,
        'create_sheet':             _excel_create_sheet,
        'switch_sheet':             _excel_switch_sheet,
        'delete_sheet':             _excel_delete_sheet,
        'list_sheets':              _excel_list_sheets,
        # extended operations (added in v2)
        'find_replace':             _excel_find_replace,
        'sort_range':               _excel_sort_range,
        'set_column_width':         _excel_set_column_width,
        'freeze_panes':             _excel_freeze_panes,
        'unfreeze_panes':           _excel_unfreeze_panes,
        'merge_cells':              _excel_merge_cells,
        'unmerge_cells':            _excel_unmerge_cells,
        'rename_sheet':             _excel_rename_sheet,
        'rename_file':              _excel_rename_file,
        'read_range':               _excel_read_range,
        # new in this session
        'read_cell':                _excel_read_cell,
        'undo':                     _excel_undo_last,
        'undo_last':                _excel_undo_last,   # alias
        'filter_table':             _excel_filter_table,
        'clear_filter':             _excel_clear_filter,
        'close_excel_app':          _excel_close,
    })
    log.info('Registered %d default Excel tools', len(reg))
