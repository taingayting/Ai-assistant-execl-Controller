"""Sara tool system.

Provides a :class:`ToolRegistry` that decouples action dispatch from the
concrete tool implementations (Excel, future Word/PowerPoint/DB tools).

Quick start::

    from tools import registry
    result = registry.execute("create_excel", path="out.xlsx", headers=["A","B"])
"""

from tools.tool_registry import ToolRegistry, registry, register_default_excel_tools

__all__ = ["ToolRegistry", "registry", "register_default_excel_tools"]
