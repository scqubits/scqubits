"""Typed state container for the Explorer widget's UI references.

The ``ExplorerUI`` dataclass replaces the legacy
``Explorer.ui: dict[str, Any]`` keyed by string literals.  Switching
to a dataclass gives typo detection (a misspelled attribute access
raises ``AttributeError`` instead of returning ``None`` from a
missing dict key) and documents the full set of widgets the Explorer
keeps cached references to.

Field types are intentionally ``Any`` for the widget references --
the ipyvuetify widgets do not carry static type information that
narrowing helps with, and the legacy dict was effectively ``Any``
already.  Dict-valued fields use ``field(default_factory=dict)`` so
the dataclass can be constructed empty and populated in-place by
``Explorer.__init__``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExplorerUI:
    """All UI-widget references the Explorer caches on ``self.ui``."""

    add_plot_dialog: Any = None
    sweep_param_dropdown: Any = None
    sweep_value_slider: Any = None
    param_sliders: dict[str, Any] = field(default_factory=dict)
    param_sliders_container: Any = None
    fixed_param_sliders: dict[str, Any] = field(default_factory=dict)
    top_bar: Any = None
    panel_switch_by_plot_id: dict[Any, Any] = field(default_factory=dict)
    panel_switches_by_subsys_name: dict[str, Any] = field(default_factory=dict)
    panel_switches: dict[str, Any] = field(default_factory=dict)
