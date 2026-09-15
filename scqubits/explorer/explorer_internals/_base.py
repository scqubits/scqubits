"""PanelBuilder protocol for per-PlotType dispatch in the Explorer widget.

Each concrete builder lives in its own file under
``scqubits/explorer/explorer_internals/`` and is registered in
``scqubits.explorer.explorer_internals.__init__``'s
:data:`PANEL_BUILDERS` map.  Two responsibilities per builder:

* :meth:`PanelBuilder.build_panel` renders the actual matplotlib
  artists into a pre-allocated ``(Figure, Axes)`` slot, taking the
  current ``ParameterSlice`` and consulting any persistent settings
  state on the ``Explorer``.
* :meth:`PanelBuilder.build_settings_ui` builds the per-plot
  settings widget list and registers any per-``PlotID`` state on the
  ``ExplorerSettings`` instance (e.g. level sliders).

The protocol is ``@runtime_checkable`` so the test suite can assert
that every entry in :data:`PANEL_BUILDERS` satisfies the contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from scqubits.core.param_sweep import ParameterSlice
    from scqubits.explorer.explorer_settings import ExplorerSettings
    from scqubits.explorer.explorer_widget import Explorer, PlotID
    from scqubits.ui.gui_defaults import PlotType


@runtime_checkable
class PanelBuilder(Protocol):
    """Per-``PlotType`` strategy bundling panel rendering and settings UI."""

    plot_type: ClassVar["PlotType"]

    def build_panel(
        self,
        explorer: "Explorer",
        plot_id: "PlotID",
        param_slice: "ParameterSlice",
        fig_ax: "tuple[Figure, Axes]",
    ) -> "tuple[Figure, Axes]":
        """Render this builder's panel into ``fig_ax`` and return it."""
        ...

    def build_settings_ui(
        self,
        settings: "ExplorerSettings",
        plot_id: "PlotID",
    ) -> list[Any]:
        """Build the per-plot settings widget list for this builder."""
        ...
