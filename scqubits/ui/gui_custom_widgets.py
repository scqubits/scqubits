# gui_custom_widgets.py
#
# This file is part of scqubits: a Python package for superconducting qubits,
# Quantum 5, 583 (2021). https://quantum-journal.org/papers/q-2021-11-17-583/
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import collections

from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
    OrderedDict,
)

import matplotlib as mp

import scqubits.utils.misc as utils

try:
    import ipyvuetify as v
    import ipywidgets
    import traitlets
except ImportError:
    _HAS_IPYVUETIFY = False
else:
    _HAS_IPYVUETIFY = True

try:
    from IPython.display import display
except ImportError:
    _HAS_IPYTHON = False
else:
    _HAS_IPYTHON = True

if TYPE_CHECKING:
    from scqubits.explorer.explorer_widget import PlotID


if _HAS_IPYTHON and _HAS_IPYVUETIFY:

    class ValidatedNumberField(v.TextField):
        _typecheck_func: callable = None
        _type = None

        num_value = None  # must determine appropriate traitlet type dynamically

        @utils.Required(ipyvuetify=_HAS_IPYVUETIFY, IPython=_HAS_IPYTHON)
        def __init__(
            self,
            v_model,
            num_type=None,
            v_min=None,
            v_max=None,
            step=None,
            filled=True,
            **kwargs,
        ):
            self.name = kwargs.pop("name", None)
            self._type = num_type if num_type is not None else type(v_model)
            if num_type == float:
                TraitletClass = traitlets.Float
                self._typecheck_func = utils.is_string_float
                self.step = step if step is not None else 0.1
            elif num_type == int:
                TraitletClass = traitlets.Int
                self._typecheck_func = utils.is_string_int
                self.step = step if step is not None else 1
            else:
                raise Exception(f"Not a supported number type: {num_type}")
            self.add_traits(
                num_value=TraitletClass(read_only=True).tag(sync=True),
                v_min=TraitletClass(allow_none=True).tag(sync=True),
                v_max=TraitletClass(allow_none=True).tag(sync=True),
            )
            self.v_min = v_min
            self.v_max = v_max

            super().__init__(v_model=v_model, filled=filled, **kwargs)

        @traitlets.validate("v_model")
        def _validate_v_model(self, state):
            if self.is_valid():
                self.error = False
                self.rules = []
            else:
                self.error = True
                self.rules = ["invalid"]
            return state["value"]

        @traitlets.observe("v_model")
        def _observe_v_model(self, change):
            if not self.error:
                self.set_trait("num_value", self._type(change["new"]))

        def is_valid(self):
            if (
                not self._typecheck_func(self.v_model)
                or (
                    self.v_min not in [None, ""]
                    and self._type(self.v_model) < self.v_min
                )
                or (
                    self.v_max not in [None, ""]
                    and self._type(self.v_model) > self.v_max
                )
            ):
                return False
            return True

    class NumberEntryWidget(ValidatedNumberField):
        """A widget consisting of a text field and a slider, linked to each other. The text field acts as the main
        class, while the slider is stored as a class attribute and displayed alongside.
        """

        def __init__(
            self,
            label,
            v_model=None,
            num_type=float,
            step=None,
            v_min=None,
            v_max=None,
            s_min=None,
            s_max=None,
            text_kwargs=None,
            slider_kwargs=None,
        ):
            text_kwargs = text_kwargs or {}
            slider_kwargs = slider_kwargs or {}
            super().__init__(
                label=label,
                v_model=v_model,
                num_type=num_type,
                step=step,
                v_min=v_min,
                v_max=v_max,
                **text_kwargs,
            )

            if "style_" not in slider_kwargs:
                slider_kwargs["style_"] = "max-width: 240px; min-width: 220px;"
            if "class_" not in slider_kwargs:
                slider_kwargs["class_"] = "pt-3"
            self.slider = v.Slider(
                min=s_min,
                max=s_max,
                step=step,
                v_model=v_model,
                thumb_label=True,
                **slider_kwargs,
            )

            self.slider.on_event("change", self.slider_click)
            self.observe(self.update_slider, names="num_value")

            ipywidgets.jslink((self, "v_max"), (self.slider, "max"))
            ipywidgets.jslink((self, "v_min"), (self.slider, "min"))
            ipywidgets.jslink((self, "disabled"), (self.slider, "disabled"))

        def _ipython_display_(self):
            display(self.widget())

        def slider_click(self, *args):
            self.v_model = self.slider.v_model

        def widget(self):
            return v.Container(
                class_="d-flex flex-row ml-2 pb-0 pt-1",
                style_="min-width: 220px; max-width: 220px",
                children=[self, self.slider],
            )

        def update_slider(self, *args):
            if self.error:
                return
            if self.num_value > self.slider.max:
                self.slider.color = "red"
                self.slider.v_model = self.slider.max
            elif self.num_value < self.slider.min:
                self.slider.color = "red"
                self.slider.v_model = self.slider.min
            else:
                self.slider.color = ""
                self.slider.v_model = self.num_value

        def update_text(self, *args):
            self.v_model = self.slider.v_model

    class InitializedSelect(v.Select):
        def __init__(self, **kwargs):
            if "v_model" not in kwargs and "items" in kwargs:
                kwargs["v_model"] = kwargs["items"][0]
            super().__init__(**kwargs)

    class LinkedButton(v.Btn):
        def __init__(self, ref=None, **kwargs):
            onclick = kwargs.pop("onclick", None)
            super().__init__(**kwargs)
            self.ref = ref

            if onclick:
                self.on_event("click", onclick)

    class LinkedSwitch(v.Switch):
        def __init__(self, ref, **kwargs):
            super().__init__(**kwargs)
            self.ref = ref

    class vTooltipBtn(v.Tooltip):
        def __init__(self, tooltip, bottom=False, left=True, **kwargs):
            self.btn = LinkedButton(v_on="tooltip.on", **kwargs)
            super().__init__(
                bottom=bottom,
                left=left,
                v_slots=[
                    {"name": "activator", "variable": "tooltip", "children": self.btn}
                ],
                children=[tooltip],
            )

    class ClickChip(v.Chip):
        def __init__(self, **kwargs):
            onclick_close = kwargs.pop("click_close", None)
            onclick = kwargs.pop("onclick", None)
            super().__init__(**kwargs)

            if onclick_close:
                self.on_event("click:close", onclick_close)
            if onclick:
                self.on_event("click", onclick)

    class DiscreteSetSlider(v.Slider):
        def __init__(self, param_name, param_vals, **kwargs):
            self.val_count = len(param_vals)
            self.param_name = param_name
            self.param_vals = param_vals
            super().__init__(
                min=0, max=(self.val_count - 1), step=1, v_model=0, **kwargs
            )
            self.label = f"{self.param_name}={self.current_value():.3f}"

            self.observe(self.update_textfield, names="v_model")

        def current_value(self):
            return self.param_vals[int(self.v_model)]

        def update_textfield(self, *args):
            self.label = f"{self.param_name}={self.current_value():.3f}"

    class IconButton(LinkedButton):
        def __init__(self, icon_name, **kwargs):
            super().__init__(
                **kwargs,
                min_width=40,
                width=40,
                height=40,
                elevation="0",
                children=[v.Icon(children=[icon_name])],
            )

    def flex_row(widgets: List[v.VuetifyWidget], class_="", **kwargs) -> v.Container:
        return v.Container(
            class_="d-flex flex-row " + class_, children=widgets, **kwargs
        )

    def flex_column(widgets: List[v.VuetifyWidget], class_="", **kwargs) -> v.Container:
        return v.Container(
            class_="d-flex flex-column " + class_, children=widgets, **kwargs
        )

    class PanelBase:
        def __init__(self, panel_id=None, content_list=None, width="49.5%"):
            content_list = content_list if content_list else []
            self.content_row = v.Container(
                children=content_list,
                class_="d-flex flex-row",
                style_="justify-content: center",
            )
            self.card = v.Card(
                id=str(panel_id),
                max_width=width,
                min_width=width,
                elevation=2,
                class_="mx-1 my-1",
                children=[self.content_row],
            )
            self.panel_id = panel_id

        def set_content(self, content_list):
            self.content_row.children = content_list

    class ClosablePanel(PanelBase):
        def __init__(self, panel_id: "PlotID" = None, content_list=None, width="49.5%"):
            super().__init__(panel_id=panel_id, content_list=content_list, width=width)

            self.btn = v.Btn(
                class_="mx-1",
                icon=True,
                size="xx-small",
                elevation=0,
                children=[v.Icon(children="mdi-close-circle")],
            )

            self.settings_btn = LinkedButton(
                # class_="mx-1",
                style_="margin-left: auto;",
                icon=True,
                size="xx-small",
                elevation=0,
                children=[v.Icon(children="mdi-settings")],
                ref=panel_id,
            )

            self.title = v.CardTitle(
                style_="margin-left: auto;", children=[v.Html(children=[str(panel_id)])]
            )

            self.card.children = [
                v.Container(
                    class_="d-flex flex-row justify-center",
                    children=[self.title, self.settings_btn, self.btn],
                ),
                self.content_row,
            ]

    class ClosablePlotPanel(ClosablePanel):
        def __init__(
            self,
            fig: mp.figure.Figure,
            axes: mp.axes.Axes,
            panel_id: Optional["PlotID"] = None,
            width="49%",
        ):
            self.fig = fig
            self.axes = axes
            title = self.axes.title.get_text()
            self.axes.title.set_text("")

            self.output = ipywidgets.Output(
                layout=ipywidgets.Layout(object_fit="contain")
            )
            super().__init__(content_list=[self.output], panel_id=panel_id, width=width)
            self.title.children = [
                v.Html(
                    style_="font-weight: normal; color: 0.2;",
                    tag="div",
                    children=[title],
                )
            ]

    class PlotPanelCollection:
        def __init__(
            self,
            toggle_switches_by_plot_id: Dict["PlotID", v.Switch],
            ncols: int = 2,
            plot_choice_dialog: Callable = None,
            plot_settings_dialog: Callable = None,
        ):
            self.ncols = ncols
            self.plot_choice_dialog = plot_choice_dialog
            self.plot_settings_dialog = plot_settings_dialog
            self.panel_by_btn: OrderedDict[
                v.Btn, ClosablePlotPanel
            ] = collections.OrderedDict()
            self.panel_by_id: OrderedDict[
                str, ClosablePlotPanel
            ] = collections.OrderedDict()
            self.toggle_switches_by_plot_id = toggle_switches_by_plot_id

            self.container = v.Container(
                children=[],
                style_="background: #eeeeee; position: relative; top: -70px",
                class_="d-flex flex-row flex-wrap p-0 m-0",
                width="100%",
            )

        def choose_plot(self, *args, **kwargs):
            if self.plot_choice_dialog:
                self.plot_choice_dialog()

        def axes_list(self) -> List[mp.axes.Axes]:
            return [panel.axes for panel in self.panel_by_btn.values()]

        def card_list(self) -> List[v.Card]:
            """Returns the list of all the cards in the deck.

            Returns
            ------
                A list of the cards in the deck
            """
            cards = [panel.card for panel in self.panel_by_btn.values()]
            return cards

        def id_list(self) -> List[str]:
            """Returns the list of all the ids in the deck.

            Returns
            ------
                A list of the ids in the deck
            """
            return list(self.panel_by_id.keys())

        def setup_card(self, new_panel: ClosablePanel):
            """Sets up a new card, connecting its close button to the card collection's `close_card` method."""
            card = new_panel.card
            card.id = new_panel.panel_id
            btn = new_panel.btn
            self.panel_by_btn[btn] = new_panel
            self.panel_by_id[new_panel.panel_id] = new_panel
            btn.on_event("click", self.close_panel)
            new_panel.settings_btn.on_event("click", self.settings_dialog)
            self.container.children = self.card_list()

        def new_plot_panel(self, panel_id: "PlotID", fig, axes):
            """Adds a new plot card to the grid."""
            closable_panel = ClosablePlotPanel(
                fig, axes, panel_id=panel_id, width=f"{100 / self.ncols - 1}%"
            )
            self.setup_card(closable_panel)
            with closable_panel.output:
                mp.pyplot.show()

        def close_panel(self, close_btn: LinkedButton, *args, **kwargs):
            """Remove card from dictionary and from the grid."""
            panel_id = self.panel_by_btn[close_btn].panel_id

            if self.toggle_switches_by_plot_id[panel_id].v_model:
                self.toggle_switches_by_plot_id[panel_id].v_model = False
            else:
                self.panel_by_btn.pop(close_btn)
                self.panel_by_id.pop(panel_id)
                self.container.children = self.card_list()

        def close_panel_by_id(self, panel_id: "PlotID"):
            self.close_panel(self.panel_by_id[panel_id].btn)

        def settings_dialog(self, settings_btn: LinkedButton, *args, **kwargs):
            """Bring up plot panel settings dialog."""
            self.plot_settings_dialog(settings_btn.ref)

        def resize_all(self, width=None, height=None):
            """Resizes all cards in the grid.

            Parameters
            ----------
            width:
                width of the cards in the grid
            height:
                height of the cards in the grid
            """
            for card in self.card_list():
                if width:
                    card.width = width
                if height:
                    card.height = height

        def change_cols(self, ncols: int):
            """
            Changes the number of columns in the grid by setting `explorer.cols` and resizing all widgets in the grid.

            Parameters
            ----------
            ncols:
                number of columns in the grid

            """
            self.ncols = ncols
            self.resize_all(width=f"{100 / self.ncols - 1}%")

        def show(self):
            """
            The show function is the main function of a component. It returns
            a VDOM object that will be rendered in the browser. The show function
            is called every time an event occurs, and it's return value is used to
            update the DOM.

            Returns
            -------
                A container with the full widget.
            """
            return v.Container(
                class_="mx-0 px-0 my-0 py-0",
                width="100%",
                children=[
                    LinkedButton(
                        fab=True,
                        position="fixed",
                        class_="mx-2 my-2",
                        style_="z-index: 1000",
                        onclick=self.choose_plot,
                        children=[v.Icon(children="mdi-plus")],
                    ),
                    self.container,
                ],
            )
