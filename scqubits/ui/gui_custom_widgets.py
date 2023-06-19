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
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, OrderedDict, Union

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
    from scqubits.ui.explorer_widget import PlotID


if _HAS_IPYTHON and _HAS_IPYVUETIFY:

    class vValidatedNumberField(v.TextField):
        _typecheck_func: callable = None
        _type = None
        _current_value = None
        num_value = None

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

            self._current_value = v_model
            super().__init__(v_model=v_model, filled=filled, **kwargs)

            if num_type == float:
                TraitletClass = traitlets.Float
                self._typecheck_func = lambda data: utils.is_string_float(data)
                self.step = step if step is not None else 0.1
            elif num_type == int:
                TraitletClass = traitlets.Int
                self._typecheck_func = lambda data: utils.is_string_int(data)
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

            self.set_trait("num_value", self._num_value())

            self.observe(self.is_entry_valid, names="v_model")

        def is_entry_valid(self, *args, **kwargs):
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
                self.error = True
                self.rules = ["invalid"]
                return False

            self.rules = [True]
            self.error = False
            self.set_trait("num_value", self._num_value())
            return True

        def _num_value(self):
            if not self.error:
                self._current_value = self._type(self.v_model)
            return self._current_value

    class vNumberEntryWidget(vValidatedNumberField):
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
                min=s_min, max=s_max, step=step, v_model=v_model, **slider_kwargs
            )

            # self._continuous_update_in_progress = False
            # self.slider.on_event("start", self.slider_in_progress_toggle)
            # self.slider.on_event("end", self.slider_in_progress_toggle)
            # self.slider.on_event("click", self.slider_in_progress_toggle)
            # self.slider.observe(self.update_textfield, names="v_model")
            self.observe(self.update_slider, names="num_value")

            ipywidgets.jslink((self, "v_model"), (self.slider, "v_model"))

            ipywidgets.jslink((self, "v_max"), (self.slider, "max"))
            ipywidgets.jslink((self, "v_min"), (self.slider, "min"))
            ipywidgets.jslink((self, "disabled"), (self.slider, "disabled"))

        def _ipython_display_(self):
            display(self.widget())

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

    class vInitSelect(v.Select):
        def __init__(self, **kwargs):
            if "v_model" not in kwargs and "items" in kwargs:
                kwargs["v_model"] = kwargs["items"][0]
            super().__init__(**kwargs)

    class vBtn(v.Btn):
        def __init__(self, ref=None, **kwargs):
            onclick = kwargs.pop("onclick", None)
            super().__init__(**kwargs)
            self.ref = ref

            if onclick:
                self.on_event("click", onclick)


    class vRefSwitch(v.Switch):
        def __init__(self, ref, **kwargs):
            super().__init__(**kwargs)
            self.ref = ref


    class vTooltipBtn(v.Tooltip):
        def __init__(self, tooltip, bottom=False, left=True, **kwargs):
            self.btn = vBtn(v_on="tooltip.on", **kwargs)
            super().__init__(
                bottom=bottom,
                left=left,
                v_slots=[
                    {"name": "activator", "variable": "tooltip", "children": self.btn}
                ],
                children=[tooltip],
            )

    class vChip(v.Chip):
        def __init__(self, **kwargs):
            onclick_close = kwargs.pop("click_close", None)
            onclick = kwargs.pop("onclick", None)
            super().__init__(**kwargs)

            if onclick_close:
                self.on_event("click:close", onclick_close)
            if onclick:
                self.on_event("click", onclick)

    class vDiscreteSetSlider(v.Slider):
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


    class IconButton(vBtn):
        def __init__(self, icon_name, **kwargs):
            super().__init__(
                **kwargs,
                min_width=40,
                width=40,
                height=40,
                elevation="0",
                children=[v.Icon(children=[icon_name])],
            )

    class vNavbarElement(v.ExpansionPanels):
        def __init__(
            self,
            header,
            content: Union[None, v.ExpansionPanelContent] = None,
            children: Union[None, List[v.VuetifyWidget]] = None,
            **kwargs,
        ):
            assert (content and not children) or (children and not content)

            content = (
                content
                if isinstance(content, v.ExpansionPanelContent)
                else v.ExpansionPanelContent(
                    class_="text-no-wrap",
                    style_="transform: scale(0.9)",
                    children=children,
                )
            )

            super().__init__(
                **kwargs,
                **dict(
                    transition=False,
                    flat=True,
                    v_model=None,
                    children=[
                        v.ExpansionPanel(
                            accordion=True,
                            children=[
                                v.ExpansionPanelHeader(
                                    disable_icon_rotate=True,
                                    style_="font-size: 16px; font-weight: 500",
                                    class_="text-no-wrap",
                                    children=[header],
                                ),
                                content,
                            ],
                        )
                    ],
                ),
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
    def __init__(self, panel_id: "PlotID"=None, content_list=None, width="49.5%"):
        super().__init__(panel_id=panel_id, content_list=content_list, width=width)

        self.btn = v.Btn(
            class_="mx-1",
            icon=True,
            size="xx-small",
            elevation=0,
            children=[v.Icon(children="mdi-close-circle")],
        )

        self.settings_btn = vBtn(
            # class_="mx-1",
            style_="margin-left: auto;",
            icon=True,
            size="xx-small",
            elevation=0,
            children=[v.Icon(children="mdi-settings")],
            ref=panel_id
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
    def __init__(self, fig: mp.figure.Figure, axes: mp.axes.Axes, panel_id: Optional["PlotID"] = None, width="49%"):
        self.fig = fig
        self.axes = axes
        title = self.axes.title.get_text()
        self.axes.title.set_text("")

        self.output = ipywidgets.Output(layout=ipywidgets.Layout(object_fit="contain"))
        super().__init__(content_list=[self.output], panel_id=panel_id, width=width)
        self.title.children = [
            v.Html(
                style_="font-weight: normal; color: 0.2;", tag="div", children=[title]
            )
        ]


class PlotPanelCollection:
    def __init__(
        self,
        toggle_switches_by_plot_id: Dict["PlotID", v.Switch],
        ncols: int = 2,
        plot_choice_dialog: Callable = None,
        plot_settings_dialog: Callable = None
    ):
        self.ncols = ncols
        self.plot_choice_dialog = plot_choice_dialog
        self.plot_settings_dialog = plot_settings_dialog
        self.panel_by_btn: OrderedDict[v.Btn, ClosablePlotPanel] = collections.OrderedDict()
        self.panel_by_id: OrderedDict[str, ClosablePlotPanel] = collections.OrderedDict()
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
        # else:
        #     fig = mp.pyplot.figure()
        #     axes = fig.subplots()
        #     axes.plot(np.random.random(10), np.random.random(10))
        #     fig.set_figwidth(4)
        #     fig.set_figheight(4)
        #     explorer.new_plot_card("", fig, axes)

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

    # def new_panel(explorer, plot_id, content_list=None):
    #     """Adds a new card to the grid."""
    #     card = ClosablePanel(
    #         content_list=content_list, plot_id=plot_id, width=f"{100 / explorer.ncols - 1}%"
    #     )
    #     explorer.setup_card(card)

    def new_plot_panel(self, panel_id: "PlotID", fig, axes):
        """Adds a new plot card to the grid."""
        closable_panel = ClosablePlotPanel(
            fig, axes, panel_id=panel_id, width=f"{100 / self.ncols - 1}%"
        )
        self.setup_card(closable_panel)
        with closable_panel.output:
            mp.pyplot.show()

    def close_panel(self, close_btn: vBtn, *args, **kwargs):
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

    def settings_dialog(self, settings_btn: vBtn, *args, **kwargs):
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
                vBtn(
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
