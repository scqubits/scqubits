# hspace_widget.py
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

import functools
import importlib

from typing import Dict, List, Union
from typing_extensions import Literal  # for Python 3.7 compatibility

import numpy as np

from qutip import Qobj
from scipy.sparse import csc_matrix

from scqubits.ui.gui_defaults import NAV_COLOR

try:
    import ipyvuetify as v
    import ipywidgets

    from scqubits.ui.gui_custom_widgets import ClickChip, ValidatedNumberField
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

import scqubits

from scqubits.core.oscillator import Oscillator
from scqubits.core.qubit_base import QuantumSystem, QubitBaseClass
from scqubits.utils import misc as utils

QuantumSys = Union[QubitBaseClass, Oscillator]


class HilbertSpaceUi:
    """Class for setup and display of the widget used for creation of a
    HilbertSpace object."""

    @utils.Required(ipyvuetify=_HAS_IPYVUETIFY)
    def __init__(self):
        """Set up all widget GUI elements and class attributes."""
        self.status_output = v.Container(children=[], id="status_output")
        self.subsys_candidates_dict = self.get_subsys_candidates()
        self.current_interaction_idx = None
        # self.interaction_terms_dict = collections.OrderedDict()

        self.op1subsys_widget: List[v.Select] = []
        self.op2subsys_widget: List[v.Select] = []
        self.op1_ddown_widget: List[v.Select] = []
        self.op2_ddown_widget: List[v.Select] = []
        self.g_widget: List[ValidatedNumberField] = []
        self.addhc_widget: List[v.Select] = []
        self.string_expr_widget: List[v.TextField] = []
        self.interact_box1: List[v.Container] = []
        self.interact_box2: List[v.Container] = []

        # == subsystems panel =========================================================
        self.subsys_refresh_button = v.Btn(
            children=[v.Icon(children=["mdi-refresh"])],
            width=40,
            min_width=40,
            height=40,
            class_="ml-2 mt-2",
        )

        self.subsys_widget = v.Select(
            class_="px-2",
            v_model=[],
            items=list(self.subsys_candidates_dict.keys()),
            menu_props={"closeOnContentClick": True},
            attach=True,
            chips=True,
            multiple=True,
            clear=True,
            outlined=True,
            filled=True,
            height=40,
            label="Select Subsystem(s)",
            # style_="width: 70%",
        )

        # == InteractionTerms list panel ==============================================
        self.interact_new_button = v.Btn(
            children=[v.Icon(children=["mdi-plus"])],
            width=40,
            min_width=40,
            height=40,
            class_="ml-2 align-self-center",
        )

        self.interact_chipgroup = v.ChipGroup(
            v_model=None,
            mandatory=True,
            children=[],
            id="interact_display",
            class_="ml-2 align-self-center",
            color="primary",
            active_class="text-primary",
        )

        # == Panel for specifying an InteractionTerm ==================================

        self.tabs_select_interact_type = []

        # == Central run button ==================================
        self.run_button = v.Btn(
            children=["Create HilbertSpace"],
            class_="m-2",
            style_="align: bottom;",
            disabled=True,
        )

        self.edit_interaction_card = v.Card(
            class_="d-flex flex-column align-self-center mb-2",
            width=450,
            children=[
                v.CardTitle(children=["Edit Interaction Terms"]),
                v.Container(
                    class_="d-flex flex-row",
                    children=[
                        self.interact_new_button,
                        self.interact_chipgroup,
                    ],
                ),
                v.Container(),  # SLOT 2 -- for self.tabs_select_interact_type[idx],
            ],
            id="edit_interaction_card",
            style_="display: none !important",
        )

        # == Wrap everything into boxes ===============================================
        self.all_panels = v.Card(
            class_="d-flex flex-column",
            children=[
                v.CardTitle(children=["Create Hilbert Space"]),
                v.Container(
                    class_="d-flex flex-row align-self-left my-0 py-0",
                    children=[self.subsys_widget, self.subsys_refresh_button],
                    style_="width: 50%",
                ),
                self.edit_interaction_card,
                self.run_button,
            ],
        )

        self.ui = v.Container(
            class_="d-flex flex-column", children=[self.all_panels, self.status_output]
        )

        # == Make GUI connections =====================================================
        self.connect_ui()

    @staticmethod
    def possible_operators(subsystem: str) -> List[str]:
        if subsystem is None:
            return []
        main = importlib.import_module("__main__")
        return [
            method_name
            for method_name in dir(main.__dict__[subsystem])  # type:ignore
            if "_operator" in method_name
            and method_name[0] != "_"
            and "get_" not in method_name
        ]

    def new_interact_entry_widget(self):
        self.op1subsys_widget.append(
            v.Select(
                v_model=None,
                items=self.subsys_widget.v_model,
                label="subsys1",
                outlined=True,
                dense=True,
            )
        )
        self.op2subsys_widget.append(
            v.Select(
                v_model=None,
                items=self.subsys_widget.v_model,
                label="subsys2",
                outlined=True,
                dense=True,
            )
        )
        self.op1_ddown_widget.append(
            v.Select(
                v_model=None,
                items=[],
                label="op1",
                outlined=True,
                dense=True,
            )
        )
        self.op2_ddown_widget.append(
            v.Select(
                v_model=None,
                items=[],
                label="op2",
                outlined=True,
                dense=True,
            )
        )
        self.g_widget.append(
            ValidatedNumberField(
                v_model="0",
                num_type=float,
                label="g_strength",
                style_="",
                outlined=True,
                dense=True,
                filled=False,
            )
        )
        self.addhc_widget.append(
            v.Select(
                v_model="False",
                label="add_hc",
                items=["False", "True"],
                outlined=True,
                dense=True,
            )
        )

        self.interact_box1.append(
            v.Container(
                class_="d-flex flex-column",
                children=[
                    self.op1subsys_widget[-1],
                    self.op1_ddown_widget[-1],
                    self.op2subsys_widget[-1],
                    self.op2_ddown_widget[-1],
                    self.g_widget[-1],
                    self.addhc_widget[-1],
                ],
                id="interact_box1",
            )
        )

        self.string_expr_widget.append(
            v.TextField(
                v_model="", label="expr", placeholder="e.g., EJ * cos(op1 - op2)"
            )
        )

        self.interact_box2.append(
            v.Container(
                class_="d-flex flex-column",
                children=[
                    self.string_expr_widget[-1],
                    self.op1subsys_widget[-1],
                    self.op1_ddown_widget[-1],
                    self.op2subsys_widget[-1],
                    self.op2_ddown_widget[-1],
                    self.addhc_widget[-1],
                ],
                id="interact_box2",
            )
        )

        self.tabs_select_interact_type.append(
            v.Tabs(
                v_model="tab",
                align_with_title=True,
                grow=True,
                background_color=NAV_COLOR,
                children=[
                    v.Tab(children=["g * op1 * op2"]),
                    v.Tab(children=["Python expr"]),
                    v.TabItem(
                        key="g * op1 * op2",
                        children=[self.interact_box1[-1]],
                        style_="background-color: " + NAV_COLOR,
                    ),
                    v.TabItem(
                        key="Python expr",
                        children=[self.interact_box2[-1]],
                        style_="background-color: " + NAV_COLOR,
                    ),
                ],
            )
        )

        def on_op_subsys1_selected(*args):
            value = self.op1subsys_widget[self.current_interaction_idx].v_model
            self.op1_ddown_widget[
                self.current_interaction_idx
            ].items = self.possible_operators(value)

        def on_op_subsys2_selected(*args):
            value = self.op2subsys_widget[self.current_interaction_idx].v_model
            self.op2_ddown_widget[
                self.current_interaction_idx
            ].items = self.possible_operators(value)

        self.op1subsys_widget[-1].observe(on_op_subsys1_selected, names="v_model")
        self.op2subsys_widget[-1].observe(on_op_subsys2_selected, names="v_model")

    def get_interaction_type(self, idx) -> str:
        interaction_types = {0: "InteractionTerm", 1: "InteractionTermStr"}
        tab_index = self.tabs_select_interact_type[idx].v_model
        return interaction_types[tab_index]

    def on_interact_chipgroup_change(self, *args, **kwargs):
        active_term = self.interact_chipgroup.v_model
        if active_term == self.current_interaction_idx:
            return
        if active_term is not None:
            self.current_interaction_idx = int(
                self.interact_chipgroup.children[active_term].children[0][5:]
            )
        else:
            self.current_interaction_idx = None
        self.retrieve_and_display_interact_data()

    def update_chipgroup_display(self):
        self.interact_chipgroup.children = [
            ClickChip(
                children=[f"term {idx}"],
                class_="align-self-center",
                close=True,
                click_close=self.del_interaction_term,
            )
            for idx, _ in enumerate(self.interact_box1)
        ]

        if self.current_interaction_idx is not None:
            self.interact_chipgroup.v_model = self.current_interaction_idx
        else:
            self.interact_chipgroup.v_model = None

    def connect_ui(self):
        def on_subsys_selected(*args):
            for idx, _ in enumerate(self.interact_box1):
                self.op1subsys_widget[idx].items = self.subsys_widget.v_model
                self.op2subsys_widget[idx].items = self.subsys_widget.v_model
            if self.subsys_widget.v_model:
                self.edit_interaction_card.style_ = ""
                self.run_button.disabled = False
            else:
                self.edit_interaction_card.style_ = "display: none !important;"
                self.run_button.disabled = True

        def refresh_subsys_list(*args):
            self.subsys_widget.items = list(self.get_subsys_candidates().keys())

        self.subsys_widget.observe(on_subsys_selected, names="v_model")

        self.interact_chipgroup.observe(
            self.on_interact_chipgroup_change, names="v_model"
        )

        self.subsys_refresh_button.on_event("click", refresh_subsys_list)
        self.interact_new_button.on_event("click", self.new_interaction_term)

    def get_subsys_candidates(self):
        main = importlib.import_module("__main__")
        candidates_dict = {
            name: subsys
            for name, subsys in main.__dict__.items()
            if isinstance(subsys, QuantumSystem) and name[0] != "_"
        }
        return candidates_dict

    def finish(self, callback_func, *args, **kwargs):
        main = importlib.import_module("__main__")
        interaction_list = self.validated_interact_list()
        if interaction_list is False:
            return None
        self.status_output.children = [
            v.Alert(
                children=["HilbertSpace instance created."],
                color="blue",
                text=True,
                dense=True,
                type="info",
                dismissible=True,
            )
        ]
        subsys_list = [
            main.__dict__[subsys_name] for subsys_name in self.subsys_widget.v_model
        ]
        callback_func(subsys_list, interaction_list)

    def new_interaction_term(self, *args):
        self.new_interact_entry_widget()
        self.update_chipgroup_display()
        self.retrieve_and_display_interact_data()
        self.interact_chipgroup.v_model = self.current_interaction_idx

    def del_interaction_term(self, *args):
        idx = self.current_interaction_idx
        if self.interact_box1:
            del self.op1subsys_widget[idx]
            del self.op2subsys_widget[idx]
            del self.op1_ddown_widget[idx]
            del self.op2_ddown_widget[idx]
            del self.g_widget[idx]
            del self.addhc_widget[idx]
            del self.string_expr_widget[idx]
            del self.interact_box1[idx]
            del self.interact_box2[idx]
            del self.tabs_select_interact_type[idx]
        if self.interact_box1:
            self.current_interaction_idx = 0
            self.retrieve_and_display_interact_data()
            self.update_chipgroup_display()
        else:
            self.current_interaction_idx = None
            self.edit_interaction_card.children[2].style_ = "display: none !important"
            self.update_chipgroup_display()

    def retrieve_and_display_interact_data(self, *args):
        if self.current_interaction_idx is None:
            self.edit_interaction_card.children[2].style_ = "display: none !import"
            return
        idx = self.current_interaction_idx
        self.edit_interaction_card.children[2].children = [
            self.tabs_select_interact_type[idx]
        ]
        self.edit_interaction_card.children[2].style_ = ""

    @staticmethod
    def empty_interaction_term():
        return {
            "subsys1": None,
            "op1": None,
            "subsys2": None,
            "op2": None,
            "g_strength": 0.0,
            "add_hc": "False",
            "string_expr": "",
        }

    def widgets_dict(self, idx: int) -> Dict[str, "v.VuetifyWidget"]:
        return {
            "subsys_list": self.subsys_widget,
            "subsys1": self.op1subsys_widget[idx],
            "op1": self.op1_ddown_widget[idx],
            "subsys2": self.op2subsys_widget[idx],
            "op2": self.op2_ddown_widget[idx],
            "g_strength": self.g_widget[idx],
            "add_hc": self.addhc_widget[idx],
            "string_expr": self.string_expr_widget[idx],
        }

    def subsystem_list(self) -> "List[QuantumSys]":
        main = importlib.import_module("__main__")
        return [
            eval(subsys_name, main.__dict__) for subsys_name in self.subsys_widget.items
        ]

    def validated_interact_list(
        self,
    ) -> Union[Literal[False], list]:
        self.status_output.children = []

        main = importlib.import_module("__main__")
        subsysname_list = self.subsys_widget.v_model

        interaction_list = []
        for idx, _ in enumerate(self.interact_box1):
            interaction_term = self.widgets_dict(idx)
            for subsys in ["subsys1", "subsys2"]:
                if not interaction_term[subsys]:
                    self.status_output.children = [
                        v.Alert(
                            children=[f"Error: {subsys} not specified."],
                            text=True,
                            dense=True,
                            type="error",
                            dismissible=True,
                        )
                    ]
                    return False
                if interaction_term[subsys].v_model not in subsysname_list:
                    self.status_output.children = [
                        v.Alert(
                            children=[
                                f"Error: subsystem operator '{interaction_term[subsys].v_model}' is not consistent with"
                                f" HilbertSpace subsysname_list."
                            ],
                            text=True,
                            dense=True,
                            type="error",
                            dismissible=True,
                        )
                    ]
                    return False
            operator_str_list = [
                interaction_term["op1"].v_model,
                interaction_term["op2"].v_model,
            ]
            for subsys_str, operator_str in zip(
                [
                    interaction_term["subsys1"].v_model,
                    interaction_term["subsys2"].v_model,
                ],
                operator_str_list,
            ):
                try:
                    instance = eval(f"{subsys_str}.{operator_str}", main.__dict__)
                except (AttributeError, SyntaxError, NameError):
                    self.status_output.children = [
                        v.Alert(
                            children=[
                                f"Error: operator {subsys_str}.{operator_str} is not defined or has a syntax error."
                            ],
                            text=True,
                            dense=True,
                            type="error",
                            dismissible=True,
                        )
                    ]
                    return False
                if not isinstance(
                    instance, (np.ndarray, csc_matrix, Qobj)
                ) and not callable(instance):
                    self.status_output.children = [
                        v.Alert(
                            children=[
                                f"Error (type mismatch): '{operator_str}' is not a valid operator."
                            ],
                            text=True,
                            dense=True,
                            type="error",
                            dismissible=True,
                        )
                    ]
                    return False

            subsys1_index = subsysname_list.index(interaction_term["subsys1"].v_model)
            subsys2_index = subsysname_list.index(interaction_term["subsys2"].v_model)
            op1_str = (
                f"""{interaction_term["subsys1"].v_model}.{operator_str_list[0]}"""
            )
            op2_str = (
                f"""{interaction_term["subsys2"].v_model}.{operator_str_list[1]}"""
            )
            op1 = eval(op1_str, main.__dict__)
            op2 = eval(op2_str, main.__dict__)

            if self.get_interaction_type(idx) == "InteractionTerm":
                operator_list = [(subsys1_index, op1), (subsys2_index, op2)]
                interaction_list.append(
                    scqubits.InteractionTerm(
                        g_strength=interaction_term["g_strength"].num_value,
                        operator_list=operator_list,
                        add_hc=(interaction_term["add_hc"].v_model == "True"),
                    )
                )
            else:  # current interaction type is 'InteractionTermStr'
                interaction_list.append(
                    scqubits.InteractionTermStr(
                        expr=interaction_term["string_expr"].v_model,
                        operator_list=[
                            (subsys1_index, "op1", op1),
                            (subsys2_index, "op2", op2),
                        ],
                        add_hc=(interaction_term["add_hc"].v_model == "True"),
                    )
                )
        return interaction_list


@utils.Required(ipyvuetify=_HAS_IPYVUETIFY, IPython=_HAS_IPYTHON)
def create_hilbertspace_widget(callback_func):
    """
    Display ipywidgets interface for creating a HilbertSpace object. Typically,
    this function will be called by `HilbertSpace.create()``.

    Parameters
    ----------
    callback_func: function
        Function that receives the subsystem and interaction data from the widget.
        Typically, this is ``HilbertSpace.__init__()``
    """
    ui_view = HilbertSpaceUi()

    finish_func = functools.partial(ui_view.finish, callback_func)
    ui_view.run_button.on_event("click", finish_func)
    display(ui_view.ui)
