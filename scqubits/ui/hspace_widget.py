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

from typing import List, Union

import numpy as np

from qutip import Qobj
from scipy.sparse import csc_matrix

try:
    import ipywidgets
    import ipyvuetify as v
except ImportError:
    _HAS_IPYWIDGETS = False
else:
    _HAS_IPYWIDGETS = True

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
from scqubits.ui.custom_ipyvuetify import FloatTextField

QuantumSys = Union[QubitBaseClass, Oscillator]


class HilbertSpaceUi:
    """Class for setup and display of the widget used for creation of a
    HilbertSpace object."""

    @utils.Required(ipywidgets=_HAS_IPYWIDGETS)
    def __init__(self):
        """Set up all widget GUI elements and class attributes."""
        self.status_output = v.Container(children=[])
        self.subsys_candidates_dict = self.get_subsys_candidates()
        self.interactions_count = 0
        self.current_interaction_key = ""
        self.interactions_dict = {}

        # == subsystems panel =========================================================
        self.subsys_refresh_button = v.Btn(
            children=[v.Icon(children=["mdi-refresh"])],
            width=40,
            min_width=40,
            height=40,
            class_="ml-2",
        )

        self.subsys_widget = v.Select(
            v_model="",
            items=list(self.subsys_candidates_dict.keys()),
            menu_props={"closeOnContentClick": True},
            attach=True,
            chips=True,
            multiple=True,
            clear=True,
            outlined=True,
            filled=True,
            height=40,
            label="Select Subsystems",
            style_="width: 70%",
        )

        # == InteractionTerms list panel ==============================================
        self.interact_new_button = v.Btn(
            children=[v.Icon(children=["mdi-plus"])],
            width=40,
            min_width=40,
            height=40,
            class_="ml-2",
        )

        self.interact_list_widget = v.Select(
            v_model=None,
            items=[],
            outlined=True,
            filled=True,
            height=40,
            clearable=True,
            label="Edit interaction terms",
            class_="ml-0",
            style_="width: 70%",
        )

        self.interact_display = v.ChipGroup(
            v_model="None", children=[v.Text(children=["None"])]
        )

        # == Panel for specifying an InteractionTerm ==================================
        self.op1subsys_widget = v.Select(
            items=self.subsys_widget.v_model, label="subsys1", outlined=True, dense=True
        )
        self.op2subsys_widget = v.Select(
            items=self.subsys_widget.v_model, label="subsys2", outlined=True, dense=True
        )
        self.op1_ddown_widget = v.Select(
            items=self.possible_operators(self.op1subsys_widget.v_model),
            label="op1",
            outlined=True,
            dense=True,
        )
        self.op2_ddown_widget = v.Select(
            items=self.possible_operators(self.op2subsys_widget.v_model),
            label="op2",
            outlined=True,
            dense=True,
        )
        self.g_widget = FloatTextField(
            name="g_strength",
            label="g_strength",
            onchange=lambda *args, **kwargs: None,
            outlined=True,
            dense=True,
        )
        self.addhc_widget = v.Select(
            label="add_hc", items=["False", "True"], outlined=True, dense=True
        )

        self.interact_box1 = v.Col(
            children=[
                self.op1subsys_widget,
                self.op1_ddown_widget,
                self.op2subsys_widget,
                self.op2_ddown_widget,
                self.g_widget,
                self.addhc_widget,
            ]
        )

        self.string_expr_widget = v.TextField(
            label="expr", placeholder="e.g., EJ * cos(op1 - op2)"
        )
        self.interact_box2 = v.Col(
            children=[
                self.string_expr_widget,
                self.op1subsys_widget,
                self.op1_ddown_widget,
                self.op2subsys_widget,
                self.op2_ddown_widget,
                self.addhc_widget,
            ],
        )

        self.tabs_select_interact_type = v.Tabs(
            v_model="tab",
            align_with_title=True,
            grow=True,
            background_color="lightgrey",
            children=[
                v.Tab(children=["g * op1 * op2"]),
                v.Tab(children=["Python expr"]),
                v.TabItem(
                    key="g * op1 * op2",
                    children=[self.interact_box1],
                    style_="background-color: snow",
                ),
                v.TabItem(key="Python expr", children=[self.interact_box2]),
            ],
        )
        # == Central run button ==================================
        self.run_button = v.Btn(
            children=["Create HilbertSpace"], style_="align: bottom;"
        )

        # == Wrap everything into boxes ===============================================
        self.all_panels = v.Row(
            children=[
                v.Col(
                    children=[
                        v.Row(
                            children=[self.subsys_widget, self.subsys_refresh_button],
                        ),
                        v.Text(children=["Interaction terms:"], class_="mr-3"),
                        self.interact_display,
                        v.Spacer(style_="height: 400px"),
                        self.run_button,
                    ],
                    class_="col-6",
                ),
                v.Spacer(),
                v.Sheet(
                    children=[
                        v.Row(
                            children=[
                                self.interact_list_widget,
                                self.interact_new_button,
                                self.tabs_select_interact_type,
                            ],
                        )
                    ],
                    class_="mr-3 col-5",
                    style_="background-color: snow",
                ),
            ],
        )

        self.ui = v.Col(children=[self.all_panels, self.status_output])

        # == Make GUI connections =====================================================
        self.connect_ui()

    @staticmethod
    def possible_operators(subsystem: str) -> List[str]:
        if subsystem is None or subsystem == "!!disabled!!":
            return []
        main = importlib.import_module("__main__")
        return [
            method_name + "()"
            for method_name in dir(main.__dict__[subsystem])  # type:ignore
            if "_operator" in method_name
            and method_name[0] != "_"
            and "get_" not in method_name
        ]

    def current_interaction_type(self) -> str:
        interaction_types = {0: "InteractionTerm", 1: "InteractionTermStr"}
        tab_index = self.tabs_select_interact_type.v_model
        return interaction_types[tab_index]

    def update_interact_display(self):
        self.interact_display.children = [
            v.Chip(children=[term[0]], filling=False, outlined=True)
            for term in self.interact_list_widget.items
        ]
        self.interact_display.v_model = [
            term[0] for term in self.interact_list_widget.items
        ]

    def connect_ui(self):
        def on_subsys_selected(sender, event, value):
            self.op1subsys_widget.items = value
            self.op2subsys_widget.items = value

        def on_op_subsys1_selected(sender, event, value):
            self.op1subsys_widget.v_model = value
            self.op1_ddown_widget.items = self.possible_operators(value)
            self.update_interaction_data()

        def on_op_subsys2_selected(sender, event, value):
            self.op2subsys_widget.v_model = value
            self.op2_ddown_widget.items = self.possible_operators(value)
            self.update_interaction_data()

        def on_interact_list_changed(sender, event, value):
            self.current_interaction_key = tuple(value) if value else None
            self.current_interact_change()
            print(self.interact_list_widget.items)
            self.update_interact_display()

        def on_interact_list_click_clear(sender, event, value):
            self.del_interaction_term()
            self.update_interact_display()

        def refresh_subsys_list(*args):
            self.subsys_widget.items = list(self.get_subsys_candidates().keys())

        self.subsys_widget.on_event("change", on_subsys_selected)
        self.interact_list_widget.on_event("change", on_interact_list_changed)
        self.interact_list_widget.on_event("click:clear", on_interact_list_click_clear)

        self.op1subsys_widget.on_event("change", on_op_subsys1_selected)
        self.op2subsys_widget.on_event("change", on_op_subsys2_selected)
        self.op1_ddown_widget.on_event("change", self.update_interaction_data)
        self.op2_ddown_widget.on_event("change", self.update_interaction_data)
        self.addhc_widget.on_event("change", self.update_interaction_data)
        self.string_expr_widget.on_event("change", self.update_interaction_data)

        self.subsys_refresh_button.on_event("click", refresh_subsys_list)
        self.interact_new_button.on_event("click", self.new_interaction_term)

    def get_subsys_candidates(self):
        main = importlib.import_module("__main__")
        candidates_dict = {
            name: subsys
            for name, subsys in main.__dict__.items()
            if isinstance(subsys, QuantumSystem)
        }
        return candidates_dict

    def finish(self, callback_func, *args, **kwargs):
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
        callback_func(self.subsystem_list(), interaction_list)

    def set_data(self, **kwargs):
        self.set_interact_term(**kwargs)

    def set_interact_term(self, **kwargs):
        if self.current_interaction_key:
            self.interactions_dict[self.current_interaction_key] = kwargs

    def new_interaction_term(self, *args):
        self.interactions_count += 1
        self.current_interaction_key = (f"term {self.interactions_count}",)
        self.interactions_dict[
            self.current_interaction_key
        ] = self.empty_interaction_term()
        self.interact_list_widget.items = list(self.interactions_dict.keys())
        self.interact_list_widget.v_model = self.current_interaction_key
        self.update_interact_display()

    def del_interaction_term(self, *args):
        if len(list(self.interactions_dict.keys())) > 0:
            del self.interactions_dict[self.current_interaction_key]
        if self.interactions_dict:
            self.current_interaction_key = list(self.interactions_dict.keys())[0]
            self.interact_list_widget.items = list(self.interactions_dict.keys())
            self.current_interact_change()
        else:
            self.current_interaction_key = ""
            self.interact_list_widget.items = []
            # self.tabs_select_interact_type.layout.display = "none"

    def current_interact_change(self, *args):
        if (
            not self.current_interaction_key
            or self.current_interaction_key == "!!disabled!!"
        ):
            return
        key = self.current_interaction_key
        interact_params = self.interactions_dict[key]
        self.op1subsys_widget.v_model = interact_params["subsys1"]
        self.op1_ddown_widget.v_model = interact_params["op1"]
        self.op2subsys_widget.v_model_ = interact_params["subsys2"]
        self.op2_ddown_widget.v_model = interact_params["op2"]
        self.g_widget.v_model = interact_params["g_strength"]
        self.addhc_widget.v_model = interact_params["add_hc"]
        self.string_expr_widget.v_model = interact_params["string_expr"]

    def update_interaction_data(self, *args):
        if (
            not self.current_interaction_key
            or self.current_interaction_key == "!!disabled!!"
        ):
            print("no update")
            return
        print("update", self.op1subsys_widget)
        key = self.current_interaction_key
        self.interactions_dict[key]["subsys1"] = self.op1subsys_widget.v_model
        self.interactions_dict[key]["subsys2"] = self.op2subsys_widget.v_model
        self.interactions_dict[key]["op1"] = self.op1_ddown_widget.v_model
        self.interactions_dict[key]["op2"] = self.op2_ddown_widget.v_model
        self.interactions_dict[key]["g_strength"] = self.g_widget.v_model
        self.interactions_dict[key]["add_hc"] = self.addhc_widget.v_model
        self.interactions_dict[key]["string_expr"] = self.string_expr_widget.v_model
        print(self.interact_list_widget)

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

    def widgets_dict(self):
        return {
            "subsys_list": self.subsys_widget,
            "subsys1": self.op1subsys_widget,
            "op1": self.op1_ddown_widget,
            "subsys2": self.op2subsys_widget,
            "op2": self.op2_ddown_widget,
            "g_strength": self.g_widget,
            "add_hc": self.addhc_widget,
            "string_expr": self.string_expr_widget,
        }

    def subsystem_list(self) -> "List[QuantumSys]":
        main = importlib.import_module("__main__")
        return [
            eval(subsys_name, main.__dict__) for subsys_name in self.subsys_widget.items
        ]

    def validated_interact_list(self):
        self.status_output.children = []

        main = importlib.import_module("__main__")
        subsysname_list = self.subsys_widget.v_model

        interaction_list = []
        for interaction_term in self.interactions_dict.values():
            print(interaction_term)
            for param_name in ["subsys1", "subsys2"]:
                if not interaction_term[param_name]:
                    self.status_output.children = [
                        v.Alert(
                            children=[f"Error: {param_name} not specified."],
                            text=True,
                            dense=True,
                            type="error",
                            dismissible=True,
                        )
                    ]
                    return False
                if interaction_term[param_name] not in subsysname_list:
                    self.status_output.children = [
                        v.Alert(
                            children=[
                                f"Error: subsystem operator '{interaction_term[param_name]}' is not consistent with HilbertSpace subsysname_list."
                            ],
                            text=True,
                            dense=True,
                            type="error",
                            dismissible=True,
                        )
                    ]
                    return False
            operator_str_list = [interaction_term["op1"], interaction_term["op2"]]
            for subsys_str, operator_str in zip(
                [interaction_term["subsys1"], interaction_term["subsys2"]],
                operator_str_list,
            ):
                try:
                    instance = eval(f"{subsys_str}.{operator_str}", main.__dict__)
                except (AttributeError, SyntaxError, NameError):
                    self.status_output.children = [
                        v.Alert(
                            children=[
                                f"Error: operator {operator_str} is not defined or has a syntax error."
                            ],
                            text=True,
                            dense=True,
                            type="error",
                            dismissible=True,
                        )
                    ]
                    return False
                if not isinstance(instance, (np.ndarray, csc_matrix, Qobj)):
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

            subsys1_index = subsysname_list.index(interaction_term["subsys1"])
            subsys2_index = subsysname_list.index(interaction_term["subsys2"])
            op1_str = f"""{interaction_term["subsys1"]}.{operator_str_list[0]}"""
            op2_str = f"""{interaction_term["subsys2"]}.{operator_str_list[1]}"""
            op1 = eval(op1_str, main.__dict__)
            op2 = eval(op2_str, main.__dict__)

            if self.current_interaction_type() == "InteractionTerm":
                operator_list = [(subsys1_index, op1), (subsys2_index, op2)]
                interaction_list.append(
                    scqubits.InteractionTerm(
                        g_strength=interaction_term["g_strength"],
                        operator_list=operator_list,
                        add_hc=(interaction_term["add_hc"] == "True"),
                    )
                )
            else:  # current interaction type is 'InteractionTermStr'
                interaction_list.append(
                    scqubits.InteractionTermStr(
                        expr=self.string_expr_widget.v_model,
                        operator_list=[
                            (subsys1_index, "op1", op1),
                            (subsys2_index, "op2", op2),
                        ],
                        add_hc=(interaction_term["add_hc"] == "True"),
                    )
                )
        return interaction_list


@utils.Required(ipywidgets=_HAS_IPYWIDGETS, IPython=_HAS_IPYTHON)
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
