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

QuantumSys = Union[QubitBaseClass, Oscillator]


class HilbertSpaceUi:
    """Class for setup and display of the ipywidget used for creation of a
    HilbertSpace object."""

    @utils.Required(ipywidgets=_HAS_IPYWIDGETS)
    def __init__(self):
        """Set up all widget GUI elements and class attributes."""
        self.status_output = None
        self.subsys_candidates_dict = self.get_subsys_candidates()
        self.interactions_count = 0
        self.current_interaction_key = ""
        self.interactions_dict = {}

        # == subsystems panel =========================================================
        label = ipywidgets.Label(value="Select subsystems (Ctrl-Click)")
        self.subsys_refresh_button = ipywidgets.Button(
            icon="refresh", layout=ipywidgets.Layout(width="35px")
        )
        self.subsys_toprow = ipywidgets.HBox([label, self.subsys_refresh_button])

        self.subsys_widget = ipywidgets.SelectMultiple(
            options=list(self.subsys_candidates_dict.keys()),
            rows=10,
            description="",
            disabled=False,
            layout=ipywidgets.Layout(width="220px"),
        )
        self.subsys_box = ipywidgets.VBox([self.subsys_toprow, self.subsys_widget])

        # == InteractionTerms list panel ==============================================
        label = ipywidgets.Label(value="Interaction term(s)   ")
        self.interact_new_button = ipywidgets.Button(
            icon="plus", layout=ipywidgets.Layout(width="35px")
        )
        self.interact_del_button = ipywidgets.Button(
            icon="remove", layout=ipywidgets.Layout(width="35px")
        )
        self.interact_buttons = ipywidgets.HBox(
            [label, self.interact_new_button, self.interact_del_button]
        )
        self.interact_list_widget = ipywidgets.Select(
            options=[],
            rows=10,
            description="",
            disabled=False,
            layout=ipywidgets.Layout(width="200px"),
        )
        self.interact_list_box = ipywidgets.VBox(
            [self.interact_buttons, self.interact_list_widget]
        )

        # == Panel for specifying an InteractionTerm ==================================
        self.op1subsys_widget = ipywidgets.Dropdown(
            options=self.subsys_widget.value, description="subsys1", disabled=False
        )
        self.op2subsys_widget = ipywidgets.Dropdown(
            options=self.subsys_widget.value, description="subsys2", disabled=False
        )
        self.op1_ddown_widget = ipywidgets.Dropdown(
            options=self.possible_operators(self.op1subsys_widget.value),
            description="op1",
            disabled=False,
        )
        self.op2_ddown_widget = ipywidgets.Dropdown(
            options=self.possible_operators(self.op2subsys_widget.value),
            description="op2",
            disabled=False,
        )
        self.g_widget = ipywidgets.FloatText(description="g_strength")
        self.addhc_widget = ipywidgets.Dropdown(
            description="add_hc", options=["False", "True"]
        )

        self.interact_box1 = ipywidgets.VBox(
            [
                ipywidgets.Label(value="Specify interaction"),
                self.op1subsys_widget,
                self.op1_ddown_widget,
                self.op2subsys_widget,
                self.op2_ddown_widget,
                self.g_widget,
                self.addhc_widget,
            ]
        )

        self.string_expr_widget = ipywidgets.Text(
            description="expr", placeholder="e.g., EJ * cos(op1 - op2)"
        )
        self.interact_box2 = ipywidgets.VBox(
            [
                ipywidgets.Label(value="Specify interaction"),
                self.string_expr_widget,
                self.op1subsys_widget,
                self.op1_ddown_widget,
                self.op2subsys_widget,
                self.op2_ddown_widget,
                self.addhc_widget,
            ]
        )

        self.tabs_select_interact_type = ipywidgets.Tab(
            layout=ipywidgets.Layout(width="350px")
        )
        self.tabs_select_interact_type.children = [
            self.interact_box1,
            self.interact_box2,
        ]
        self.tabs_select_interact_type.set_title(0, "g * op1 * op2")
        self.tabs_select_interact_type.set_title(1, "Python expr")
        self.tabs_select_interact_type.layout.display = "none"

        # == Central run button, status output field ==================================
        self.run_button = ipywidgets.Button(
            description="Create HilbertSpace object",
            layout=ipywidgets.Layout(width="200px"),
        )
        self.status_output = ipywidgets.Output()

        # == Wrap everything into boxes ===============================================
        self.all_panels = ipywidgets.HBox(
            [self.subsys_box, self.interact_list_box, self.tabs_select_interact_type],
            layout=ipywidgets.Layout(grid_gap="50px"),
        )
        self.ui = ipywidgets.VBox(
            [self.all_panels, self.run_button, self.status_output]
        )

        # == Make GUI connections =====================================================
        self.connect_ui()

    @staticmethod
    def possible_operators(subsystem: str) -> List[str]:
        if subsystem is None:
            return []
        main = importlib.import_module("__main__")
        return [
            method_name + "()"
            for method_name in dir(main.__dict__[subsystem])  # type:ignore
            if "_operator" in method_name and method_name[0] != "_"
        ]

    def current_interaction_type(self) -> str:
        interaction_types = {0: "InteractionTerm", 1: "InteractionTermStr"}
        tab_index = self.tabs_select_interact_type.selected_index
        return interaction_types[tab_index]

    def connect_ui(self):
        def on_subsys_selected(change):
            self.op1subsys_widget.options = change["new"]
            self.op2subsys_widget.options = change["new"]

        def on_op_subsys1_selected(change):
            self.op1_ddown_widget.options = self.possible_operators(
                self.op1subsys_widget.value
            )

        def on_op_subsys2_selected(change):
            self.op2_ddown_widget.options = self.possible_operators(
                self.op2subsys_widget.value
            )

        def on_interact_list_changed(change):
            self.current_interaction_key = change["new"]
            self.current_interact_change()

        def refresh_subsys_list(*args):
            self.subsys_widget.options = list(self.get_subsys_candidates().keys())

        self.subsys_widget.observe(on_subsys_selected, "value")
        self.interact_list_widget.observe(on_interact_list_changed, "value")

        self.op1subsys_widget.observe(on_op_subsys1_selected, "value")
        self.op2subsys_widget.observe(on_op_subsys2_selected, "value")

        self.subsys_refresh_button.on_click(refresh_subsys_list)
        self.interact_new_button.on_click(self.new_interaction_term)
        self.interact_del_button.on_click(self.del_interaction_term)

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
        with self.status_output:
            print("HilbertSpace instance created.")
        callback_func(self.subsystem_list(), interaction_list)

    def set_data(self, **kwargs):
        self.set_interact_term(**kwargs)

    def set_interact_term(self, **kwargs):
        if self.current_interaction_key:
            self.interactions_dict[self.current_interaction_key] = kwargs

    def new_interaction_term(self, *args):
        self.interactions_count += 1
        self.current_interaction_key = "term {}".format(self.interactions_count)
        self.interactions_dict[
            self.current_interaction_key
        ] = self.empty_interaction_term()
        self.interact_list_widget.options = list(self.interactions_dict.keys())
        self.tabs_select_interact_type.layout.display = "flex"

    def del_interaction_term(self, *args):
        if len(list(self.interactions_dict.keys())) > 0:
            del self.interactions_dict[self.current_interaction_key]
        if self.interactions_dict:
            self.current_interaction_key = list(self.interactions_dict.keys())[0]
            self.interact_list_widget.options = list(self.interactions_dict.keys())
            self.current_interact_change()
        else:
            self.current_interaction_key = ""
            self.interact_list_widget.options = []
            self.tabs_select_interact_type.layout.display = "none"

    def current_interact_change(self, *args):
        if not self.current_interaction_key:
            return
        key = self.current_interaction_key
        interact_params = self.interactions_dict[key]
        self.op1subsys_widget.value = interact_params["subsys1"]
        self.op1_ddown_widget.value = interact_params["op1"]
        self.op2subsys_widget.value = interact_params["subsys2"]
        self.op2_ddown_widget.value = interact_params["op2"]
        self.g_widget.value = interact_params["g_strength"]
        self.addhc_widget.value = interact_params["add_hc"]
        self.string_expr_widget.value = interact_params["string_expr"]

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
            eval(subsys_name, main.__dict__) for subsys_name in self.subsys_widget.value
        ]

    def validated_interact_list(self):
        self.status_output.clear_output()

        main = importlib.import_module("__main__")
        subsysname_list = self.subsys_widget.value

        interaction_list = []
        for interaction_term in self.interactions_dict.values():
            for param_name in ["subsys1", "subsys2"]:
                if not interaction_term[param_name]:
                    with self.status_output:
                        print("Error: {} not specified.".format(param_name))
                    return False
                if interaction_term[param_name] not in subsysname_list:
                    with self.status_output:
                        print(
                            "Error: subsystem operator '{}' is not consistent "
                            "with HilbertSpace subsysname_list.".format(
                                interaction_term[param_name]
                            )
                        )
                    return False
            operator_str_list = [interaction_term["op1"], interaction_term["op2"]]
            for subsys_str, operator_str in zip(
                [interaction_term["subsys1"], interaction_term["subsys2"]],
                operator_str_list,
            ):
                try:
                    instance = eval(subsys_str + "." + operator_str, main.__dict__)
                except (AttributeError, SyntaxError, NameError):
                    with self.status_output:
                        print(
                            "Error: operator {} is not defined or has a "
                            "syntax error.".format(operator_str)
                        )
                    return False
                if not isinstance(instance, (np.ndarray, csc_matrix, Qobj)):
                    with self.status_output:
                        print(
                            "Type mismatch: '{}' is not a valid operator.".format(
                                operator_str
                            )
                        )
                    return False

            subsys1_index = subsysname_list.index(interaction_term["subsys1"])
            subsys2_index = subsysname_list.index(interaction_term["subsys2"])
            op1_str = interaction_term["subsys1"] + "." + operator_str_list[0]
            op2_str = interaction_term["subsys2"] + "." + operator_str_list[1]
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
                        expr=self.string_expr_widget.value,
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

    out = ipywidgets.interactive_output(ui_view.set_data, ui_view.widgets_dict())
    finish_func = functools.partial(ui_view.finish, callback_func)
    ui_view.run_button.on_click(finish_func)

    display(ui_view.ui, out)
