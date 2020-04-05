# ui_base.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################


import functools
import numpy as np

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
import scqubits.utils.misc as utils


@utils.Required(ipywidgets=_HAS_IPYWIDGETS, IPython=_HAS_IPYTHON)
def create_widget(callback_func, init_params, image_filename=None):
    """
    Displays ipywidgets for initialization of a QuantumSystem object.

    Parameters
    ----------
    callback_func: function
        callback_function depends on all the parameters provided as keys (str) in the parameter_dict, and is called upon
        changes of values inside the widgets
    init_params: {str: value, str: value, ...}
        names and values of initialization parameters
    image_filename: str, optional
        file name for circuit image to be displayed alongside the qubit
    Returns
    -------

    """
    widgets = {}
    box_list = []
    for name, value in init_params.items():
        label = ipywidgets.Label(value=name)
        if isinstance(value, float):
            enter_widget = ipywidgets.FloatText
        else:
            enter_widget = ipywidgets.IntText

        widgets[name] = enter_widget(value=value, description='', disabled=False)
        box_list.append(ipywidgets.HBox([label, widgets[name]], layout=ipywidgets.Layout(justify_content='flex-end')))

    if image_filename:
        file = open(image_filename, "rb")
        image = file.read()
        image_widget = ipywidgets.Image(value=image, format='png')
        ui_widget = ipywidgets.HBox([ipywidgets.VBox(box_list), ipywidgets.VBox([image_widget])])
    else:
        ui_widget = ipywidgets.VBox(box_list)

    out = ipywidgets.interactive_output(callback_func, widgets)
    display(ui_widget, out)


class HilbertSpaceUi:
    def __init__(self):
        self.interact_current_index = 0
        self.interact_max_index = 0
        self.subsys_list = []
        self.interact_list = [self.empty_interaction_term()]

        label = ipywidgets.Label(value='HilbertSpace subsys_list')
        self.subsys_widget = ipywidgets.Textarea(placeholder='object name 1\nobject name 2\n...\n(existing  objects)')
        self.subsys_box = ipywidgets.VBox([label, self.subsys_widget])

        self.interact_new_button = ipywidgets.Button(description='New', layout=ipywidgets.Layout(width='80px'))
        self.interact_del_button = ipywidgets.Button(icon='remove', layout=ipywidgets.Layout(width='30px'))
        self.interact_right_button = ipywidgets.Button(icon='arrow-right', layout=ipywidgets.Layout(width='30px'))
        self.interact_left_button = ipywidgets.Button(icon='arrow-left', layout=ipywidgets.Layout(width='30px'))
        self.interact_buttons = ipywidgets.HBox([self.interact_new_button, self.interact_left_button,
                                                 self.interact_right_button, self.interact_del_button])

        self.op1_widget = ipywidgets.Text(description='op1', placeholder='e.g., <object>.n_operator()')
        self.op2_widget = ipywidgets.Text(description='op2', placeholder='e.g., <object>.creation_operator()')
        self.op1subsys_widget = ipywidgets.Text(description='subsys1')
        self.op2subsys_widget = ipywidgets.Text(description='subsys2')
        self.g_widget = ipywidgets.FloatText(description='g_strength')
        self.addhc_widget = ipywidgets.Dropdown(description='add_hc', options=['False', 'True'])

        self.interact_box = ipywidgets.VBox([
            self.interact_buttons,
            self.op1subsys_widget,
            self.op1_widget,
            self.op2subsys_widget,
            self.op2_widget,
            self.g_widget,
            self.addhc_widget
        ])

        self.interact_new_button.on_click(self.new_interaction_term)
        self.interact_del_button.on_click(self.del_interaction_term)
        self.interact_left_button.on_click(self.previous_interaction_term)
        self.interact_right_button.on_click(self.next_interaction_term)

        self.tab_nest = ipywidgets.widgets.Tab()
        self.tab_nest.children = [self.subsys_box, self.interact_box]
        self.tab_nest.set_title(0, 'Subsystems')
        self.tab_nest.set_title(1, 'Interactions')

        self.run_button = ipywidgets.Button(description='Finish')

        self.ui = ipywidgets.VBox([self.tab_nest, self.run_button])

    def new_interaction_term(self, *args):
        self.interact_max_index += 1
        self.interact_current_index = self.interact_max_index
        self.interact_list.append(self.empty_interaction_term())
        self.interact_index_change()

    def del_interaction_term(self, *args):
        if len(self.interact_list) == 1:
            self.interact_list[0] = self.empty_interaction_term()
        else:
            del self.interact_list[self.interact_current_index]
        if self.interact_max_index > 0:
            self.interact_max_index -= 1
        if self.interact_current_index > 0:
            self.interact_current_index -= 1
        self.interact_index_change()

    def next_interaction_term(self, *args):
        if self.interact_current_index < self.interact_max_index:
            self.interact_current_index += 1
            self.interact_index_change()

    def previous_interaction_term(self, *args):
        if self.interact_current_index > 0:
            self.interact_current_index -= 1
            self.interact_index_change()

    def interact_index_change(self, *args):
        interact_params = self.interact_list[self.interact_current_index]
        self.op1_widget.value = interact_params['op1']
        self.op1subsys_widget.value = interact_params['subsys1']
        self.op2_widget.value = interact_params['op2']
        self.op2subsys_widget.value = interact_params['subsys2']
        self.g_widget.value = interact_params['g_strength']
        self.addhc_widget.value = interact_params['add_hc']
        print(self.interact_current_index, self.interact_max_index)
        print(self.interact_list)

    @staticmethod
    def empty_interaction_term():
        return {
            'op1': '',
            'subsys1': '',
            'op2': '',
            'subsys2': '',
            'g_strength': 0.0,
            'add_hc': 'False'
        }

    def set_subsys_list(self, str_list):
        self.subsys_list = str_list.split('\n')
        while '' in self.subsys_list:
            self.subsys_list.remove('')

    def set_data(self, **kwargs):
        self.set_subsys_list(kwargs.pop('subsys_list'))
        self.set_interact_term(**kwargs)

    def set_interact_term(self, **kwargs):
        # interact_params = {key: kwargs[key] for key in ['op1', 'subsys1', 'op2', 'subsys2', 'g_strength', 'add_hc']}
        self.interact_list[self.interact_current_index] = kwargs

    def widgets_dict(self):
        return {
            'subsys_list': self.subsys_widget,
            'op1': self.op1_widget,
            'subsys1': self.op1subsys_widget,
            'op2': self.op2_widget,
            'subsys2': self.op2subsys_widget,
            'g_strength': self.g_widget,
            'add_hc': self.addhc_widget
        }

    def finish(self, callback_func, *args, **kwargs):
        hilbertspace_data = self.validated_data()
        if hilbertspace_data:
            subsystem_list, interaction_list = hilbertspace_data
            callback_func(subsystem_list, interaction_list)

    def set_err_output(self, out):
        self.error_output = out

    def validated_data(self):
        import importlib
        main = importlib.import_module('__main__')
        import scqubits.core.qubit_base as base

        self.error_output.clear_output()

        subsys_list = []
        for subsys_name in self.subsys_list:
            try:
                instance = getattr(main, subsys_name)
                subsys_list.append(instance)
            except AttributeError:
                with self.error_output:
                    print("Error: name '{}' is not defined.".format(subsys_name))
                return False
            if not isinstance(instance, scqubits.core.qubit_base.QuantumSystem):
                with self.error_output:
                    print("Type mismatch: object '{}' is not a qubit or oscillator.".format(subsys_name))
                return False

        interaction_list = []
        for interaction_term in self.interact_list:
            if interaction_term == self.empty_interaction_term():
                continue
            for param_name in ['subsys1', 'subsys2']:
                if interaction_term[param_name] not in self.subsys_list:
                    with self.error_output:
                        print("Error: subsystem operator '{}' is not consistent "
                              "with HilbertSpace subsys_list.".format(interaction_term[param_name]))
                    return False
            for param_name in ['op1', 'op2']:
                operator_str = interaction_term[param_name]
                try:
                    instance = eval(operator_str, main.__dict__)
                except (NameError, SyntaxError):
                    with self.error_output:
                        print("Error: {} '{}' is not defined or has a syntax error.".format(param_name, operator_str))
                    return False
                if not isinstance(instance, np.ndarray):
                    with self.error_output:
                        print("Type mismatch: '{}' is not a valid operator.".format(operator_str))
                    return False
            interaction_list.append(scqubits.InteractionTerm(g_strength=interaction_term['g_strength'],
                                                             op1=interaction_term['op1'],
                                                             subsys1=interaction_term['subsys1'],
                                                             op2=interaction_term['op2'],
                                                             subsys2=interaction_term['subsys2'],
                                                             add_hc=(interaction_term['add_hc'] == 'True')))
        return subsys_list, interaction_list


@utils.Required(ipywidgets=_HAS_IPYWIDGETS, IPython=_HAS_IPYTHON)
def create_hilbertspace_widget(callback_func):
    # ui_model = HilbertSpaceUiModel()
    ui_view = HilbertSpaceUi()

    out = ipywidgets.interactive_output(
        ui_view.set_data,
        ui_view.widgets_dict()
    )
    finish_func = functools.partial(ui_view.finish, callback_func)
    ui_view.run_button.on_click(finish_func)

    ui_view.set_err_output(out)
    display(ui_view.ui, out)
