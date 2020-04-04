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
    for name, value in init_params.items():
        if isinstance(value, float):
            widgets[name] = ipywidgets.FloatText(
                value=value,
                description=name,
                disabled=False,
                style={'description_width': 'initial'}
            )
        elif isinstance(value, int):
            widgets[name] = ipywidgets.IntText(
                value=value,
                min=0,
                description=name,
                disabled=False,
                style={'description_width': 'initial'}
            )

    ui_widget = ipywidgets.VBox(list(widgets.values()))
    if image_filename:
        file = open(image_filename, "rb")
        image = file.read()
        image_widget = ipywidgets.Image(
            value=image,
            format='png',
            width=600
        )
        ui_widget = ipywidgets.HBox([ui_widget, image_widget])

    out = ipywidgets.interactive_output(callback_func, widgets)
    display(ui_widget, out)


class HilbertSpaceUiModel:
    def __init__(self):
        self.subsystems = None
        self.interaction_terms = {}
        self.error_output = None
        self.message = None

    def set_params(self, **kwargs):
        self.subsystems = kwargs['subsystems'].split('\n')
        while '' in self.subsystems:
            self.subsystems.remove('')
        interact_params = {key: kwargs[key] for key in ['op1', 'subsys1', 'op2', 'subsys2', 'g_strength', 'add_hc']}
        self.interaction_terms[kwargs['interact_index']] = interact_params

    def finish(self, callback_func, *args, **kwargs):
        validated_parameters = self.validated_initdata()
        if validated_parameters:
            subsystem_list, interaction_list = validated_parameters
            return callback_func(subsystem_list, interaction_list=interaction_list)

    def set_output(self, out):
        self.error_output = out

    def validated_initdata(self):
        self.error_output.clear_output()

        import importlib
        main = importlib.import_module('__main__')

        subsys_list = []
        for subsys_name in self.subsystems:
            try:
                object = getattr(main, subsys_name)
                subsys_list.append(object)
            except NameError:
                with self.error_output:
                    print("NameError: name '{}' is not defined.".format(subsys_name))
                return False
            if not isinstance(object, scqubits.core.qubit_base.QuantumSystem):
                with self.error_output:
                    print("Type mismatch: object '{}' is not a qubit or oscillator.".format(subsys_name))
                return False

        interaction_list = []
        for key, params in self.interaction_terms.items():
            for param_name in ['subsys1', 'subsys2']:
                if params[param_name] not in self.subsystems:
                    with self.error_output:
                        print("Error: subsystem operator '{}' is not consistent "
                              "with HilbertSpace subsystems.".format(params[param_name]))
                    return False
            for param_name in ['op1', 'op2']:
                operator_str = params[param_name]

                try:
                    object = eval(operator_str, main.__dict__)
                except (NameError, SyntaxError):
                    with self.error_output:
                        print("Error: {} '{}' is not defined or has a syntax error.".format(param_name, operator_str))
                    return False
                if not isinstance(object, np.ndarray):
                    with self.error_output:
                        print("Type mismatch: '{}' is not a valid operator.".format(operator_str))
                    return False
            interaction_list.append(
                scqubits.InteractionTerm(g_strength=params['g_strength'],
                                         op1=params['op1'],
                                         subsys1=params['subsys1'],
                                         op2=params['op2'],
                                         subsys2=params['subsys2'],
                                         add_hc=params['add_hc'] == 'True')
                                    )
        return subsys_list, interaction_list

    def get_interact_params(self, index):
        if index in self.interaction_terms.keys():
            return self.interaction_terms[index]
        self.interaction_terms[index] = {
            'op1': '',
            'subsys1': '',
            'op2': '',
            'subsys2': '',
            'g_strength': 0.0,
            'add_hc': 'False'
        }
        return self.interaction_terms[index]


class HilbertSpaceUiView:
    def __init__(self, callback_params, callback_subsystems):
        self.callback_params = callback_params
        self.callback_subsystems = callback_subsystems

        self.label1 = ipywidgets.Label(value='HilbertSpace subsystems:')
        self.subsys_widget = ipywidgets.Textarea(
            value='',
            placeholder='object name 1\nobject name 2\n...',
            description='',
            disabled=False
        )
        self.box1_widget = ipywidgets.HBox([self.label1, self.subsys_widget])

        self.label2 = ipywidgets.Label(value='InteractionTerm #')
        self.interactindex_widget = ipywidgets.IntText(
            value=0,
            min=0,
            description='',
            layout=ipywidgets.Layout(width='80px'),
            disabled=False
        )
        self.box2_widget = ipywidgets.HBox([self.label2, self.interactindex_widget])

        self.label3 = ipywidgets.Label(value='op1', layout=ipywidgets.Layout(width='90px'))
        self.op1_widget = ipywidgets.Text(
            value='',
            placeholder='e.g., <Transmon object>.n_operator()',
            description='',
            disabled=False
        )
        self.box3_widget = ipywidgets.HBox([self.label3, self.op1_widget])

        self.label32 = ipywidgets.Label(value='subsys1', layout=ipywidgets.Layout(width='90px'))
        self.op1subsys_widget = ipywidgets.Dropdown(
            options=[''],
            value='',
            disabled=False
        )
        self.box32_widget = ipywidgets.HBox([self.label32, self.op1subsys_widget])

        self.label4 = ipywidgets.Label(value='op2', layout=ipywidgets.Layout(width='90px'))
        self.op2_widget = ipywidgets.Text(
            value='',
            placeholder='e.g., <Oscillator object>.creation_operator()',
            description='',
            disabled=False
        )
        self.box4_widget = ipywidgets.HBox([self.label4, self.op2_widget])

        self.label42 = ipywidgets.Label(value='subsys2', layout=ipywidgets.Layout(width='90px'))
        self.op2subsys_widget = ipywidgets.Dropdown(
            options=[''],
            value='',
            disabled=False
        )
        self.box42_widget = ipywidgets.HBox([self.label42, self.op2subsys_widget])

        self.label5 = ipywidgets.Label(value='g_strength', layout=ipywidgets.Layout(width='90px'))
        self.g_widget = ipywidgets.FloatText(
            value=0,
            description='',
            disabled=False
        )
        self.box5_widget = ipywidgets.HBox([self.label5, self.g_widget])

        self.label52 = ipywidgets.Label(value='add_hc', layout=ipywidgets.Layout(width='90px'))
        self.addhc_widget = ipywidgets.Dropdown(
            options=['True', 'False'],
            value='False',
            disabled=False
        )
        self.box52_widget = ipywidgets.HBox([self.label52, self.addhc_widget])

        self.box6_widget = ipywidgets.VBox([
            self.box2_widget,
            self.box3_widget,
            self.box32_widget,
            self.box4_widget,
            self.box42_widget,
            self.box5_widget,
            self.box52_widget
        ])

        self.tab_nest = ipywidgets.widgets.Tab()
        self.tab_nest.children = [self.box1_widget, self.box6_widget]
        self.tab_nest.set_title(0, 'Subsystems')
        self.tab_nest.set_title(1, 'Interactions')

        self.run_button = ipywidgets.Button(description='Finish')
        self.ui = ipywidgets.VBox([self.tab_nest, self.run_button])

    def interactindex_change(self, change):
        index = change['new']
        interact_params = self.callback_params(index)
        self.op1_widget.value = interact_params['op1']
        self.op1subsys_widget = interact_params['subsys1']
        self.op2_widget.value = interact_params['op2']
        self.op2subsys_widget = interact_params['subsys2']
        self.g_widget.value = interact_params['g_strength']
        self.addhc_widget.value = interact_params['add_hc']

    def subsystems_change(self, change):
        self.op1subsys_widget.options = self.callback_subsystems()
        self.op2subsys_widget.options = self.callback_subsystems()

    def widgets_dict(self):
        return {
            'subsystems': self.subsys_widget,
            'interact_index': self.interactindex_widget,
            'op1': self.op1_widget,
            'subsys1': self.op1subsys_widget,
            'op2': self.op2_widget,
            'subsys2': self.op2subsys_widget,
            'g_strength': self.g_widget,
            'add_hc': self.addhc_widget
        }


@utils.Required(ipywidgets=_HAS_IPYWIDGETS, IPython=_HAS_IPYTHON)
def create_hilbertspace_widget(callback_func):
    ui_model = HilbertSpaceUiModel()
    ui_view = HilbertSpaceUiView(ui_model.get_interact_params,
                                 lambda: ui_model.subsystems)

    finish_func = functools.partial(ui_model.finish, callback_func)

    ui_view.interactindex_widget.observe(ui_view.interactindex_change, names='value')

    out = ipywidgets.interactive_output(
        ui_model.set_params,
        ui_view.widgets_dict()
    )

    ui_view.run_button.on_click(finish_func)
    ui_view.subsys_widget.observe(ui_view.subsystems_change, names='value')

    ui_model.set_output(out)
    display(ui_view.ui, out)
