# gui.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################


import numpy as np
import scqubits as scq
from scqubits.core.qubit_base import QubitBaseClass
import inspect
from ipywidgets import widgets, Layout, HBox, VBox, Label, IntSlider, Text, AppLayout
from ipywidgets import interactive
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, Union, Tuple, List


class GUI:
    def __repr__(self):
        return ''

    def __init__(self):
        scq.settings.PROGRESSBAR_DISABLED = True
        global_defaults = {
            'mode_wavefunc': 'real',
            'mode_matrixelem': 'abs',
            'ng': {'min': 0, 'max': 1},
            'flux': {'min': 0, 'max': 1},
            'EJ': {'min': 1e-10, 'max': 70},
            'EC': {'min': 1e-10, 'max': 50},
            'int': {'min': 1, 'max': 30},
            'float': {'min': 0, 'max': 30}
        }
        transmon_defaults = {**global_defaults,
            'scan_param': 'ng',
            'operator': 'n_operator',
            'ncut': {'min': 10, 'max': 50},
            'scale': 1
        }
        tunabletransmon_defaults = {**global_defaults,
            'scan_param': 'flux',
            'operator': 'n_operator',
            'EJmax': global_defaults['EJ'],
            'd': {'min': 0, 'max': 1},
            'ncut': {'min': 10, 'max': 50},
            'scale': 1
        }
        fluxonium_defaults = {**global_defaults,
            'scan_param': 'flux',
            'operator': 'n_operator',
            'EL': {'min': 1e-10, 'max': 30},
            'cutoff': {'min': 10, 'max': 120},
            'scale': 1
        }
        fluxqubit_defaults = {**global_defaults,
            'scan_param': 'flux',
            'operator': 'n_1_operator',
            'ncut': {'min': 5, 'max': 30},
            'EJ1': global_defaults['EJ'],
            'EJ2': global_defaults['EJ'],
            'EJ3': global_defaults['EJ'],	
            'ECJ1': global_defaults['EC'],
            'ECJ2': global_defaults['EC'],
            'ECJ3': global_defaults['EC'],
            'ECg1': global_defaults['EC'],
            'ECg2': global_defaults['EC'],
            'ng1': global_defaults['ng'],
            'ng2': global_defaults['ng'],
            'scale': None
        }
        zeropi_defaults = {**global_defaults,   
            'scan_param': 'flux',
            'operator': 'n_theta_operator',
            'ncut': {'min': 5, 'max': 50},
            'EL': {'min': 1e-10, 'max': 30},
            'ECJ': global_defaults['EC'],
            'dEJ': {'min': 0, 'max': 1},
            'dCJ': {'min': 0, 'max': 1},
            'scale': None
        }
        fullzeropi_defaults = {**global_defaults,   
            'scan_param': 'flux',
            'operator': 'n_theta_operator',
            'ncut': {'min': 5, 'max': 50},
            'EL': {'min': 1e-10, 'max': 30},
            'ECJ': global_defaults['EC'],
            'dEJ': {'min': 0, 'max': 1},
            'dCJ': {'min': 0, 'max': 1},
            'dEL': {'min': 0, 'max': 1},
            'dC': {'min': 0, 'max': 1},
            'zeropi_cutoff': {'min': 10, 'max': 50},
            'zeta_cutoff': {'min': 10, 'max': 50},
            'scale': None
        }
        self.qubit_defaults = {
            'Transmon': transmon_defaults, 
            'TunableTransmon': tunabletransmon_defaults, 
            'Fluxonium': fluxonium_defaults, 
            'FluxQubit': fluxqubit_defaults,
            'ZeroPi': zeropi_defaults,
            'FullZeroPi': fullzeropi_defaults
        }
        self.grid_defaults = {
            'grid_min_val': -6*np.pi,
            'grid_max_val': 6*np.pi,
            'grid_pt_count': 50
        }
        self.plot_choices = { 
            'Energy spectrum': 'plot_evals_vs_paramvals', 
            'Wavefunctions': 'plot_wavefunction', 
            'Matrix element scan': 'plot_matelem_vs_paramvals', 
            'Matrix elements': 'plot_matrixelements'
        }
        self.supported_qubits = [ 'Transmon', 'TunableTransmon', 'Fluxonium', 'FluxQubit', 'ZeroPi', 'FullZeroPi' ]
        self.active_defaults: Dict[ str, Union[ str, Dict[ str, Union[ int, float ] ] ] ] = {}
        self.fig: Figure
        self.qubit_base_params: Dict[ str, Union[ int, float ] ] = {}
        self.qubit_scan_params: Dict[ str, Union[ int, float ] ] = {}
        self.qubit_plot_options_widgets: Dict[ widgets ] = {}
        self.qubit_and_plot_choice_widgets: Dict[ widgets ] = {}
        self.qubit_params_widgets: Dict[ widgets ] = {}
        self.active_qubit: QubitBaseClass
    
        self.set_qubit('Transmon')
        self.create_qubit_and_plot_choice_widgets()
        
        qubit_and_plot_choice_display, plot_display = self.create_GUI()
        display(qubit_and_plot_choice_display, plot_display)
        
    #Initialization Methods -------------------------------------------------------------------------------------------------
    def initialize_qubit(self, qubit_name: str) -> None:
        """Initializes self.active_qubit to the user's choice
        using the default parameters of chosen qubit.

        Parameters
        ----------
        qubit_name:

        """
        QubitClass = getattr(scq, qubit_name)
        init_params = QubitClass.default_params()

        if qubit_name == 'ZeroPi' or qubit_name == 'FullZeroPi':
            init_params['grid'] = scq.Grid1d(
                                            min_val = self.grid_defaults['grid_min_val'], 
                                            max_val = self.grid_defaults['grid_max_val'], 
                                            pt_count = self.grid_defaults['grid_pt_count'])
          
        self.active_qubit = QubitClass(**init_params)

    def set_qubit(self, qubit_name: str) -> None:
        """Initializes the qubit and creates all the necessary widgets
        for the GUI.

        Parameters
        ----------
        qubit_name:
            
        """
        self.active_defaults = self.qubit_defaults[qubit_name]
        self.initialize_qubit(qubit_name)
        self.create_params_dict()
        self.create_plot_settings_widgets()
        self.create_qubit_params_widgets()

    def get_operators(self) -> List[ str ]:
        """Return a list of operators.
        Note that this list omits any operators that start with "_".

        Returns
        -------
        List[ str ]

        """
        operator_list = []
        for name, val in inspect.getmembers(self.active_qubit):
            if "operator" in name and name[0] != "_":
                operator_list.append(name)
        return operator_list

    def get_plot_choices(self) -> List[ str ]:
        """Return a list of plot choices.
        
        Returns
        -------
        List[ str ]

        """
        plot_choices_list = []
        for plot_name, plot_method in self.plot_choices.items():
            if hasattr(self.active_qubit, plot_method):
                plot_choices_list.append(plot_name)
        return plot_choices_list

    #Widget Methods -------------------------------------------------------------------------------------------------
    def scan_dropdown_eventhandler(self, change):
        self.qubit_plot_options_widgets['scan_range_slider'].description = '{} range'.format(change.new)

        self.qubit_plot_options_widgets['scan_range_slider'].min = self.active_defaults[change.new]['min']
        self.qubit_plot_options_widgets['scan_range_slider'].max = self.active_defaults[change.new]['max']
        self.qubit_plot_options_widgets['scan_range_slider'].value = [ self.active_defaults[change.new]['min'], self.active_defaults[change.new]['max'] ]
        self.qubit_params_widgets[change.old].disabled = False
        self.qubit_params_widgets[change.new].disabled = True
            
    def save_button_clicked_action(self, *args):
        self.fig.savefig(self.qubit_plot_options_widgets['filename_text'].value)

    #Plot Methods -------------------------------------------------------------------------------------------------
    def evals_vs_paramvals_interactive(self, 
        scan_value: str, 
        scan_range: Tuple[ float, float ], 
        eigenvalue_amount_value: int, 
        subtract_ground_tf: bool, 
        **params: Dict[ str, Union[ float, int ] ]) -> None:
        """This is the method associated with qubit_plot_interactive that allows for us to interact with plot_evals_vs_paramvals().

        Parameters
        ----------
        scan_value:
            Current value of the scan parameter dropdown.

        scan_range:
            Sets the interval [ min, max ] through
            which plot_evals_vs_paramvals() will plot over.

        eigenvalue_amount_value:
            The number of eigenvalues that will be plotted.

        subtract_ground_tf:
            Determines whether we subtract away the ground energy or not.
            Initially set to False.

        **params:
            Dictionary of current qubit parameter values (taken from the sliders)
        """
        scan_min, scan_max = scan_range
        self.active_qubit.set_params(**params)
        np_list = np.linspace(scan_min, scan_max, 150)
        self.fig, _ = self.active_qubit.plot_evals_vs_paramvals(scan_value, np_list, evals_count=eigenvalue_amount_value, subtract_ground=subtract_ground_tf)

    def zeropi_evals_vs_paramvals_interactive(self, 
        scan_value: str, 
        scan_range: Tuple[ float, float ], 
        grid_range: Tuple[ float, float ],
        eigenvalue_amount_value: int, 
        subtract_ground_tf: bool, 
        **params: Dict[ str, Union[ float, int ] ]) -> None:
        """This is the method associated with qubit_plot_interactive that allows for us to interact with plot_evals_vs_paramvals().

        Parameters
        ----------
        scan_value:
            Current value of the scan parameter dropdown.

        scan_range:
            Sets the interval [ min, max ] through
            which plot_evals_vs_paramvals() will plot over.

        grid_range:
            Current position in the grid.

        eigenvalue_amount_value:
            The number of eigenvalues that will be plotted.

        subtract_ground_tf:
            Determines whether we subtract away the ground energy or not.
            Initially set to False.

        **params:
            Dictionary of current qubit parameter values (taken from the sliders)
        """
        params.update(self.grid_defaults)
        scan_min, scan_max = scan_range
        grid_min, grid_max = grid_range
        self.active_qubit.set_params(**params)
        np_list = np.linspace(scan_min, scan_max, 50)
        dynamic_grid = scq.Grid1d(min_val = grid_min, max_val = grid_max, pt_count = 50)
        self.fig, _ = self.active_qubit.plot_evals_vs_paramvals(scan_value, np_list, grid = dynamic_grid, evals_count=eigenvalue_amount_value, subtract_ground=subtract_ground_tf)

    def matelem_vs_paramvals_interactive(self, 
        operator_value: str, 
        scan_value: str, 
        scan_range: Tuple[ float, float ], 
        matrix_element_amount_value: int,  
        mode_value: str, 
        **params: Dict[ str, Union[ float, int ] ]) -> None:
        """This is the method associated with qubit_plot_interactive that allows for us to interact with plot_matelem_vs_paramvals().

        Parameters
        ----------
        operator_value:
            Current value of the operator dropdown.

        scan_value:
            Current value of the scan parameter dropdown.

        scan_range:
            Sets the interval [ min, max ] through
            which plot_matelem_vs_paramvals() will plot over.

        matrix_element_amount_value:
            The number of elements that will be shown.

        mode_value:
            Current value of the mode dropdown.

        **params:
            Dictionary of current qubit parameter values (taken from the sliders)
        """
        scan_min, scan_max = scan_range
        self.active_qubit.set_params(**params)
        np_list = np.linspace(scan_min,scan_max,150)
        self.fig, _ = self.active_qubit.plot_matelem_vs_paramvals(operator_value, scan_value, np_list, select_elems=matrix_element_amount_value, mode = mode_value)
    
    def zeropi_matelem_vs_paramvals_interactive(self, 
        operator_value: str, 
        scan_value: str, 
        scan_range: Tuple[ float, float ], 
        grid_range: Tuple[ float, float ],
        matrix_element_amount_value: int,  
        mode_value: str, 
        **params: Dict[ str, Union[ float, int ] ]) -> None:
        """This is the method associated with qubit_plot_interactive that allows for us to interact with plot_matelem_vs_paramvals().

        Parameters
        ----------
        operator_value:
            Current value of the operator dropdown.

        scan_value:
            Current value of the scan parameter dropdown.

        scan_range:
            Sets the interval [ min, max ] through
            which plot_matelem_vs_paramvals() will plot over.

        grid_range:
            Current position in the grid.
            
        matrix_element_amount_value:
            The number of elements that will be shown.

        mode_value:
            Current value of the mode dropdown.

        **params:
            Dictionary of current qubit parameter values (taken from the sliders)
        """
        params.update(self.grid_defaults)
        scan_min, scan_max = scan_range
        grid_min, grid_max = grid_range
        self.active_qubit.set_params(**params)
        np_list = np.linspace(scan_min,scan_max,50)
        dynamic_grid = scq.Grid1d(min_val = grid_min, max_val = grid_max, pt_count = 50)
        self.fig, _ = self.active_qubit.plot_matelem_vs_paramvals(operator_value, scan_value, np_list, grid = dynamic_grid, select_elems=matrix_element_amount_value, mode = mode_value)

    def wavefunction_interactive(self, 
        eigenvalue: Union[ List[ int ], int ], 
        mode_value: str, 
        manual_scale_tf: bool,
        scale_value: float,
        **params: Dict[ str, Union[ float, int ] ]) -> None:  
        """This is the method associated with qubit_plot_interactive that allows for us to interact with plot_wavefunction().

        Parameters
        ----------
        eigenvalue:
            If the active qubit is not FluxQubit, then eigenvalue is the current list of eigenvalues that will be plotted.
            If the active qubit is FluxQubit, then eigenvalue is the current eigenvalue specified that will be plotted.

        mode_value:
            Current value of the mode dropdown.

        manual_scale_tf:

        **params:
            Dictionary of current qubit parameter values (taken from the sliders)
        """
        if manual_scale_tf:
            self.qubit_plot_options_widgets['wavefunction_scale_slider'].disabled = False
        else:
            self.qubit_plot_options_widgets['wavefunction_scale_slider'].disabled = True
            scale_value = None 

        self.active_qubit.set_params(**params)
        self.fig, _ = self.active_qubit.plot_wavefunction(which = eigenvalue, mode = mode_value, scaling = scale_value)
      
    def fluxqubit_wavefunction_interactive(self, 
        eigenvalue: Union[ List[ int ], int ], 
        mode_value: str, 
        manual_scale_tf: bool,
        **params: Dict[ str, Union[ float, int ] ]) -> None:  
        """This is the method associated with qubit_plot_interactive that allows for us to interact with plot_wavefunction().

        Parameters
        ----------
        eigenvalue:
            If the active qubit is not FluxQubit, then eigenvalue is the current list of eigenvalues that will be plotted.
            If the active qubit is FluxQubit, then eigenvalue is the current eigenvalue specified that will be plotted.

        mode_value:
            Current value of the mode dropdown.

        manual_scale_tf:

        **params:
            Dictionary of current qubit parameter values (taken from the sliders)
        """
        self.active_qubit.set_params(**params)
        self.fig, _ = self.active_qubit.plot_wavefunction(which = eigenvalue, mode = mode_value)

    def zeropi_wavefunction_interactive(self, 
        eigenvalue: Union[ List[ int ], int ], 
        grid_range: Tuple[ float, float ],
        mode_value: str, 
        **params: Dict[ str, Union[ float, int ] ]) -> None:  
        """This is the method associated with qubit_plot_interactive that allows for us to interact with plot_wavefunction().

        Parameters
        ----------
        eigenvalue:
            If the active qubit is not FluxQubit, then eigenvalue is the current list of eigenvalues that will be plotted.
            If the active qubit is FluxQubit, then eigenvalue is the current eigenvalue specified that will be plotted.

        grid_range:
            Current position in the grid.

        mode_value:
            Current value of the mode dropdown.

        **params:
            Dictionary of current qubit parameter values (taken from the sliders)
        """
        params.update(self.grid_defaults)
        grid_min, grid_max = grid_range
        self.active_qubit.set_params(**params)
        dynamic_grid = scq.Grid1d(min_val = grid_min, max_val = grid_max, pt_count = 50)
        self.fig, _ = self.active_qubit.plot_wavefunction(which = eigenvalue, grid = dynamic_grid, mode = mode_value)

    def matrixelements_interactive(self, 
        operator_value: str, 
        eigenvalue_amount_value: int, 
        mode_value: str, 
        show_numbers_tf: bool, 
        show3d_tf: bool, 
        **params: Dict[ str, Union[ float, int ] ]):
        """This is the method associated with qubit_plot_interactive that allows for us to interact with plot_matrixelements().

        Parameters
        ----------
        operator_value:
            Current value of the operator dropdown.

        eigenvalue_amount_value:
            The number of eigenvalues that will be plotted

        mode_value:
            Current value of the mode operator.

        show_numbers_tf:
            Determines whether the numerical values will be shown in the 2D plot.
            Initially set to False.
            
        show3d_tf:
            Determines whether a 3D version of the 2D plot will be shown.
            Initially set to True.

        **params:
            Dictionary of current qubit parameter values (taken from the sliders)
        """
        self.active_qubit.set_params(**params)
        self.fig, _ = self.active_qubit.plot_matrixelements(operator_value, evals_count=eigenvalue_amount_value, mode = mode_value, show_numbers = show_numbers_tf, show3d = show3d_tf)

    def zeropi_matrixelements_interactive(self, 
        operator_value: str, 
        eigenvalue_amount_value: int, 
        mode_value: str, 
        grid_range: Tuple[ float, float ],
        show_numbers_tf: bool, 
        show3d_tf: bool, 
        **params: Dict[ str, Union[ float, int ] ]):
        """This is the method associated with qubit_plot_interactive that allows for us to interact with plot_matrixelements().

        Parameters
        ----------
        operator_value:
            Current value of the operator dropdown.

        eigenvalue_amount_value:
            The number of eigenvalues that will be plotted

        mode_value:
            Current value of the mode operator.

        grid_range:
            Current position in the grid.

        show_numbers_tf:
            Determines whether the numerical values will be shown in the 2D plot.
            Initially set to False.
            
        show3d_tf:
            Determines whether a 3D version of the 2D plot will be shown.
            Initially set to True.

        **params:
            Dictionary of current qubit parameter values (taken from the sliders)
        """
        params.update(self.grid_defaults)
        grid_min, grid_max = grid_range
        self.active_qubit.set_params(**params)
        dynamic_grid = scq.Grid1d(min_val = grid_min, max_val = grid_max, pt_count = 50)
        self.fig, _ = self.active_qubit.plot_matrixelements(operator_value, evals_count=eigenvalue_amount_value, mode = mode_value, grid = dynamic_grid, show_numbers = show_numbers_tf, show3d = show3d_tf)

    #Create Methods -------------------------------------------------------------------------------------------------
    def create_params_dict(self) -> None:
        """Initializes qubit_base_params and qubit_dropdown_params.
        """
        self.qubit_base_params = dict(self.active_qubit.default_params())
        del self.qubit_base_params['truncated_dim']

        self.qubit_scan_params = dict(self.qubit_base_params)
        if 'ncut' in self.qubit_scan_params.keys():
            del self.qubit_scan_params['ncut']
        elif 'cutoff' in self.qubit_scan_params.keys():
            del self.qubit_scan_params['cutoff']

    def create_plot_settings_widgets(self):
        """Creates all the widgets that will be used
        for general plotting options. These will be all the widgets in the first column.
        """
        self.qubit_plot_options_widgets = {}
        std_layout = Layout(width='300px')

        operator_dropdown_list = self.get_operators()
        scan_dropdown_list = self.qubit_scan_params.keys()
        mode_dropdown_list = [('Re(·)', 'real'), ('Im(·)', 'imag'), ('|·|', 'abs'), (u'|\u00B7|\u00B2', 'abs_sqr')]
        
        file = open(self.active_qubit._image_filename, "rb")
        image = file.read()

        self.qubit_plot_options_widgets = {
            'qubit_info_image_widget': widgets.Image(
                value = image, 
                format = 'jpg', 
                layout = Layout(width='700px')),
            'save_button': widgets.Button(
                icon = 'save', 
                layout = widgets.Layout(width='35px')),
            'filename_text': widgets.Text(
                value ='plot.pdf', 
                description ='', 
                disabled = False),
            'scan_dropdown': widgets.Dropdown(
                options = scan_dropdown_list, 
                value = self.active_defaults['scan_param'], 
                description = 'Scan over', 
                disabled = False, 
                layout = std_layout),
            'mode_dropdown': widgets.Dropdown(
                options = mode_dropdown_list, 
                description = 'Plot as:', 
                disabled = False, 
                layout = std_layout),
            'operator_dropdown': widgets.Dropdown(
                options = operator_dropdown_list, 
                value = self.active_defaults['operator'], 
                description = 'Operator', 
                disabled = False, 
                layout = std_layout),
            'scan_range_slider': widgets.FloatRangeSlider(
                min = self.active_defaults[self.active_defaults['scan_param']]['min'], 
                max = self.active_defaults[self.active_defaults['scan_param']]['max'], 
                value = [ self.active_defaults[self.active_defaults['scan_param']]['min'], self.active_defaults[self.active_defaults['scan_param']]['max'] ],
                step = 0.05,
                description = '{} range'.format(self.active_defaults['scan_param']), 
                continuous_update = False,
                layout = std_layout),
            'grid_range_slider': widgets.FloatRangeSlider( 
                min = -12*np.pi, 
                max = 12*np.pi, 
                value = [ -6*np.pi, 6*np.pi ],
                step = 0.05,
                description = 'Grid range',
                continuous_update = False,
                layout = std_layout),
            'eigenvalue_amount_slider': widgets.IntSlider(
                min = 1, 
                max = 10, 
                value = 7, 
                description = 'Highest state', 
                continuous_update=False, 
                layout = std_layout),
            'matrix_element_amount_slider': widgets.IntSlider(
                min = 1, 
                max = 6,
                value = 4, 
                description = 'Highest state', 
                continuous_update = False, 
                layout = std_layout),
            'fluxqubit_state_slider': widgets.IntSlider(
                min = 0, 
                max = 10, 
                value = 0, 
                description = 'State', 
                continuous_update = False, 
                layout = std_layout),
            'wavefunction_scale_slider': widgets.FloatSlider(
                min = 0.1,
                max = 15,
                value = self.active_defaults['scale'],
                description = '\u03c8 ampl.',
                continuous_update = False,
                layout = std_layout),
            'qubit_state_selector': widgets.SelectMultiple(
                options = range( 0, 10 ), 
                value = [ 0, 1, 2 ], 
                description = 'States', 
                disabled = False, 
                continuous_update = False, 
                layout = std_layout),
            'show_numbers_checkbox': widgets.Checkbox(
                value = False, 
                description = 'Show values', 
                disabled = False),
            'show3d_checkbox': widgets.Checkbox(
                value = True, 
                description = 'Show 3D', 
                disabled = False),
            'subtract_ground_checkbox': widgets.Checkbox(
                value = False, 
                description = 'Subtract E\u2080', 
                disabled = False),
            'manual_scale_checkbox': widgets.Checkbox(
                value = False,
                description = 'Manual Scaling',
                disabled = False)
        }
        self.qubit_plot_options_widgets['save_button'].on_click(self.save_button_clicked_action)
        self.qubit_plot_options_widgets['scan_dropdown'].observe(self.scan_dropdown_eventhandler, names='value')    

    def create_qubit_params_widgets(self):
        """Creates all the widgets that will be used 
        for changing the parameter values for the specified qubit.
        These will be all the widgets after the first column.
        """
        #We need to clear qubit_params_widgets since the previous widgets from the old qubit will still be initialized otherwise.
        self.qubit_params_widgets.clear()
        for param_name, param_val in self.qubit_base_params.items():
            if param_name == 'grid':
                continue 

            if isinstance(param_val, int):
                kwargs = self.active_defaults.get(param_name) or self.active_defaults['int']
                self.qubit_params_widgets[param_name] = widgets.IntSlider(
                    **kwargs, 
                    value = param_val, 
                    description = '{}:'.format(param_name), 
                    continuous_update=False,
                    layout = Layout(width='300px'))
            else:
                kwargs = self.active_defaults.get(param_name) or self.active_defaults['float']
                self.qubit_params_widgets[param_name] = widgets.FloatSlider(
                    **kwargs, 
                    value = param_val, 
                    description = '{}:'.format(param_name), 
                    continuous_update=False,
                    layout = Layout(width='300px'))

    def create_qubit_and_plot_choice_widgets(self):
        """Creates all the widgets that controls 
        which qubit or plot the user can choose from.
        """
        self.qubit_and_plot_choice_widgets = {
            'qubit_buttons' : widgets.ToggleButtons(
                options=self.supported_qubits, 
                description='Qubits:', 
                layout=widgets.Layout(width='800px')),
            'plot_buttons' : widgets.ToggleButtons(
                options=self.get_plot_choices(), 
                description='Plot:', 
                button_style='info'),
            'show_qubitinfo_checkbox' : widgets.Checkbox(
                value = False, 
                description = 'qubit info', 
                disabled = False)
        }

    def create_plot_option_columns(self, qubit_plot_interactive: widgets.interactive) -> List[ widgets.VBox ]:
        """Divides the widgets into columns.

        Parameters
        ----------
        qubit_plot_interactive:
            Current interactive chosen.

        Returns
        -------
        List[ widgets.VBox ]
            Each widgets.VBox contains a list of widgets.
            The first element of the list contains the plot_widgets 
            while the remaining elements contain qubit_params_widgets.
        """
        widgets_per_column = 7
        base_index = (len(qubit_plot_interactive.children) - 1) - len(self.qubit_base_params)
        initial_index = base_index
        end_index = base_index + widgets_per_column
        widget_list = [VBox([*qubit_plot_interactive.children[0:base_index]])] 
        
        while end_index < len(qubit_plot_interactive.children):
            widget_list.append(VBox([*qubit_plot_interactive.children[initial_index:end_index]]))
            initial_index += widgets_per_column
            end_index += widgets_per_column
        widget_list.append(VBox([*qubit_plot_interactive.children[initial_index:-1]]))
        return widget_list

    def display_interactive(self, qubit_plot_interactive: widgets.interactive) -> None:
        """Displays an organized output for the current interactive

        Parameters
        ----------
        qubit_plot_interactive: widgets.interactive
            Current interactive chosen.
        """
        if qubit_plot_interactive is None:
            display('FullZeroPi currently does not have Wavefunctions implemented.')
            return None
        
        output = qubit_plot_interactive.children[-1]
        output.layout = Layout(align_items = 'center')
        widget_columns = self.create_plot_option_columns(qubit_plot_interactive)
        qubit_plot_interactive.children = (widgets.HBox(
                                    widget_columns, 
                                    layout=Layout(margin='2px'), 
                                    box_style='info'), 
                                widgets.HBox([
                                    self.qubit_plot_options_widgets['save_button'], 
                                    self.qubit_plot_options_widgets['filename_text']
                                    ], 
                                    layout=Layout(margin='2px', justify_content='flex-end')), 
                                output)
        display(qubit_plot_interactive)

    def display_qubit_info(self, qubit_info: bool) -> None:
        """

        Parameters
        ----------
        qubit_info: bool
        """
        if qubit_info:
            image_box = widgets.Box(layout = Layout(justify_content = 'center'))
            image_box.children = [ self.qubit_plot_options_widgets['qubit_info_image_widget'] ]
            display(image_box)
    
    def energy_scan_qubit_plot(self) -> widgets.interactive:
        self.qubit_params_widgets[self.qubit_plot_options_widgets['scan_dropdown'].value].disabled = True

        if isinstance(self.active_qubit, scq.ZeroPi) or isinstance(self.active_qubit, scq.FullZeroPi):
            qubit_plot_interactive = widgets.interactive(
                self.zeropi_evals_vs_paramvals_interactive,
                scan_value = self.qubit_plot_options_widgets['scan_dropdown'], 
                scan_range = self.qubit_plot_options_widgets['scan_range_slider'], 
                grid_range = self.qubit_plot_options_widgets['grid_range_slider'],
                subtract_ground_tf = self.qubit_plot_options_widgets['subtract_ground_checkbox'], 
                eigenvalue_amount_value = self.qubit_plot_options_widgets['eigenvalue_amount_slider'], 
                **self.qubit_params_widgets)
        else:
            qubit_plot_interactive = widgets.interactive(
                self.evals_vs_paramvals_interactive,
                scan_value = self.qubit_plot_options_widgets['scan_dropdown'], 
                scan_range = self.qubit_plot_options_widgets['scan_range_slider'], 
                subtract_ground_tf = self.qubit_plot_options_widgets['subtract_ground_checkbox'], 
                eigenvalue_amount_value = self.qubit_plot_options_widgets['eigenvalue_amount_slider'], 
                **self.qubit_params_widgets)

        return qubit_plot_interactive

    def matelem_scan_qubit_plot(self) -> widgets.interactive:
        self.qubit_plot_options_widgets['mode_dropdown'].value = self.active_defaults['mode_matrixelem']
        self.qubit_params_widgets[self.qubit_plot_options_widgets['scan_dropdown'].value].disabled = True

        if isinstance(self.active_qubit, scq.ZeroPi) or isinstance(self.active_qubit, scq.FullZeroPi):
            qubit_plot_interactive = widgets.interactive(
                self.zeropi_matelem_vs_paramvals_interactive, 
                operator_value = self.qubit_plot_options_widgets['operator_dropdown'],
                scan_value = self.qubit_plot_options_widgets['scan_dropdown'], 
                scan_range = self.qubit_plot_options_widgets['scan_range_slider'], 
                grid_range = self.qubit_plot_options_widgets['grid_range_slider'],
                matrix_element_amount_value = self.qubit_plot_options_widgets['matrix_element_amount_slider'], 
                mode_value = self.qubit_plot_options_widgets['mode_dropdown'],
                **self.qubit_params_widgets)
        else:
            qubit_plot_interactive = widgets.interactive(
                self.matelem_vs_paramvals_interactive, 
                operator_value = self.qubit_plot_options_widgets['operator_dropdown'],
                scan_value = self.qubit_plot_options_widgets['scan_dropdown'], 
                scan_range = self.qubit_plot_options_widgets['scan_range_slider'], 
                matrix_element_amount_value = self.qubit_plot_options_widgets['matrix_element_amount_slider'], 
                mode_value = self.qubit_plot_options_widgets['mode_dropdown'],
                **self.qubit_params_widgets)

        return qubit_plot_interactive

    def wavefunction_qubit_plot(self) -> widgets.interactive:
        if isinstance(self.active_qubit, scq.FullZeroPi):
            qubit_plot_interactive = None
        else:
            self.qubit_plot_options_widgets['mode_dropdown'].value = self.active_defaults['mode_wavefunc']
            self.qubit_params_widgets[self.qubit_plot_options_widgets['scan_dropdown'].value].disabled = False

            if isinstance(self.active_qubit, scq.FluxQubit) or isinstance(self.active_qubit, scq.ZeroPi):
                which_widget = self.qubit_plot_options_widgets['fluxqubit_state_slider']
            else:
                which_widget = self.qubit_plot_options_widgets['qubit_state_selector']

            if isinstance(self.active_qubit, scq.ZeroPi):
                qubit_plot_interactive = widgets.interactive(
                                                self.zeropi_wavefunction_interactive,  
                                                eigenvalue = which_widget,
                                                grid_range = self.qubit_plot_options_widgets['grid_range_slider'],
                                                mode_value = self.qubit_plot_options_widgets['mode_dropdown'],
                                                **self.qubit_params_widgets)
            elif isinstance(self.active_qubit, scq.FluxQubit):
                qubit_plot_interactive = widgets.interactive(
                                self.fluxqubit_wavefunction_interactive,  
                                eigenvalue = which_widget,
                                mode_value = self.qubit_plot_options_widgets['mode_dropdown'],
                                manual_scale_tf = self.qubit_plot_options_widgets['manual_scale_checkbox'],
                                **self.qubit_params_widgets)
            else:
                qubit_plot_interactive = widgets.interactive(
                                self.wavefunction_interactive,  
                                eigenvalue = which_widget,
                                mode_value = self.qubit_plot_options_widgets['mode_dropdown'],
                                manual_scale_tf = self.qubit_plot_options_widgets['manual_scale_checkbox'],
                                scale_value = self.qubit_plot_options_widgets['wavefunction_scale_slider'],
                                **self.qubit_params_widgets)
        
        return qubit_plot_interactive

    def matelem_qubit_plot(self) -> widgets.interactive:
        self.qubit_plot_options_widgets['mode_dropdown'].value = self.active_defaults['mode_matrixelem']
        self.qubit_params_widgets[self.qubit_plot_options_widgets['scan_dropdown'].value].disabled = False

        if isinstance(self.active_qubit, scq.ZeroPi) or isinstance(self.active_qubit, scq.FullZeroPi):
            qubit_plot_interactive = widgets.interactive(
                                            self.zeropi_matrixelements_interactive, 
                                            operator_value = self.qubit_plot_options_widgets['operator_dropdown'],                                                  
                                            eigenvalue_amount_value = self.qubit_plot_options_widgets['eigenvalue_amount_slider'], 
                                            mode_value = self.qubit_plot_options_widgets['mode_dropdown'],
                                            grid_range = self.qubit_plot_options_widgets['grid_range_slider'],
                                            show_numbers_tf = self.qubit_plot_options_widgets['show_numbers_checkbox'],
                                            show3d_tf = self.qubit_plot_options_widgets['show3d_checkbox'],
                                            **self.qubit_params_widgets)
        else:
            qubit_plot_interactive = widgets.interactive(
                                            self.matrixelements_interactive, 
                                            operator_value = self.qubit_plot_options_widgets['operator_dropdown'],                                                  
                                            eigenvalue_amount_value = self.qubit_plot_options_widgets['eigenvalue_amount_slider'], 
                                            mode_value = self.qubit_plot_options_widgets['mode_dropdown'],
                                            show_numbers_tf = self.qubit_plot_options_widgets['show_numbers_checkbox'],
                                            show3d_tf = self.qubit_plot_options_widgets['show3d_checkbox'],
                                            **self.qubit_params_widgets)
        return qubit_plot_interactive

    def create_qubit_plot_interactive(self, plot_value: str) -> widgets.interactive:
        if plot_value == 'Energy spectrum':
            return self.energy_scan_qubit_plot()
        elif plot_value == 'Matrix element scan':
            return self.matelem_scan_qubit_plot()
        elif plot_value == 'Wavefunctions':
            return self.wavefunction_qubit_plot()
        elif plot_value == 'Matrix elements':
            return self.matelem_qubit_plot()

    def display_qubit_plot_interactive(self, 
        qubit_value: str, 
        qubit_info: bool, 
        plot_value: str) -> None:
        """Creates the interactive and then displays it.

        Parameters
        ----------
        qubit_value:
            Current qubit chosen.

        qubit_info:

        plot_value:
            Current plot option chosen
        """       
        self.set_qubit(qubit_value)
        self.display_qubit_info(qubit_info)
        qubit_plot_interactive = self.create_qubit_plot_interactive(plot_value)
        self.display_interactive(qubit_plot_interactive)

    def create_GUI(self) -> Tuple[ widgets.VBox, widgets.interactive_output ]:
        """Creates an interactive (e.g. the buttons at the top) that 
        interacts with qubit_plot_interactive.

        Returns
        -------
        Tuple[ widgets.VBox, widgets.interactive_output ]
            
        """
        qubit_choice_hbox = widgets.HBox([self.qubit_and_plot_choice_widgets['qubit_buttons'], self.qubit_and_plot_choice_widgets['show_qubitinfo_checkbox']])
        plot_choice_hbox  = widgets.HBox([self.qubit_and_plot_choice_widgets['plot_buttons']])
        
        qubit_and_plot_choice_widgets = widgets.VBox([qubit_choice_hbox, plot_choice_hbox])
        
        qubit_and_plot_choice_interactive = widgets.interactive_output(
                                            self.display_qubit_plot_interactive,
                                            {'qubit_value': self.qubit_and_plot_choice_widgets['qubit_buttons'],
                                            'qubit_info': self.qubit_and_plot_choice_widgets['show_qubitinfo_checkbox'],
                                            'plot_value': self.qubit_and_plot_choice_widgets['plot_buttons']})
        qubit_and_plot_choice_interactive.layout.width = '975px'

        return qubit_and_plot_choice_widgets, qubit_and_plot_choice_interactive