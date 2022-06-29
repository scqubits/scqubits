# gui_defaults.py
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

import numpy as np

EL_range = {"min": 1e-5, "max": 10.0}
EJ_range = {"min": 1e-5, "max": 70.0}
EC_range = {"min": 1e-5, "max": 10.0}
flux_range = {"min": 0.0, "max": 1.0}
ng_range = {"min": 0.0, "max": 1.0}
int_range = {"min": 1, "max": 30}
float_range = {"min": 0.0, "max": 30.0}
ncut_range = {"min": 10, "max": 50}

global_defaults = {
    "mode_wavefunc": "real",
    "mode_matrixelem": "abs",
    "ng": ng_range,
    "flux": flux_range,
    "EJ": EJ_range,
    "EC": EC_range,
    "int": int_range,
    "float": float_range,
    "scale": 1,
    "num_sample": 150,
}

transmon_defaults = {
    **global_defaults,
    "scan_param": "ng",
    "operator": "n_operator",
    "ncut": ncut_range,
}

tunabletransmon_defaults = {
    **global_defaults,
    "scan_param": "flux",
    "operator": "n_operator",
    "EJmax": EJ_range,
    "d": {"min": 0.0, "max": 1.0},
    "ncut": ncut_range,
}

fluxonium_defaults = {
    **global_defaults,
    "scan_param": "flux",
    "operator": "n_operator",
    "EL": EL_range,
    "cutoff": {"min": 10, "max": 120},
}

fluxqubit_defaults = {
    **global_defaults,
    "scan_param": "flux",
    "operator": "n_1_operator",
    "ncut": ncut_range,
    "EJ1": EJ_range,
    "EJ2": EJ_range,
    "EJ3": EJ_range,
    "ECJ1": EC_range,
    "ECJ2": EC_range,
    "ECJ3": EC_range,
    "ECg1": EC_range,
    "ECg2": EC_range,
    "ng1": ng_range,
    "ng2": ng_range,
    "scale": None,
    "num_sample": 100,
}

zeropi_defaults = {
    **global_defaults,
    "scan_param": "flux",
    "operator": "n_theta_operator",
    "ncut": ncut_range,
    "EL": EL_range,
    "ECJ": EC_range,
    "dEJ": {"min": 0.0, "max": 1.0},
    "dCJ": {"min": 0.0, "max": 1.0},
    "scale": None,
    "num_sample": 50,
}

fullzeropi_defaults = {
    **global_defaults,
    "scan_param": "flux",
    "operator": "n_theta_operator",
    "ncut": ncut_range,
    "EL": EL_range,
    "ECJ": EC_range,
    "dEJ": {"min": 0.0, "max": 1.0},
    "dCJ": {"min": 0.0, "max": 1.0},
    "dEL": {"min": 0.0, "max": 1.0},
    "dC": {"min": 0.0, "max": 1.0},
    "zeropi_cutoff": {"min": 5, "max": 30},
    "zeta_cutoff": {"min": 5, "max": 30},
    "scale": None,
    "num_sample": 50,
}

cos2phiqubit_defaults = {
    **global_defaults,
    "scan_param": "flux",
    "operator": "phi_operator",
    "EL": EL_range,
    "ECJ": EC_range,
    "dEJ": {"min": 0, "max": 0.99},
    "dL": {"min": 0, "max": 0.99},
    "dCJ": {"min": 0, "max": 0.99},
    "ncut": ncut_range,
    "zeta_cut": {"min": 10, "max": 50},
    "phi_cut": {"min": 5, "max": 30},
    "scale": None,
    "num_sample": 50,
}

qubit_defaults = {
    "Transmon": transmon_defaults,
    "TunableTransmon": tunabletransmon_defaults,
    "Fluxonium": fluxonium_defaults,
    "FluxQubit": fluxqubit_defaults,
    "ZeroPi": zeropi_defaults,
    "FullZeroPi": fullzeropi_defaults,
    "Cos2PhiQubit": cos2phiqubit_defaults,
}

grid_defaults = {
    "grid_min_val": -6 * np.pi,
    "grid_max_val": 6 * np.pi,
    "grid_pt_count": 50,
}

phi_grid_defaults = {
    "grid_min_val": -np.pi,
    "grid_max_val": np.pi,
    "grid_pt_count": 150,
}

paramvals_from_papers = {
    "Transmon": {
        "Paik et al. [J1, 3d], PRL 107, 240501, 1779 (2011)": {
            "params": {
                "EJ": 21.1,
                "EC": 0.301,
            },
            "link": "https://www.researchgate.net/publication/221745534_Observation_of_High_Coherence_in_Josephson_Junction_Qubits_Measured_in_a_Three-Dimensional_Circuit_QED_Architecture",
        },
        "ibm_manila Q1 (04/2022)": {
            "params": {
                "EJ": 11.34,
                "EC": 0.293,
            },
            "link": "",
        },
        "[CPB] Bouchiat et al., Phys. Scr. 165 (1998)": {
            "params": {
                "EJ": 4.167,
                "EC": 52.09,
            },
            "link": "https://www.researchgate.net/publication/231107577_Quantum_Coherence_with_a_Single_Cooper_Pair",
        },
    },
    "TunableTransmon": {
        "Arute et al. [mean], Nature 574, 505 (2019)": {
            "params": {
                "EJmax": 32.7,
                "EC": 0.195,
            },
            "link": "https://www.researchgate.net/publication/336744162_Quantum_supremacy_using_a_programmable_superconducting_processor",
        },
    },
    "Fluxonium": {
        "Manucharyan et al., Science 326, 113 (2009)": {
            "params": {
                "EJ": 9.0,
                "EC": 2.5,
                "EL": 0.52,
            },
            "link": "https://www.science.org/doi/10.1126/science.1175552",
        },
        "Zhang et al. [heavy], PRX 111, 011010 (2021)": {
            "params": {
                "EJ": 3.395,
                "EC": 0.479,
                "EL": 0.132,
            },
            "link": "https://www.researchgate.net/publication/348531750_Universal_Fast-Flux_Control_of_a_Coherent_Low-Frequency_Qubit",
        },
        "Pechenezhskiy et al. [blochnium], Nature 585, 368 (2020)": {
            "params": {
                "EJ": 3.395,
                "EC": 0.479,
                "EL": 0.132,
            },
            "link": "https://www.researchgate.net/publication/344280109_The_superconducting_quasicharge_qubit",
        },
    },
    "ZeroPi": {
        "Gyenis et al., PRX Quantum 2, 010339 (2021)": {
            "params": {
                "EJ": 6.0,
                "ECJ": 2.28,
                "EC": 0.184,
                "EL": 0.38,
            },
            "link": "https://www.researchgate.net/publication/349840068_Experimental_Realization_of_a_Protected_Superconducting_Circuit_Derived_from_the_0_-_p_Qubit",
        },
        "Groszkowski et al. [set 1, deep], NJP 20, 043053 (2018)": {
            "params": {
                "EJ": 10.0,
                "ECJ": 20.0,
                "EC": 0.02,
                "EL": 0.008,
            },
            "link": "https://www.researchgate.net/publication/331293936_Control_and_Coherence_Time_Enhancement_of_the_0-p_Qubit",
        },
        "Groszkowski et al. [set 2, soft],  NJP 20, 043053 (2018)": {
            "params": {
                "EJ": 10.0,
                "ECJ": 20.0,
                "EC": 0.04,
                "EL": 0.04,
            },
            "link": "https://www.researchgate.net/publication/331293936_Control_and_Coherence_Time_Enhancement_of_the_0-p_Qubit",
        },
    },
}

plot_choices = [
    "Energy spectrum",
    "Wavefunctions",
    "Matrix element scan",
    "Matrix elements",
]

supported_qubits = [
    "Transmon",
    "TunableTransmon",
    "Fluxonium",
    "FluxQubit",
    "ZeroPi",
    "FullZeroPi",
    "Cos2PhiQubit",
]

slow_qubits = ["FluxQubit", "ZeroPi", "FullZeroPi", "Cos2PhiQubit"]

subsys_panel_names = [
    "Energy spectrum",
    "Wavefunctions",
    "Matrix elements",
    "Anharmonicity",
    "Self-Kerr",
]

composite_panel_names = ["Transitions", "Cross-Kerr, ac-Stark", "Custom data"]

common_panels = ["Energy spectrum", "Wavefunctions"]

mode_dropdown_list = [
    ("Re(·)", "real"),
    ("Im(·)", "imag"),
    ("|·|", "abs"),
    ("|\u00B7|\u00B2", "abs_sqr"),
]

default_panels = {qubit_name: common_panels for qubit_name in supported_qubits}
default_panels["Oscillator"] = []
default_panels["KerrOscillator"] = []
default_panels["Composite"] = ["Transitions"]

PLOT_HEIGHT = "500px"
FIG_WIDTH_INCHES = 6
