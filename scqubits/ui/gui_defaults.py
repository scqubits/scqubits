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


global_defaults = {
    "mode_wavefunc": "real",
    "mode_matrixelem": "abs",
    "ng": {"min": 0, "max": 1},
    "flux": {"min": 0, "max": 1},
    "EJ": {"min": 1e-10, "max": 70},
    "EC": {"min": 1e-10, "max": 5},
    "int": {"min": 1, "max": 30},
    "float": {"min": 0, "max": 30},
}

transmon_defaults = {
    **global_defaults,
    "scan_param": "ng",
    "operator": "n_operator",
    "ncut": {"min": 10, "max": 50},
    "scale": 1,
    "num_sample": 150,
}

tunabletransmon_defaults = {
    **global_defaults,
    "scan_param": "flux",
    "operator": "n_operator",
    "EJmax": global_defaults["EJ"],
    "d": {"min": 0, "max": 1},
    "ncut": {"min": 10, "max": 50},
    "scale": 1,
    "num_sample": 150,
}

fluxonium_defaults = {
    **global_defaults,
    "scan_param": "flux",
    "operator": "n_operator",
    "EC": {"min": 1e-2, "max": 5},
    "EL": {"min": 1e-10, "max": 2},
    "cutoff": {"min": 10, "max": 120},
    "scale": 1,
    "num_sample": 150,
}

fluxqubit_defaults = {
    **global_defaults,
    "scan_param": "flux",
    "operator": "n_1_operator",
    "ncut": {"min": 5, "max": 30},
    "EJ1": global_defaults["EJ"],
    "EJ2": global_defaults["EJ"],
    "EJ3": global_defaults["EJ"],
    "ECJ1": global_defaults["EC"],
    "ECJ2": global_defaults["EC"],
    "ECJ3": global_defaults["EC"],
    "ECg1": global_defaults["EC"],
    "ECg2": global_defaults["EC"],
    "ng1": global_defaults["ng"],
    "ng2": global_defaults["ng"],
    "scale": None,
    "num_sample": 100,
}

zeropi_defaults = {
    **global_defaults,
    "scan_param": "flux",
    "operator": "n_theta_operator",
    "ncut": {"min": 5, "max": 50},
    "EL": {"min": 1e-10, "max": 3},
    "ECJ": {"min": 1e-10, "max": 30},
    "dEJ": {"min": 0, "max": 1},
    "dCJ": {"min": 0, "max": 1},
    "scale": None,
    "num_sample": 50,
}

fullzeropi_defaults = {
    **global_defaults,
    "scan_param": "flux",
    "operator": "n_theta_operator",
    "ncut": {"min": 5, "max": 50},
    "EL": {"min": 1e-10, "max": 3},
    "ECJ": {"min": 1e-10, "max": 30},
    "dEJ": {"min": 0, "max": 1},
    "dCJ": {"min": 0, "max": 1},
    "dEL": {"min": 0, "max": 1},
    "dC": {"min": 0, "max": 1},
    "zeropi_cutoff": {"min": 5, "max": 30},
    "zeta_cutoff": {"min": 5, "max": 30},
    "scale": None,
    "num_sample": 50,
}

cos2phiqubit_defaults = {
    **global_defaults,
    "scan_param": "flux",
    "operator": "phi_operator",
    "EL": {"min": 1e-10, "max": 5},
    "ECJ": {"min": 1e-10, "max": 30},
    "dEJ": {"min": 0, "max": 0.99},
    "dL": {"min": 0, "max": 0.99},
    "dCJ": {"min": 0, "max": 0.99},
    "ncut": {"min": 5, "max": 50},
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

paramvals_from_papers = {
    "Transmon": {
        "Paik et al. [J1, 3d], PRL 107, 240501, 1779 (2011)": {
            "params": {
                "EJ": 21.1,
                "EC": 0.301,
            },
            "link": "https://www.researchgate.net/publication/221745534_Observation_of_High_Coherence_in_Josephson_Junction_Qubits_Measured_in_a_Three-Dimensional_Circuit_QED_Architecture"
        },
        "ibm_manila Q1 (04/2022)": {
            "params": {
                "EJ": 11.34,
                "EC": 0.293,
            },
            "link": ""
        },
        "[CPB] Bouchiat et al., Phys. Scr. 165 (1998)": {
            "params": {
                "EJ": 4.167,
                "EC": 52.09,
            },
            "link": "https://www.researchgate.net/publication/231107577_Quantum_Coherence_with_a_Single_Cooper_Pair"
        }
    },
    "TunableTransmon": {
        "Arute et al. [mean], Nature 574, 505 (2019)": {
            "params": {
                "EJmax": 32.7,
                "EC": 0.195,
            },
            "link": "https://www.researchgate.net/publication/336744162_Quantum_supremacy_using_a_programmable_superconducting_processor"
        },
    },
    "Fluxonium": {
        "Manucharyan et al., PRA 76, 042319 (2007)": {
            "params": {
                "EJ": 9.0,
                "EC": 2.5,
                "EL": 0.52,
            },
            "link": "https://www.researchgate.net/publication/1861297_Charge_insensitive_qubit_design_derived_from_the_Cooper_pair_box"
        },
        "Zhang et al. [heavy], PRX 111, 011010 (2021)": {
            "params": {
                "EJ": 3.395,
                "EC": 0.479,
                "EL": 0.132,
            },
            "link": "https://www.researchgate.net/publication/348531750_Universal_Fast-Flux_Control_of_a_Coherent_Low-Frequency_Qubit"
        },
        "Pechenezhskiy et al. [blochnium], Nature 585, 368 (2020)": {
            "params": {
                "EJ": 3.395,
                "EC": 0.479,
                "EL": 0.132,
            },
            "link": "https://www.researchgate.net/publication/344280109_The_superconducting_quasicharge_qubit"
        },
    },
    "ZeroPi": {
        "Gyenis et al., PRX Quantum 2, 010339 (2021)": {
            "params": {
                "EJ": 6.0,
                "ECJ": 2.28,
                "ECS": 0.184,
                "EL": 0.38,
            },
            "link": "https://www.researchgate.net/publication/349840068_Experimental_Realization_of_a_Protected_Superconducting_Circuit_Derived_from_the_0_-_p_Qubit"
        },
        "Groszkowski et al. [set 1, deep], NJP 20, 043053 (2018)": {
            "params": {
                "EJ": 10.0,
                "ECJ": 20.0,
                "EC": 0.02,
                "EL": 0.008,
            },
            "link": "https://www.researchgate.net/publication/331293936_Control_and_Coherence_Time_Enhancement_of_the_0-p_Qubit"
        },
        "Groszkowski et al. [set 2, soft],  NJP 20, 043053 (2018)": {
            "params": {
            "EJ": 10.0,
            "ECJ": 20.0,
            "EC": 0.04,
            "EL": 0.04,
            },
            "link": "https://www.researchgate.net/publication/331293936_Control_and_Coherence_Time_Enhancement_of_the_0-p_Qubit"
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

PLOT_HEIGHT = '500px'
FIG_WIDTH_INCHES = 6
