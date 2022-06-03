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

paramvals_from_papers = {
    "Transmon": {
        "Paik et al. [J1, 3d], PRL 107, 240501, 1779 (2011)": {
            "EJ": 21.1,
            "EC": 0.301,
        },
        "ibm_manila Q1 (04/2022)": {
            "EJ": 11.34,
            "EC": 0.293,
        },
        "[CPB] Bouchiat et al., Phys. Scr. 165 (1998)": {
            "EJ": 4.167,
            "EC": 52.09,
        }
    },
    "TunableTransmon": {
        "Arute et al. [mean], Nature 574, 505 (2019)": {
            "EJmax": 32.7,
            "EC": 0.195,
        },
    },
    "Fluxonium": {
        "Manucharyan et al., Science 326, 113 (2009)": {
            "EJ": 9.0,
            "EC": 2.5,
            "EL": 0.52
        },
        "Zhang et al. [heavy], PRX 111, 011010 (2021)": {
            "EJ": 3.395,
            "EC": 0.479,
            "EL": 0.132
        },
        "Pechenezhskiy et al. [blochnium], Nature 585, 368 (2020)": {
            "EJ": 3.395,
            "EC": 0.479,
            "EL": 0.132
        },
    },
    "ZeroPi": {
        "Gyenis et al., PRX Quantum 2, 010339 (2021)": {
            "EJ": 6.0,
            "ECJ": 2.28,
            "ECS": 0.184,
            "EL": 0.38
        },
        "Groszkowski et al. [set 1, deep], NJP 20, 043053 (2018)": {
            "EJ": 10.0,
            "ECJ": 20.0,
            "EC": 0.02,
            "EL": 0.008
        },
        "Groszkowski et al. [set 2, soft],  NJP 20, 043053 (2018)": {
            "EJ": 10.0,
            "ECJ": 20.0,
            "EC": 0.04,
            "EL": 0.04
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
