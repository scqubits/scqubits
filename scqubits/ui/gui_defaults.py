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

import base64
import collections
import enum
import os

try:
    import ipyvuetify as v
    import ipywidgets
except ImportError:
    _HAS_IPYVUETIFY = False
else:
    _HAS_IPYVUETIFY = True


gui_plot_choice_dict = collections.OrderedDict(
    [
        ("Energy spectrum", "En.png"),
        ("Wavefunctions", "psi.png"),
        ("Matrix elements", "Me.png"),
        ("Matrix-element sweep", "MeS.png"),
        ("Coherence times", "T1.png"),
    ]
)

gui_sweep_plots = [0, 3, 4]

gui_plot_icon_filenames = list(gui_plot_choice_dict.values())
gui_icon_filenames = gui_plot_icon_filenames + ["scq-logo.png"]
gui_plot_type_names = list(gui_plot_choice_dict.keys())


icons = {}
if _HAS_IPYVUETIFY:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")

    for name in gui_icon_filenames:
        full_path = os.path.join(path, name)
        file = open(full_path, "rb")
        image = file.read()
        image_base64 = base64.b64encode(image).decode("ascii")
        icons[name] = v.Img(src=f"data:image/png;base64,{image_base64}", width=50)


STEP = 1e-2
EL_range = {"v_min": STEP, "v_max": 10.0}
EJ_range = {"v_min": STEP, "v_max": 70.0}
EC_range = {"v_min": STEP, "v_max": 10.0}
flux_range = {"v_min": 0.0, "v_max": 1.0}
ng_range = {"v_min": 0.0, "v_max": 1.0}
int_range = {"v_min": 1, "v_max": 30}
float_range = {"v_min": 0.0, "v_max": 30.0}
ncut_range = {"v_min": 6, "v_max": 50}

global_defaults = {
    "mode_wavefunc": "Re(·)",
    "mode_matrixelem": "|·|",
    "ng": ng_range,
    "flux": flux_range,
    "EJ": EJ_range,
    "EC": EC_range,
    "int": int_range,
    "float": float_range,
    "scale": 1.0,
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
    "EJ_max": EJ_range,
    "d": {"v_min": 0.0, "v_max": 1.0},
    "ncut": ncut_range,
}

fluxonium_defaults = {
    **global_defaults,
    "scan_param": "flux",
    "operator": "n_operator",
    "EL": EL_range,
    "cutoff": {"v_min": 10, "v_max": 120},
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

snail_EJ_range = {"v_min": STEP, "v_max": 2e3}
snail_EC_range = {"v_min": STEP, "v_max": 300.0}

snailmon_defaults = {
    **global_defaults,
    "scan_param": "flux",
    "operator": "n_1_operator",
    "ncut": ncut_range,
    "EJ1": snail_EJ_range,
    "EJ2": snail_EJ_range,
    "EJ3": snail_EJ_range,
    "EJ4": snail_EJ_range,
    "ECJ1": snail_EC_range,
    "ECJ2": snail_EC_range,
    "ECJ3": snail_EC_range,
    "ECJ4": snail_EC_range,
    "ECg1": snail_EC_range,
    "ECg2": snail_EC_range,
    "ECg3": snail_EC_range,
    "ECg4": snail_EC_range,
    "ng1": ng_range,
    "ng2": ng_range,
    "ng3": ng_range,
    "scale": None,
    "num_sample": 100,
}

zeropi_defaults = {
    **global_defaults,
    "scan_param": "flux",
    "operator": "n_theta_operator",
    "ncut": ncut_range,
    "EL": EL_range,
    "ECJ": {"v_min": STEP, "v_max": 25.0},
    "dEJ": {"v_min": 0.0, "v_max": 1.0},
    "dCJ": {"v_min": 0.0, "v_max": 1.0},
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
    "dEJ": {"v_min": 0.0, "v_max": 1.0},
    "dCJ": {"v_min": 0.0, "v_max": 1.0},
    "dEL": {"v_min": 0.0, "v_max": 1.0},
    "dC": {"v_min": 0.0, "v_max": 1.0},
    "zeropi_cutoff": {"v_min": 5, "v_max": 30},
    "zeta_cutoff": {"v_min": 5, "v_max": 30},
    "scale": None,
    "num_sample": 50,
}

cos2phiqubit_defaults = {
    **global_defaults,
    "scan_param": "flux",
    "operator": "phi_operator",
    "EL": EL_range,
    "ECJ": EC_range,
    "dEJ": {"v_min": 0, "v_max": 0.99},
    "dL": {"v_min": 0, "v_max": 0.99},
    "dCJ": {"v_min": 0, "v_max": 0.99},
    "ncut": ncut_range,
    "zeta_cut": {"v_min": 10, "v_max": 50},
    "phi_cut": {"v_min": 5, "v_max": 30},
    "scale": None,
    "num_sample": 50,
}

bifluxon_defaults = {
    **global_defaults,
    "scan_param": "flux",
    "operator": "n_theta_operator",
    "ncut": ncut_range,
    "EL": EL_range,
    "ECL": EC_range,
    "dEJ": {"v_min": 0.0, "v_max": 1.0},
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
    "Snailmon": snailmon_defaults,
    "Bifluxon": bifluxon_defaults,
}

paramvals_from_papers = {
    "Transmon": {
        "Paik et al. [J1, 3d], PRL 107, 240501, 1779 (2011)": {
            "params": {
                "EJ": 21.1,
                "EC": 0.301,
            },
            "link": "https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.107.240501",
        },
        "ibm_manila Q1 (04/2022)": {
            "params": {
                "EJ": 11.34,
                "EC": 0.293,
            },
            "link": "https://quantum-computing.ibm.com/",
        },
        "[CPB] Bouchiat et al., Phys. Scr. 1998, 165 (1998)": {
            "params": {
                "EJ": 4.167,
                "EC": 52.09,
            },
            "link": "https://iopscience.iop.org/article/10.1238/Physica.Topical.076a00165/meta",
        },
    },
    "TunableTransmon": {
        "Arute et al. [mean], Nature 574, 505 (2019)": {
            "params": {
                "EJmax": 32.7,
                "EC": 0.195,
            },
            "link": "https://www.nature.com/articles/s41586-019-1666-5",
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
            "link": "https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.011010",
        },
        "Pechenezhskiy et al. [blochnium], Nature 585, 368 (2020)": {
            "params": {
                "EJ": 4.70,
                "EC": 7.07,
                "EL": 0.0665,
            },
            "link": "https://arxiv.org/pdf/1907.02937.pdf",
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
            "link": "https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.010339",
        },
        "Groszkowski et al. [set 1, deep], NJP 20, 043053 (2018)": {
            "params": {
                "EJ": 10.0,
                "ECJ": 20.0,
                "EC": 0.02,
                "EL": 0.008,
            },
            "link": "https://iopscience.iop.org/article/10.1088/1367-2630/aab7cd",
        },
        "Groszkowski et al. [set 2, soft],  NJP 20, 043053 (2018)": {
            "params": {
                "EJ": 10.0,
                "ECJ": 20.0,
                "EC": 0.04,
                "EL": 0.04,
            },
            "link": "https://iopscience.iop.org/article/10.1088/1367-2630/aab7cd",
        },
    },
    "Bifluxon": {
        "Kalashnikov et al., PRX Quantum 1, 010307 (2020)": {
            "params": {
                "EJ": 27.2,
                "EL": 0.94,
                "EC": 7.7,
                "ECL": 10.0,
                "dEJ": 0.0,
            },
            "link": "https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.1.010307",
        },
    },
}

# Plot categories available in the single-qubit GUI
plot_choices = [
    "Energy spectrum",
    "Wavefunctions",
    "Matrix elements",
    "Matrix element scan",
    "Coherence times",
]

# The following qubits are supported by the GUI
supported_qubits = [
    "Transmon",
    "TunableTransmon",
    "Fluxonium",
    "FluxQubit",
    "ZeroPi",
    "FullZeroPi",
    "Cos2PhiQubit",
    # "Snailmon",
    # "Bifluxon",
]

# The following qubits are supported by the GUI, but are slow, so auto-updating is disabled by default
slow_qubits = [
    "FluxQubit",
    "ZeroPi",
    "FullZeroPi",
    "Cos2PhiQubit",
    # "Snailmon",
    # "Bifluxon",
]


# Explorer plot names
class PlotType(enum.Enum):
    ENERGY_SPECTRUM = "Energy spectrum"
    WAVEFUNCTIONS = "Wavefunctions"
    MATRIX_ELEMENTS = "Matrix elements (fixed)"
    MATRIX_ELEMENT_SCAN = "Matrix elements (sweep)"
    ANHARMONICITY = "Anharmonicity"
    SELF_KERR = "Self-Kerr"
    TRANSITIONS = "Transitions"
    CROSS_KERR = "Cross-Kerr"
    AC_STARK = "ac Stark"


# Plot types associated with individual subsystems (used in Explorer class)
subsys_plot_types = (
    PlotType.ENERGY_SPECTRUM,
    PlotType.WAVEFUNCTIONS,
    PlotType.MATRIX_ELEMENTS,
    PlotType.MATRIX_ELEMENT_SCAN,
    PlotType.ANHARMONICITY,
    PlotType.SELF_KERR,
)


# Plot names for composite-system plots (used in Explorer class)
composite_plot_types = [PlotType.TRANSITIONS, PlotType.CROSS_KERR, PlotType.AC_STARK]


# Plots that are activated for all `supported_qubits` when entering the Explorer class
common_panels = [PlotType.ENERGY_SPECTRUM, PlotType.WAVEFUNCTIONS]

# Options for plotting complex-valued data
mode_dropdown_dict = {
    "Re(·)": "real",
    "Im(·)": "imag",
    "|·|": "abs",
    "|\u00B7|\u00B2": "abs_sqr",
}

mode_dropdown_list = list(mode_dropdown_dict.keys())

# Default panels for each qubit type, used as default in Explorer class
default_panels = {qubit_name: common_panels for qubit_name in supported_qubits}
default_panels["Oscillator"] = []
default_panels["KerrOscillator"] = []
default_panels["Composite"] = [PlotType.TRANSITIONS]

# Supported panels for each qubit type, used in Explorer class
supported_panels = {qubit_name: subsys_plot_types for qubit_name in supported_qubits}
supported_panels["Oscillator"] = [PlotType.ENERGY_SPECTRUM, PlotType.SELF_KERR]
supported_panels["KerrOscillator"] = [PlotType.ENERGY_SPECTRUM]
supported_panels["Composite"] = composite_plot_types

# Default plot options used in Explorer class
PLOT_HEIGHT = "500px"
FIG_WIDTH_INCHES = 6
FIG_DPI = 150
NAV_COLOR = "#deebf9"
