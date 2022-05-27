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
        "Manucharyan et al., PRA 76, 042319 (2007)": {
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

PLOT_HEIGHT = '500px'
FIG_WIDTH_INCHES = 6
