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
