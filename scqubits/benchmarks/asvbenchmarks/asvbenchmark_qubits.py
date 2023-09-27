import numpy as np
import scqubits as scq

# from scqubits.benchmarks.asvconftest import StandardTests
# import scqubits.benchmarks.asvconftest as asvtest


class TestQubit:
    def time_transmon_get_spectrum_vs_paramvals(self):
        init_params = scq.Transmon.default_params()
        qbt = scq.Transmon(**init_params)
        ng_list = np.linspace(-2, 2, 100)
        qbt.get_spectrum_vs_paramvals(
            "ng", ng_list, evals_count=4, subtract_ground=False, get_eigenstates=True
        )

    def time_tunabletransmon_get_spectrum_vs_paramvals(self):
        init_params = scq.TunableTransmon.default_params()
        qbt = scq.TunableTransmon(**init_params)
        flux = np.linspace(0, 1, 100)
        qbt.get_spectrum_vs_paramvals(
            "flux", flux, evals_count=4, subtract_ground=False, get_eigenstates=True
        )

    def time_fluxonium_get_spectrum_vs_paramvals(self):
        init_params = scq.Fluxonium.default_params()
        qbt = scq.Fluxonium(**init_params)
        flux_list = np.linspace(-0.5, 0.5, 51)
        qbt.get_spectrum_vs_paramvals(
            "flux",
            flux_list,
            evals_count=4,
            subtract_ground=False,
            get_eigenstates=True,
        )

    def time_fluxqubit_get_spectrum_vs_paramvals(self):
        init_params = scq.FluxQubit.default_params()
        qbt = scq.FluxQubit(**init_params)
        flux_list = np.linspace(0.45, 0.55, 50)
        qbt.get_spectrum_vs_paramvals(
            "flux",
            flux_list,
            evals_count=4,
            subtract_ground=False,
            get_eigenstates=True,
        )

    def time_cos2phi_get_spectrum_vs_paramvals(self):
        init_params = scq.Cos2PhiQubit.default_params()
        qbt = scq.Cos2PhiQubit(**init_params)
        flux_list = np.linspace(0.0, 0.5, 3)
        qbt.get_spectrum_vs_paramvals(
            "flux",
            flux_list,
            evals_count=3,
            subtract_ground=False,
            get_eigenstates=True,
        )

    def time_ZeroPi_get_spectrum_vs_paramvals(self):
        phi_grid = scq.Grid1d(-19.0, 19.0, 200)
        init_params = scq.ZeroPi.default_params()
        init_params["grid"] = phi_grid
        qbt = scq.ZeroPi(**init_params)
        flux_list = np.linspace(0.0, 0.5, 3)
        qbt.get_spectrum_vs_paramvals(
            "flux",
            flux_list,
            evals_count=3,
            subtract_ground=False,
            get_eigenstates=True,
        )

    def time_FullZeroPi_get_spectrum_vs_paramvals(self):
        phi_grid = scq.Grid1d(-19.0, 19.0, 200)
        init_params = scq.FullZeroPi.default_params()
        init_params["grid"] = phi_grid
        qbt = scq.FullZeroPi(**init_params)
        flux_list = np.linspace(0.0, 0.5, 3)
        qbt.get_spectrum_vs_paramvals(
            "flux",
            flux_list,
            evals_count=3,
            subtract_ground=False,
            get_eigenstates=True,
        )
