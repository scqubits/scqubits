import numpy as np
import scqubits as scq

class Test_effective_scan:

    def time_transmon_t2_effective_scan(self):
        init_params = scq.Transmon.default_params()
        qbt = scq.Transmon(**init_params)
        flux_list = np.linspace(-1, 1, 151)
        for flux in flux_list:
            qbt.set_and_return('flux', flux).t2_effective()

    def time_fluxonium_t2_effective_scan(self):
        init_params = scq.Fluxonium.default_params()
        qbt = scq.Fluxonium(**init_params)
        flux_list = np.linspace(-1, 1, 151)
        for flux in flux_list: 
            qbt.set_and_return('flux', flux).t2_effective()

    def time_zeropi_t2_effective_scan(self):
        init_params = scq.ZeroPi.default_params()
        phi_grid = scq.Grid1d(-6*np.pi, 6*np.pi, 49)
        qbt = scq.ZeroPi(grid=phi_grid, **init_params)
        flux_list = np.linspace(-1, 1, 101)
        for flux in flux_list:
            qbt.set_and_return('flux', flux).t2_effective()