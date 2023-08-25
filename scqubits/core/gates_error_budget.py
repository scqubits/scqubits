import numpy as np
from qutip import destroy, sigmaz, tensor, sigmax, sigmay, propagator, Qobj, qeye, Options
from scipy.special import erf
import scipy as sp

from QRAM_utils.utils import id_wrap_ops, calc_fidel_chi, my_to_chi


def calc_fidel_4(prop, gate):
    return (
            np.abs(
                np.trace(prop.dag() * prop) + np.abs(np.trace(prop.dag() * gate)) ** 2
            )
            / 20
    )

def calc_fidel_2(prop, gate):
    return (
            np.abs(
                np.trace(prop.dag() * prop) + np.abs(np.trace(prop.dag() * gate)) ** 2
            )
            / 6
    )


class GatesErrorBudget:
    def __init__(self, Gamma_1s, Gamma_2s, dt=1.0):
        self.Gamma_1s = Gamma_1s
        self.Gamma_2s = Gamma_2s
        Gamma_phis = []
        for Gamma_1, Gamma_2 in zip(Gamma_1s, Gamma_2s):
            Gamma_phis.append(Gamma_2 - Gamma_1 / 2)
        self.Gamma_phis = Gamma_phis
        self.dt = dt

    def construct_c_ops(self):
        c_ops = self.construct_c_ops_sm()
        c_ops += self.construct_c_ops_sp()
        c_ops += self.construct_c_ops_sz()
        return c_ops

    def construct_c_ops_sz(self):
        c_ops = []
        dims = len(self.Gamma_1s) * [2, ]
        for idx, Gamma_phi in enumerate(self.Gamma_phis):
            sz = id_wrap_ops(sigmaz(), idx, dims)
            c_ops.append(np.sqrt(Gamma_phi / 2) * sz)
        return c_ops

    def construct_c_ops_sm(self):
        c_ops = []
        dims = len(self.Gamma_1s) * [2, ]
        for idx, Gamma_1 in enumerate(self.Gamma_1s):
            sm = id_wrap_ops(destroy(2), idx, dims)
            # factor of 2 here comes from equal contribution of up and down
            c_ops.append(np.sqrt(Gamma_1 / 2) * sm)
        return c_ops

    def construct_c_ops_sp(self):
        c_ops_sm = self.construct_c_ops_sm()
        c_ops_sp = [c_op.dag() for c_op in c_ops_sm]
        return c_ops_sp

    def H_eff_iswap(self):
        return 0.25 * (tensor(sigmax(), sigmax()) + tensor(sigmay(), sigmay()))

    def H_eff_bswap(self):
        return 0.25 * (tensor(sigmax(), sigmax()) - tensor(sigmay(), sigmay()))

    def ideal_sqiswap(self):
        time = np.pi / 2
        return (-1j * time * self.H_eff_iswap()).expm()

    def ideal_sqbswap(self):
        time = np.pi / 2
        return (-1j * time * self.H_eff_bswap()).expm()

    @staticmethod
    def gaussian(t, args=None):
        gate_time = args["gate_time"]
        sigma = args["sigma"]
        half_time = gate_time / 2
        return np.exp(-0.5 * (t - half_time) ** 2 / sigma ** 2)

    def _setup_gate_gaussian(self, gate_time, gate_type="iswap"):
        if gate_type == "iswap":
            H_eff = self.H_eff_iswap()
        elif gate_type == "bswap":
            H_eff = self.H_eff_bswap()
        else:
            raise RuntimeError("iswap and bswap only gates supported")
        sigma = gate_time / 4
        args = {"gate_time": gate_time, "sigma": sigma}
        amp = np.pi / (2 * np.sqrt(2.0 * np.pi * sigma**2) * erf(np.sqrt(2)))
        H_eff = amp * H_eff
        tlist = np.linspace(0.0, gate_time, int(gate_time / self.dt))
        return (H_eff, tlist, args)

    def run_gate_gaussian(self, gate_time, c_ops, gate_type="iswap"):
        (H_eff, tlist, args) = self._setup_gate_gaussian(gate_time, gate_type=gate_type)
        real_prop = propagator([[H_eff, self.gaussian], ], tlist, c_op_list=c_ops, args=args)
        return real_prop[-1]

    def run_gate_gaussian_crosstalk(self, gate_time, c_ops, gate_type="iswap", gate_params=None):
        (H_eff, tlist, args) = self._setup_gate_gaussian(gate_time, gate_type=gate_type)
        alpha = gate_params["alpha"]
        beta = gate_params["beta"]
        omega_a = gate_params["omega_a"]
        omega_b = gate_params["omega_b"]
        if gate_type == "iswap":
            omega_d = omega_b - omega_a
        elif gate_type == "bswap":
            omega_d = omega_b + omega_a
        XI = tensor(sigmax(), qeye(2))
        YI = tensor(sigmay(), qeye(2))
        IX = tensor(qeye(2), sigmax())
        IY = tensor(qeye(2), sigmay())

        def XI_coeff(t, args=None):
            return (alpha * self.gaussian(t, args=args)
                    * np.cos(omega_d * t)
                    * np.cos(omega_a * t)
                    )

        def YI_coeff(t, args=None):
            return (alpha * self.gaussian(t, args=args)
                    * np.cos(omega_d * t)
                    * np.sin(omega_a * t)
                    )

        def IX_coeff(t, args=None):
            return (beta * self.gaussian(t, args=args)
                    * np.cos(omega_d * t)
                    * np.cos(omega_b * t)
                    )

        def IY_coeff(t, args=None):
            return (beta * self.gaussian(t, args=args)
                    * np.cos(omega_d * t)
                    * np.sin(omega_b * t)
                    )

        H = [[H_eff, self.gaussian], [XI, XI_coeff], [YI, YI_coeff], [IX, IX_coeff], [IY, IY_coeff]]
        real_prop = propagator(H, tlist, c_op_list=c_ops, args=args)
        return real_prop[-1]

    def run_gate_gaussian_counter_single(
            self,
            gate_time,
            c_ops,
            amp=None,
            omega_d=None,
            gate_params=None,
    ):
        if omega_d is None:
            omega_d = gate_params["omega"]
        sigma = gate_time / 4
        if amp is None:
            amp = np.sqrt(2.0 * np.pi) / (gate_time * erf(np.sqrt(2)))
        omega = gate_params["omega"]
        args = {"gate_time": gate_time, "sigma": sigma}
        tlist = np.linspace(0.0, gate_time, int(gate_time / self.dt))
        phase = gate_params["phase"]

        def X_coeff(t, args=None):
            return (amp * self.gaussian(t, args=args)
                    * np.cos(omega_d * t + phase)
                    * np.cos(omega * t))

        def Y_coeff(t, args=None):
            return (amp * self.gaussian(t, args=args)
                    * np.cos(omega_d * t + phase)
                    * np.sin(omega * t))

        H = [[sigmax(), X_coeff], [sigmay(), Y_coeff]]
        options = Options(rtol=1e-10, atol=1e-10, nsteps=2000)
        real_prop = propagator(H, tlist, c_op_list=c_ops, args=args, options=options)
        return real_prop[-1]

    def run_gate_gaussian_counter(
            self,
            gate_time,
            c_ops,
            amp=None,
            omega_d=None,
            gate_type="iswap",
            gate_params=None
    ):
        XX = tensor(sigmax(), sigmax())
        XY = tensor(sigmax(), sigmay())
        YX = tensor(sigmay(), sigmax())
        YY = tensor(sigmay(), sigmay())
        omega_a = gate_params["omega_a"]
        omega_b = gate_params["omega_b"]
        phase = gate_params["phase"]
        if gate_type == "iswap" and omega_d is None:
            omega_d = omega_b - omega_a
        elif gate_type == "bswap" and omega_d is None:
            omega_d = omega_b + omega_a
        sigma = gate_time / 4
        if amp is None:
            amp = np.pi / (2 * np.sqrt(2.0 * np.pi * sigma ** 2) * erf(np.sqrt(2)))
        args = {"gate_time": gate_time, "sigma": sigma}
        tlist = np.linspace(0.0, gate_time, int(gate_time / self.dt))

        def XX_coeff(t, args=None):
            return (amp * self.gaussian(t, args=args)
                    * np.cos(omega_d * t + phase)
                    * np.cos(omega_a * t)
                    * np.cos(omega_b * t))

        def XY_coeff(t, args=None):
            return (amp * self.gaussian(t, args=args)
                    * np.cos(omega_d * t + phase)
                    * np.cos(omega_a * t)
                    * np.sin(omega_b * t))

        def YX_coeff(t, args=None):
            return (amp * self.gaussian(t, args=args)
                    * np.cos(omega_d * t + phase)
                    * np.sin(omega_a * t)
                    * np.cos(omega_b * t))

        def YY_coeff(t, args=None):
            return (amp * self.gaussian(t, args=args)
                    * np.cos(omega_d * t + phase)
                    * np.sin(omega_a * t)
                    * np.sin(omega_b * t))

        H = [[XX, XX_coeff], [YY, YY_coeff], [XY, XY_coeff], [YX, YX_coeff]]
        options = Options(rtol=1e-10, atol=1e-10, nsteps=2000)
        real_prop = propagator(H, tlist, c_op_list=c_ops, args=args, options=options)
        return real_prop[-1]

    def run_gate_rectangular(self, gate_time, c_ops, amp=None, gate_type="iswap"):
        if gate_type == "iswap":
            H_eff = self.H_eff_iswap()
        elif gate_type == "bswap":
            H_eff = self.H_eff_bswap()
        else:
            raise RuntimeError("iswap and bswap only gates supported")
        if amp is None:
            amp = np.pi / (2 * gate_time)
        tlist = np.linspace(0.0, gate_time, int(gate_time/self.dt))
        real_prop = propagator(amp * H_eff, tlist, c_op_list=c_ops)
        return real_prop[-1]

    def optimize_gate_params(self, gate_time, gate_type="iswap"):
        if gate_type == "iswap":
            init_omega_d = omega_b - omega_a
            ideal_gate = self.ideal_sqiswap()
        elif gate_type == "bswap":
            init_omega_d = omega_b + omega_a
            ideal_gate = self.ideal_sqbswap()
        else:
            raise RuntimeError("iswap and bswap only gates supported")
        sigma = gate_time / 4
        init_amp = np.pi / (2 * np.sqrt(2.0 * np.pi * sigma ** 2) * erf(np.sqrt(2)))

        def _run_counter(params):
            amp, omega_d = params
            res = self.run_gate_gaussian_counter(
                gate_time,
                [],
                amp=amp,
                omega_d=omega_d,
                gate_type=gate_type,
                gate_params={"omega_a": omega_a, "omega_b": omega_b},
            )
            return 1 - calc_fidel_4(res, ideal_gate)
        result = sp.optimize.minimize(_run_counter, x0=(init_amp, init_omega_d))
        return result, (init_amp, init_omega_d)

    @staticmethod
    def estimated_infidelity(gate_time, Gamma, dim=4):
        return (dim / (2 * (dim + 1))) * gate_time * Gamma

# if __name__=="__main__":
#     omega = 2.0 * np.pi * 0.0484  #0.0618  # #
#     gate_params = {"omega": omega, "phase": 0.0}
#     T1a = 180000.  # 100 us
#     T1b = 300000.  # 100 us
#     T2a = 250000.
#     T2b = 300000
#     Gamma_1s = [1 / T1a, 1 / T1b]
#     Gamma_2s = [1 / T2a, 1 / T2b]
#     gateserrorbudgeta = GatesErrorBudget([1 / T1a, ], [1/T2a, ])
#     gate_time = 83.3  #65.1  #
#     gate = gateserrorbudgeta.run_gate_gaussian_counter_single(
#         gate_time, [], gate_params=gate_params
#     )
#     # X/2
#     ideal_gate = Qobj((1 / np.sqrt(2)) * np.array([[1, -1j],
#                                                    [-1j, 1]]))
#     infidel = 1 - calc_fidel_2(gate, ideal_gate)
#     print(0)


if __name__ == "__main__":
    omega_a = 2.0 * np.pi * 0.0484
    omega_b = 2.0 * np.pi * 0.0618
    gate_params = {"omega_a": omega_a, "omega_b": omega_b, "phase": 0.0}
    T1a = 180000.  # 100 us
    T1b = 300000.  # 100 us
    T2a = 250000.
    T2b = 300000
    Gamma_1s = [1 / T1a, 1 / T1b]
    Gamma_2s = [1 / T2a, 1 / T2b]
    gateserrorbudget = GatesErrorBudget(Gamma_1s, Gamma_2s)
    c_ops_sz = gateserrorbudget.construct_c_ops_sz()
    c_ops_sm = gateserrorbudget.construct_c_ops_sm()
    c_ops_sp = gateserrorbudget.construct_c_ops_sp()
    c_ops_list = [c_ops_sz, c_ops_sm, c_ops_sp, c_ops_sz+c_ops_sm+c_ops_sp]
    c_ops_str = ["sz", "sm", "sp", "all"]
    sqbswap = gateserrorbudget.ideal_sqbswap()
    sqiswap = gateserrorbudget.ideal_sqiswap()
    gate_time = 101.6
    sigma = gate_time / 4
    amp = np.pi / (2 * np.sqrt(2.0 * np.pi * sigma ** 2) * erf(np.sqrt(2)))
    gate = gateserrorbudget.run_gate_gaussian_counter(gate_time, [], gate_type="bswap", gate_params=gate_params,
                                                      amp=amp)
    infidel = 1 - calc_fidel_4(gate, sqbswap)
    print("beyond RWA", infidel)
    gate_params = {"omega_a": omega_a, "omega_b": omega_b,
                   "alpha": 2.0 * np.pi * 0.001 * 2.0 * np.pi * 0.29 * 3,
                   "beta": 2.0 * np.pi * 0.001 * 2.0 * np.pi * 0.29 * 3, "phase": 0.0}
    gate = gateserrorbudget.run_gate_gaussian_crosstalk(gate_time, [], gate_type="bswap", gate_params=gate_params)
    infidel = 1 - calc_fidel_4(gate, sqbswap)
    # sqiswap
    gate_time = 257.8
    gate = gateserrorbudget.run_gate_gaussian_crosstalk(gate_time, [], gate_type="iswap", gate_params=gate_params)
    infidel = 1 - calc_fidel_4(gate, sqiswap)
    print("crosstalk", infidel)
    # #opt_result, (init_amp, init_omega_d) = gateserrorbudget.optimize_gate_params(gate_time, gate_type="bswap")
    # for idx, c_ops in enumerate(c_ops_list):
    #     gate = gateserrorbudget.run_gate_gaussian(gate_time, c_ops, gate_type="bswap")
    #     chi_real = my_to_chi(gate)
    #     infidel = 1 - calc_fidel_chi(chi_real, my_to_chi(sqbswap))
    #     print(c_ops_str[idx], infidel)
    # # beyond RWA contrs
    # gate = gateserrorbudget.run_gate_gaussian_counter(gate_time, [], gate_type="bswap", gate_params=gate_params)
    # infidel = 1 - calc_fidel_4(gate, sqbswap)
    # print("beyond RWA", infidel)
    gate_time = 257.0
    #opt_result, (init_amp, init_omega_d) = gateserrorbudget.optimize_gate_params(gate_time, gate_type="iswap")
    # for idx, c_ops in enumerate(c_ops_list):
    #     gate = gateserrorbudget.run_gate_gaussian(gate_time, c_ops, gate_type="iswap")
    #     chi_real = my_to_chi(gate)
    #     infidel = 1 - calc_fidel_chi(chi_real, my_to_chi(sqiswap))
    #     print(c_ops_str[idx], infidel)
    sigma = gate_time / 4
    amp = np.pi / (2 * np.sqrt(2.0 * np.pi * sigma ** 2) * erf(np.sqrt(2)))
    gate = gateserrorbudget.run_gate_gaussian_counter(gate_time, [], gate_type="iswap", gate_params=gate_params,
                                                      amp=amp)
    # testgate = gate.data.toarray()
    # testgatedims = gate.dims
    # testgate[0, 0] = testgate[3, 3] = 1.0
    # testgate[0, 3] = testgate[3, 0] = 0.0
    # gate = Qobj(testgate, dims=testgatedims)
    infidel = 1 - calc_fidel_4(gate, sqiswap)
    print("beyond RWA", infidel)
    print(0.0)
