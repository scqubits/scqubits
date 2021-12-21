import cmath

import numpy as np
from qutip import qeye, sigmax, sigmaz, tensor, Qobj, propagator
from scipy.interpolate import interp1d

import scqubits.io_utils.fileio_serializers as serializers


sqrtiSWAP = np.zeros((4, 4), dtype=complex)
sqrtiSWAP[0, 0] = sqrtiSWAP[3, 3] = 1.0
sqrtiSWAP[1, 2] = sqrtiSWAP[2, 1] = -1j / np.sqrt(2.0)
sqrtiSWAP[1, 1] = sqrtiSWAP[2, 2] = 1.0 / np.sqrt(2.0)
sqrtiSWAP_Q = Qobj(sqrtiSWAP, dims=[[2, 2], [2, 2]])

ZA = tensor(sigmaz(), qeye(2))
ZB = tensor(qeye(2), sigmaz())
XA = tensor(sigmax(), qeye(2))
XB = tensor(qeye(2), sigmax())
XX = tensor(sigmax(), sigmax())
II = tensor(qeye(2), qeye(2))


class PulseConstruction(serializers.Serializable):
    def __init__(self, H0, HXI, HIX, HXX, control_dt=0.05):
        self.H0 = H0
        self.norm_HXI = np.max(np.abs(HXI))
        self.norm_HIX = np.max(np.abs(HIX))
        self.norm_HXX = np.max(np.abs(HXX))
        self.HXI = HXI / self.norm_HXI
        self.HIX = HIX / self.norm_HIX
        self.HXX = HXX / self.norm_HXX
        self.control_dt = control_dt

    def RZ(self, theta, Z_):
        return (-1j * theta * Z_ / 2.0).expm()

    def fix_with_single_q_gates(self, gate_):
        alpha = cmath.phase(gate_[0, 0])
        beta = cmath.phase(gate_[1, 1])
        gamma = cmath.phase(gate_[1, 2])
        return np.array(
            [alpha + beta, alpha - gamma - np.pi / 2, -beta + gamma + np.pi / 2]
        )

    def multiply_with_single_q_gates(self, gate):
        (t1, t2, t3) = self.fix_with_single_q_gates(gate)
        return (
            self.RZ(t1, ZA) * self.RZ(t2, ZB) * gate * self.RZ(0, ZA) * self.RZ(t3, ZB)
        )

    def time_from_angle(self, angle, freq):
        return (-angle % (4.0 * np.pi)) / (2.0 * np.pi * freq)

    def calc_fidel_4(self, prop, gate):
        prop = Qobj(prop[0:4, 0:4], dims=[[2, 2], [2, 2]])
        return (
            np.trace(prop.dag() * prop) + np.abs(np.trace(prop.dag() * gate)) ** 2
        ) / 20

    def calc_fidel_2(self, prop, gate):
        prop = Qobj(prop[0:2, 0:2])
        return (
            np.trace(prop.dag() * prop) + np.abs(np.trace(prop.dag() * gate)) ** 2
        ) / 6

    def get_controls_only_sine(self, freq, amp):
        sintime = 1.0 / freq
        sin_eval_times = np.linspace(0.0, sintime, int(sintime / self.control_dt) + 1)
        sin_pulse = amp * np.sin(2.0 * np.pi * freq * sin_eval_times)
        return sin_pulse, sin_eval_times

    def amp_from_freq_id(self, freq):
        return 2.0 * np.pi * freq * 2.4048 / 2.0

    def amp_from_freq_sqrtiswap(self, omega, omega_d):
        return (
            0.125
            * np.pi
            * (omega_d ** 2 - omega ** 2)
            / (omega_d * np.sin(np.pi * omega / omega_d))
        )

    def run_sqrt_iswap(self, freq, amp):
        controls, times = self.get_controls_only_sine(freq, amp)
        control_spline_c = interp1d(times, controls, fill_value="extrapolate")
        H = [self.H0, [self.HXX, lambda t, a: control_spline_c(t)]]
        prop = propagator(H, times)
        return prop[-1], controls, times

    def synchronize(self, ta, tb, max_freq=0.25, min_freq=0.125):
        if ta <= tb:
            return self._synchronize(ta, tb, max_freq, min_freq)
        else:
            output, _ = self._synchronize(tb, ta, max_freq, min_freq)
            flipped_output = np.flip(output, axis=1)
            return (flipped_output, (ta, tb))

    def _synchronize(self, ta, tb, max_freq, min_freq):
        """Assume ta <= tb"""
        tmax = 1.0 / max_freq
        tmin = 1.0 / min_freq
        if ta == tb == 0.0:
            return np.array([(None, None)]), (ta, tb)
        elif (1.0 / max_freq) <= (tb - ta) <= (1.0 / min_freq):
            return np.array([(1.0 / (tb - ta), None)]), (ta, tb)
        elif (tb - ta) < (1.0 / max_freq):
            new_freq = (tb - ta + tmax) ** (-1)
            return np.array([(new_freq, max_freq)]), (ta, tb)
        else:  # (tb - ta) > 1. / min_freq
            tavg = (tmin + tmax) / 2.0
            freq_avg = 1.0 / tavg
            n, r = divmod(tb - ta, tavg)
            return np.array(int(n) * ((freq_avg, None),) + ((1.0 / r, None),)), (ta, tb)

    def _concatenate_for_qubit(self, freq, total_pulse, total_times):
        amp = self.amp_from_freq_id(freq)
        controls, times = self.get_controls_only_sine(freq, amp)
        total_pulse = self.concatenate_times_or_controls(
            (total_pulse, controls), self.concatenate_two_controls
        )
        total_times = self.concatenate_times_or_controls(
            (total_times, times), self.concatenate_two_times
        )
        return total_pulse, total_times

    def parse_synchronize(self, synchronize_output):
        total_pulse_a = np.array([])
        total_pulse_b = np.array([])
        total_times_a = np.array([])
        total_times_b = np.array([])
        output, (t_a, t_b) = synchronize_output
        for (freq_a_, freq_b_) in output:
            if freq_a_ is not None:
                total_pulse_a, total_times_a = self._concatenate_for_qubit(
                    freq_a_, total_pulse_a, total_times_a
                )
            if freq_b_ is not None:
                total_pulse_b, total_times_b = self._concatenate_for_qubit(
                    freq_b_, total_pulse_b, total_times_b
                )
        # here we add the delay part that actually gives us Z rotations
        delay_time_a = np.linspace(0.0, t_a, int(t_a / self.control_dt) + 1)
        delay_time_b = np.linspace(0.0, t_b, int(t_b / self.control_dt) + 1)
        total_times_a = self.concatenate_times_or_controls(
            (total_times_a, delay_time_a), self.concatenate_two_times
        )
        total_times_b = self.concatenate_times_or_controls(
            (total_times_b, delay_time_b), self.concatenate_two_times
        )
        total_pulse_a = self.concatenate_times_or_controls(
            (total_pulse_a, np.zeros_like(delay_time_a)), self.concatenate_two_controls
        )
        total_pulse_b = self.concatenate_times_or_controls(
            (total_pulse_b, np.zeros_like(delay_time_b)), self.concatenate_two_controls
        )
        return total_pulse_a, total_times_a, total_pulse_b, total_times_b

    def concatenate_times_or_controls(self, t_c_tuple, concatenator):
        if len(t_c_tuple) == 1:
            return t_c_tuple[0]
        concat_first_two = concatenator(t_c_tuple[0], t_c_tuple[1])
        if len(t_c_tuple) == 2:
            return concat_first_two
        return concatenator((concat_first_two,) + t_c_tuple[2:])

    def concatenate_two_times(self, times_1, times_2):
        if times_1.size == 0:
            return times_2
        if times_2.size == 0:
            return times_1
        return np.concatenate((times_1, times_1[-1] + times_2[1:]))

    def concatenate_two_controls(self, controls_1, controls_2):
        if controls_1.size == 0:
            return controls_2
        if controls_2.size == 0:
            return controls_1
        assert np.allclose(controls_1[-1], 0.0) and np.allclose(controls_2[-1], 0.0)
        return np.concatenate((controls_1, controls_2[1:]))
