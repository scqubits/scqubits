import cmath

import numpy as np
import scipy as sp
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
        self.omega_a = np.real(H0[2, 2] - H0[0, 0])
        self.omega_b = np.real(H0[1, 1] - H0[0, 0])
        self.norm_HXI = HXI[0, 2]
        self.norm_HIX = HIX[0, 1]
        self.norm_HXX = HXX[0, 3]
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

    def time_from_angle(self, angle, omega):
        return (-angle % (4.0 * np.pi)) / omega

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

    def get_controls_only_sine(self, omega, amp):
        sintime = 2.0 * np.pi / omega
        sin_eval_times = np.linspace(0.0, sintime, int(sintime / self.control_dt) + 1)
        sin_pulse = amp * np.sin(omega * sin_eval_times)
        return sin_pulse, sin_eval_times

    def amp_from_omega_id(self, omega, bessel_zero_num=1):
        bessel_zero_val = sp.special.jn_zeros(0, bessel_zero_num)[0]
        return omega * bessel_zero_val / 2.0

    def amp_from_omega_sqrtiswap(self, omega, omega_d):
        return (
            0.125
            * np.pi
            * (omega_d ** 2 - omega ** 2)
            / (omega_d * np.sin(np.pi * omega / omega_d))
        )

    def run_sqrt_iswap(self, omega, amp):
        controls, times = self.get_controls_only_sine(omega, amp)
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
        amp = self.amp_from_omega_id(2.0 * np.pi * freq)
        controls, times = self.get_controls_only_sine(2.0 * np.pi * freq, amp)
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
        concat_first_two = concatenator((t_c_tuple[0], t_c_tuple[1]))
        if len(t_c_tuple) == 2:
            return concat_first_two
        return concatenator((concat_first_two,) + t_c_tuple[2:])

    def concatenate_two_times(self, times_1_2):
        times_1, times_2 = times_1_2
        if times_1.size == 0:
            return times_2
        if times_2.size == 0:
            return times_1
        return np.concatenate((times_1, times_1[-1] + times_2[1:]))

    def concatenate_two_controls(self, controls_1_2):
        controls_1, controls_2 = controls_1_2
        if controls_1.size == 0:
            return controls_2
        if controls_2.size == 0:
            return controls_1
        assert np.allclose(controls_1[-1], 0.0) and np.allclose(controls_2[-1], 0.0)
        return np.concatenate((controls_1, controls_2[1:]))

    def full_pulses(self, freq_sum_divisor=3):
        omega_drive = (self.omega_a + self.omega_b) / freq_sum_divisor
        omega = self.H0[2, 2] - self.H0[1, 1]
        amp = self.amp_from_omega_sqrtiswap(omega, omega_drive)
        prop, controls_2q, control_eval_times_2q = self.run_sqrt_iswap(omega_drive, amp)
        omega_array = np.array([self.omega_a, self.omega_b, self.omega_b])
        times = self.time_from_angle(self.fix_with_single_q_gates(prop), omega_array)
        parsed_output_after = self.parse_synchronize(self.synchronize(times[0], times[1]))
        parsed_output_before = self.parse_synchronize(self.synchronize(0.0, times[2]))
        after_pulse_a, after_times_a, after_pulse_b, after_times_b = parsed_output_after
        before_pulse_a, before_times_a, before_pulse_b, before_times_b = parsed_output_before
        total_pulse_a = self.concatenate_times_or_controls((before_pulse_a, np.zeros_like(controls_2q),
                                                            after_pulse_a), self.concatenate_two_controls)
        total_pulse_b = self.concatenate_times_or_controls((before_pulse_b, np.zeros_like(controls_2q),
                                                            after_pulse_b), self.concatenate_two_controls)
        total_pulse_c = self.concatenate_times_or_controls((np.zeros_like(before_pulse_a), controls_2q,
                                                            np.zeros_like(after_pulse_a)),
                                                           self.concatenate_two_controls)
        total_times_a = self.concatenate_times_or_controls((before_times_a, np.zeros_like(controls_2q),
                                                            after_times_a), self.concatenate_two_times)
        total_times_b = self.concatenate_times_or_controls((before_times_b, np.zeros_like(controls_2q),
                                                            after_times_b), self.concatenate_two_times)
        total_times_c = self.concatenate_times_or_controls((np.zeros_like(before_times_a), control_eval_times_2q,
                                                            np.zeros_like(after_times_a)),
                                                           self.concatenate_two_times)
        spline_a = interp1d(total_times_a, total_pulse_a)
        spline_b = interp1d(total_times_b, total_pulse_b)
        spline_c = interp1d(total_times_c, total_pulse_c)
        return spline_a, spline_b, spline_c, total_times_a
