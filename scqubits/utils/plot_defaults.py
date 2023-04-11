# plot_defaults.py
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


from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

import scqubits.core.constants as constants
import scqubits.core.units as units

if TYPE_CHECKING:
    from scqubits.core.storage import DataStore, SpectrumData, WaveFunction


NAME_REPLACEMENTS = {
    "ng": r"$n_g$",
    "ng1": r"$n_{g1}$",
    "ng2": r"$n_{g2}$",
    "EJ": r"$E_J$",
    "EJ1": r"$E_{J1}$",
    "EJ2": r"$E_{J2}$",
    "EJ3": r"$E_{J3}$",
    "EC": r"$E_C$",
    "EL": r"$E_L$",
    "flux": r"$\Phi_{ext}/\Phi_0$",
}


def recast_name(raw_name: Union[str, None]) -> str:
    if raw_name in NAME_REPLACEMENTS:
        return NAME_REPLACEMENTS[raw_name]
    return raw_name or ""


def set_wavefunction_scaling(
    wavefunctions: "List[WaveFunction]",
    potential_vals: np.ndarray,
) -> float:
    """
    Sets the scaling parameter for 1d wavefunctions

    Returns
    -------
      scaling factor
    """
    # Do not attempt to scale down amplitudes to very small energy spacings, i.e. if
    # energy spacing is smaller than y_range * Y_RANGE_THRESHOLD_FRACTION, then do
    # not apply additional downscaling
    Y_RANGE_THRESHOLD_FRACTION = 1 / 12

    # If energy spacing is used for scaling, fill no more than this fraction of the
    # spacing.
    FILLING_FRACTION = 0.9

    # Largest allowed wavefunction amplitude range as fraction of y_range.
    MAX_AMPLITUDE_FRACTION = 1 / 7

    # Amplitude threshold for applying any scaling at all. Note that the imaginary
    # part of a wavefunction may be nominally 0; do not scale up in that case.
    PRECISION_THRESHOLD = 1e-6

    wavefunc_count = len(wavefunctions)
    energies = [wavefunc.energy for wavefunc in wavefunctions]

    e_max = np.max(energies)
    e_min = np.min(energies)
    e_range = e_max - e_min
    y_min = np.min(potential_vals)  # lowest value of potential energy
    y_max = e_max + 0.3 * e_range  # maximum eigenenergy plus padding
    y_range = y_max - y_min

    amplitudes = np.asarray([wavefunc.amplitudes for wavefunc in wavefunctions])

    def amplitude_mins() -> np.ndarray:
        return np.apply_along_axis(func1d=np.min, axis=1, arr=amplitudes)

    def amplitude_maxs() -> np.ndarray:
        return np.apply_along_axis(func1d=np.max, axis=1, arr=amplitudes)

    def max_amplitude_range() -> float:
        return np.max(amplitude_maxs() - amplitude_mins())

    if (
        max_amplitude_range() < PRECISION_THRESHOLD
    ):  # amplitude likely just zero (e.g., mode='imag'); do not scale up
        return 1
    else:
        scale_factor = (
            y_range * MAX_AMPLITUDE_FRACTION / max_amplitude_range()
        )  # set amplitudes to largest acceptable
        amplitudes *= scale_factor

        if wavefunc_count == 1:
            return scale_factor

        amplitude_fillings = np.pad(np.abs(amplitude_mins()), [0, 1]) + np.pad(
            np.abs(amplitude_maxs()), [1, 0]
        )
        amplitude_fillings = amplitude_fillings[1:-1]

        energy_spacings = np.pad(energies, [0, 1]) - np.pad(energies, [1, 0])
        energy_spacings = energy_spacings[1:-1]

        for energy_gap, amplitude_filling in zip(energy_spacings, amplitude_fillings):
            if energy_gap > y_range * Y_RANGE_THRESHOLD_FRACTION:
                if amplitude_filling > energy_gap * FILLING_FRACTION:
                    scale_factor *= energy_gap * FILLING_FRACTION / amplitude_filling
                    amplitudes *= energy_gap * FILLING_FRACTION / amplitude_filling
                    amplitude_fillings *= (
                        energy_gap * FILLING_FRACTION / amplitude_filling
                    )
        return scale_factor


def wavefunction1d_discrete(mode: Optional[str] = None) -> Dict[str, Any]:
    """Plot defaults for plotting.wavefunction1d_discrete.

    Parameters
    ----------
    mode:
        amplitude modifier, needed to give the correct default y label"""
    ylabel = r"$\psi_j(n)$"
    if mode:
        ylabel = constants.MODE_STR_DICT[mode](ylabel)
    return {"xlabel": "n", "ylabel": ylabel}


def wavefunction2d() -> Dict[str, Any]:
    """Plot defaults for plotting.wavefunction2d"""
    return {"figsize": (8, 3)}


def contours(
    x_vals: Union[List[float], np.ndarray], y_vals: Union[List[float], np.ndarray]
) -> Dict[str, Any]:
    """Plot defaults for plotting.contours"""
    aspect_ratio = (y_vals[-1] - y_vals[0]) / (x_vals[-1] - x_vals[0])
    figsize = (8, 8 * aspect_ratio)
    return {"figsize": figsize}


def matrix() -> Dict[str, Any]:
    """Plot defaults for plotting.matrix"""
    return {"figsize": (10, 5)}


def evals_vs_paramvals(specdata: "SpectrumData", **kwargs) -> Dict[str, Any]:
    """Plot defaults for plotting.evals_vs_paramvals"""
    kwargs["xlabel"] = kwargs.get("xlabel") or recast_name(specdata.param_name)
    kwargs["ylabel"] = kwargs.get("ylabel") or "energy [{}]".format(units.get_units())
    return kwargs


def matelem_vs_paramvals(
    specdata: Union["SpectrumData", "DataStore"]
) -> Dict[str, Any]:
    """Plot defaults for plotting.matelem_vs_paramvals"""
    return {"xlabel": recast_name(specdata.param_name), "ylabel": "matrix element"}


def chi(param_name: Union[str, None], **kwargs) -> Dict[str, Any]:
    """Plot defaults for sweep_plotting.chi"""
    kwargs["xlabel"] = kwargs.get("xlabel") or recast_name(param_name)
    kwargs["ylabel"] = kwargs.get("ylabel") or r"$\chi_j$ [{}]".format(
        units.get_units()
    )
    return kwargs


def chi01(param_name: Union[str, None], yval: float, **kwargs) -> Dict[str, Any]:
    """Plot defaults for sweep_plotting.chi01"""
    kwargs["xlabel"] = kwargs.get("xlabel") or recast_name(param_name)
    kwargs["ylabel"] = kwargs.get("ylabel") or r"$\chi_{{01}}$ [{}]".format(
        units.get_units()
    )
    kwargs["title"] = kwargs.get("title") or r"$\chi_{{01}}=${:.4f} {}".format(
        yval, units.get_units()
    )
    return kwargs


def charge_matrixelem(param_name: str, **kwargs) -> Dict[str, Any]:
    """Plot defaults for sweep_plotting.charge_matrixelem"""
    kwargs["xlabel"] = kwargs.get("xlabel") or recast_name(param_name)
    kwargs["ylabel"] = kwargs.get("ylabel") or r"$|\langle i |n| j \rangle|$"
    return kwargs


# supported keyword arguments for plotting and sweep_plotting functions
SPECIAL_PLOT_OPTIONS = [
    "fig_ax",
    "figsize",
    "filename",
    "grid",
    "x_range",
    "y_range",
    "ymax",
]
