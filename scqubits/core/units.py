# units.py
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


import warnings

from typing import List

# Currently set units, referred to elsewhere as "system units" (must be one of the units
# in `_supported_units`) Often, system units need to be converted to "standard
# units", which are considered to be `[Hz]` or `2pi/[s]`
_current_units = "GHz"

# Units that we currently support
_supported_units = ["GHz", "MHz", "kHz", "Hz"]

# Numerical factor between a given unit and Hz
_units_factor = {"GHz": 1e9, "MHz": 1e6, "kHz": 1e3, "Hz": 1.0}

# labels for time units obtained from 1/frequency units
_units_time_labels = {"GHz": r"$ns$", "MHz": r"$\mu s$", "kHz": r"$ms$", "Hz": r"$s$"}


def get_units() -> str:
    """Get system units."""
    return _current_units


def set_units(units: str) -> str:
    """Set system units."""
    # Importing here avoids a cyclic import problem.
    from scqubits.core.qubit_base import QuantumSystem

    # Show a warning if we are changing units after some `QuantumSystems`
    # may have been instantiated.
    if QuantumSystem._quantumsystem_counter > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "Changing units (by calling set_units()) after initializing qubit"
                " instances is likely to cause unintended inconsistencies.",
                UserWarning,
            )

    if units not in _supported_units:
        raise ValueError(
            "Unsupported system units given. Must be one of: {}".format(
                str(_supported_units)
            )
        )

    global _current_units
    _current_units = units
    return units


def get_units_time_label(units: str = None) -> str:
    """Get a latex representation of 1/units"""
    units = units or _current_units
    if units not in _supported_units:
        raise ValueError(
            "Unsupported system units given. Must be one of: {}".format(
                str(_supported_units)
            )
        )

    return _units_time_labels[units]


def show_supported_units() -> List[str]:
    """Returns a list of supported system units."""
    return _supported_units


def to_standard_units(value: float) -> float:
    r"""
    Converts `value` (a frequency or angular frequency) from currently set system units,
    to standard units (Hz or  2pi/s).

    Parameters
    ----------
    value: float
        a frequency or angular frequency assumed to be in system units.

    Returns
    -------
    float:
        frequency or angular frequency converted to Hz or 2pi/s
    """
    return value * _units_factor[_current_units]


def from_standard_units(value: float) -> float:
    r"""
    Converts `value` (a frequency or angular frequency) from standard units
    (`[Hz]` or  `2\pi / [s]`) to currently set system units.

    Parameters
    ----------
    value: float
        a frequency or angular frequency assumed to be in standard units
        (`[Hz]` or  `2\pi / [s]`)
    Returns
    -------
    float:
        frequency or angular frequency converted to system units

    """
    return value / _units_factor[_current_units]


def units_scale_factor(units: str = None) -> float:
    """
    Return a numerical scaling factor that converts form Hz to `units`.
    (given as argument or, by default, stored in  `_current_units`) .
    """
    units = _current_units if units is None else units

    if units not in _supported_units:
        raise ValueError(
            "Unsupported units given. Must be one of: {}".format(str(_supported_units))
        )
    return _units_factor[units]
