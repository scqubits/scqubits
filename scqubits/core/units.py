# units.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

# Currently set units (must be one of the units in `_supported_units`
_current_units = 'GHz'

# Units that we currently support
_supported_units = ['GHz', 'MHz', 'kHz', 'Hz']

# Numerical factor between a given unit and Hz
_units_factor = {'GHz': 1e9, 'MHz': 1e6, 'kHz': 1e3, 'Hz': 1, }


def get_units():
    """Get current units.
    """
    return _current_units


def set_units(units):
    """Set current units.
    """
    global _current_units

    if units not in _supported_units:
        raise ValueError("Unsupported units given. Must be one of: {}".format(str(_supported_units)))
    else:
        _current_units = units

    return units


def show_supported_units():
    """Returns a list of supported units.
    """
    return _supported_units_list


# TODO must be a better name for this?!
def units_scale_factor(units=None):
    """
    Return a numerical scaling factor that converts form Hz to `units` 
    (given as argument or, by default, stored in  `_current_units`) .
    """
    global _current_units
    units = _current_units if units is None else units

    if units not in _supported_units:
        raise ValueError("Unsupported units given. Must be one of: {}".format(str(_supported_units)))
    return _units_factor[units]


