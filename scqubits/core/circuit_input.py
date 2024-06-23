import pyparsing as pp
import os

from typing import List, Tuple
from scqubits.utils.misc import is_string_float
from pyparsing import Group, Opt, Or, Literal, Suppress
import numpy as np
import scipy as sp
import sympy as sm

# *****************************************************************
#  OUR GRAMMAR DEFINITIONS
# *****************************************************************
#
# Pattern for branch definitions in yaml file:
#      - [ <branch_type>, <node1>, <node2>, <param>, <aux params> ]       or
#      - [ <branch_type>, <node1>, <node2>, <param1>, <param2> ]
#
# where <param>:   <symbol> = <number>
#                  <number>
#                  <symbol>
#
# The last option is valid only if <symbol> has previously been
# assigned. Optionally, <number> may be grouped with valid physical
# unit.


# - Ignore in parsing ********************************************
# Mark the following as characters / character combinations not
# to be recorded in the parsed result. I.e., the grammar may expect
# these in various places, but they are not carried forward into
# the parsed result
BEG = Suppress(Literal("-") + Literal("["))
END = Suppress("]")
CM = Suppress(",")
QM = Opt(Suppress('"'))  # optional quotation mark: may use JJ, or "JJ"


# - Numbers ******************************************************
INT = pp.common.integer  # unsigned integer
NUM = pp.common.fnumber  # float


# - Branch types ***************************************************
branch_type_names = ["C", "L"]

# build up dictionary of branch types
# allow, for example, both "JJ" as well as just JJ
BRANCH_TYPES = {name: QM + name + QM for name in branch_type_names}
for BTYPE in BRANCH_TYPES.values():
    BTYPE.set_results_name("branch_type")

JJ_ORDER = pp.Word(pp.nums).add_condition(
    lambda tokens: int(tokens[0]) > 0, message="Junction order must be greater than 0."
)
BRANCH_TYPES["JJ"] = (
    QM + pp.Combine(pp.Word("JJ") + Opt(JJ_ORDER) + Opt(pp.Word("s"))) + QM
)  # defining grammar to find "JJi" where i is an optional natural number


# - Units: prefixes etc. **************************************************
prefix_dict = {
    "T": 1e12,  # Tera
    "G": 1e9,  # Giga
    "M": 1e6,  # Mega
    "k": 1e3,  # kilo
    "m": 1e-3,  # milli
    "u": 1e-6,  # micro
    "n": 1e-9,  # nano
    "p": 1e-12,  # pico
    "f": 1e-15,  # femto
}
PREFIX = pp.Char(list(prefix_dict.keys()))

energy_names = ["EJ", "EC", "EL"]


UNITS_FREQ_ENERGY = Literal("Hz") ^ Literal("J") ^ Literal("eV")

UNITS = {name: Opt(PREFIX, None) for name in energy_names}
UNITS["EJ"] += UNITS_FREQ_ENERGY ^ Literal("A") ^ Literal("H")  # Ampere, Henry
UNITS["EC"] += UNITS_FREQ_ENERGY ^ Literal("F")  # Farad
UNITS["EL"] += UNITS_FREQ_ENERGY ^ Literal("H")  # Henry
for name, unit in UNITS.items():
    unit.leave_whitespace()  # allow only "kHz", not "k Hz"
    unit.set_name(f"{name}_UNITS")

# - Parameter specifications --------------------------------
SYMBOL = pp.common.identifier
VALUES = {name: NUM + Opt(" ") + Opt(UNITS[name], None) for name in energy_names}
ASSIGNS = {
    name: SYMBOL + Suppress(Literal("=")) + VALUES[name] for name in energy_names
}

PARAMS = {
    name: Or(
        [
            Group(ASSIGNS[name])("ASSIGN"),
            Group(SYMBOL)("SYMBOL"),
            Group(VALUES[name])("VALUE"),
        ]
    )
    for name in energy_names
}  # can specify in three ways

# # - Branch specifications ------------------------------------------------------
aux_val = pp.Word(
    pp.printables.replace("]", "").replace("[", "") + " "
)  # allowing for numerical expressions in auxiliary params
AUX_PARAM = Group(pp.ZeroOrMore(CM + SYMBOL + Suppress(Literal("=")) + aux_val))(
    "AUX_PARAM"
)

order_count = pp.Empty()


def find_jj_order(str_result: str, location: int, tokens: pp.ParseResults):
    from scqubits.core.circuit_utils import _junction_order

    JJ_TYPE = BEG + BRANCH_TYPES["JJ"]
    JJ_TYPE.add_parse_action(lambda tokens: _junction_order(tokens[0]))
    return JJ_TYPE.parse_string(str_result)


order_count.set_parse_action(find_jj_order)

BRANCH_JJ = (
    BEG
    + BRANCH_TYPES["JJ"]("BRANCH_TYPE")
    + CM
    + INT("node1")
    + CM
    + INT("node2")
    + CM
    + pp.counted_array(PARAMS["EJ"] + CM, int_expr=order_count)("EJ_VALS")
    + PARAMS["EC"]("EC")
    + AUX_PARAM
    + END
)

BRANCH_C = (
    BEG
    + BRANCH_TYPES["C"]("BRANCH_TYPE")
    + CM
    + INT("node1")
    + CM
    + INT("node2")
    + CM
    + PARAMS["EC"]("EC")
    + AUX_PARAM
    + END
)

BRANCH_L = (
    BEG
    + BRANCH_TYPES["L"]("BRANCH_TYPE")
    + CM
    + INT("node1")
    + CM
    + INT("node2")
    + CM
    + PARAMS["EL"]("EL")
    + AUX_PARAM
    + END
)

BRANCHES = Or([BRANCH_JJ, BRANCH_C, BRANCH_L])

# uncomment to create a html describing the grammar of this language
# BRANCHES.create_diagram("branches.html")


# - For filtering out only the code specifying the branches -----------------
def remove_comments(code: str) -> str:
    return pp.pythonStyleComment.suppress().transformString(code)


def remove_branchline(code: str) -> str:
    return Suppress(Literal("branches") + ":").transformString(code)


def strip_empty_lines(code: str) -> str:
    return os.linesep.join(
        [line.lstrip() for line in code.splitlines() if line.lstrip()]
    )


pp.autoname_elements()


# - Parsing and processing ParsedResults data ------------------------------
def parse_code_line(code_line: str, _branch_count):
    """

    Args:
        code_line (str): string describing the branch from the input file
        _branch_count (_type_): the count of the branch in a given circuit

    Returns:
        branch_type: str
        node_idx1: int
        node_idx2: int
        params: str
        aux_params: str
        _branch_count: int
    """
    pp_result = BRANCHES.parse_string(code_line)
    branch_type = pp_result.BRANCH_TYPE[0]

    branch_type, node_idx1, node_idx2, *params, aux_params = pp_result

    return branch_type, node_idx1, node_idx2, params, aux_params, _branch_count


def convert_value_to_GHz(val, units):
    """
    Converts a given value and units to energy in GHz. The units are given in a string in the format "pU"
    where p is an optional multiplier prefix and U is units. For example: "pH", "nA", "fF", "eV"

    Args:
        val (float): value in given units
        units (str): units described in a string with an optional multiplier prefix

    Raises:
        ValueError: If the unit is unknown.

    Returns:
        float: Energy in GHz
    """
    # all the possible units
    # capacitance F, inductance H, current A, energy J, frequency Hz, eV
    # returns value in GHz
    if units is None:
        return val  # default is GHz

    prefix = 1 if units[0] is None else prefix_dict[units[0]]
    val *= prefix
    unit_str = units[1]

    h = sp.constants.h
    e = sp.constants.e
    Φ0 = sp.constants.h / (2 * e)
    if unit_str == "Hz":
        return val * 1e-9
    elif unit_str == "J":
        return val / h * 1e-9
    elif unit_str == "eV":
        return val * 1.602176634e-19 / h * 1e-9
    elif unit_str == "F":
        return e**2 / (2 * val * h) * 1e-9
    elif unit_str == "H":
        return Φ0**2 / (val * h * (2 * np.pi) ** 2) * 1e-9
    elif unit_str == "A":
        return val * Φ0 / (2 * np.pi * h) * 1e-9
    else:
        raise ValueError(f"Unknown unit {unit_str}")


def process_param(pattern):
    """
    Returns a tuple containing (symbol, value) given a pattern as detected by pyparsing.
    Either the symbol or the value can be returned to be none, when the symbol is already assigned or no symbol is assigned to the given branch parameter.
    """
    name = pattern.getName()
    if name == "ASSIGN":
        sym = sm.symbols(pattern[0])
        val = pattern[1]
        units = None if pattern[-1] == None else pattern[-2:]
        val_converted = convert_value_to_GHz(val, units)
        return sym, val_converted
    if name == "SYMBOL":
        return sm.symbols(pattern[0]), None
    if name == "VALUE":
        units = None if pattern[-1] == None else pattern[-2:]
        converted_val = convert_value_to_GHz(pattern[0], units)
        return None, converted_val
    if name == "AUX_PARAM":
        num_of_aux_params = int(len(pattern) / 2)
        aux_params = {}
        for idx in range(num_of_aux_params):
            aux_params[pattern[2 * idx]] = (
                float(pattern[2 * idx + 1])
                if is_string_float(pattern[2 * idx + 1])
                else pattern[2 * idx + 1]
            )
        return aux_params
