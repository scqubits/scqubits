# spectrum_utils.py
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

import cmath

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import qutip as qt

from numpy import ndarray
from qutip import Qobj
from scipy.sparse import csc_matrix, dia_matrix

if TYPE_CHECKING:
    from scqubits import Oscillator, ParameterSweep, SpectrumData
    from scqubits.core.qubit_base import QubitBaseClass
    from scqubits.io_utils.fileio_qutip import QutipEigenstates

from scqubits.utils.typedefs import QuantumSys


def order_eigensystem(
    evals: np.ndarray, evecs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Takes eigenvalues and corresponding eigenvectors and orders them (in place)
    according to the eigenvalues (from smallest to largest; real valued eigenvalues
    are assumed). Compare http://stackoverflow.com/questions/22806398.

    Parameters
    ----------
    evals:
        array of eigenvalues
    evecs:
        array containing eigenvectors; evecs[:, 0] is the first eigenvector etc.
    """
    ordered_evals_indices = evals.argsort()  # sort manually
    evals[:] = evals[ordered_evals_indices]
    evecs[:] = evecs[:, ordered_evals_indices]
    return evals, evecs


def extract_phase(complex_array: np.ndarray, position: Optional[int] = None) -> float:
    """Extracts global phase from `complex_array` at given `position`. If position is
    not specified, the `position` is set as follows. Find the maximum between the
    leftmost point and the halfway point of the wavefunction. The position of that
    point is used to determine the phase factor to be eliminated.

    Parameters
    ----------
    complex_array:
        complex-valued array
    position:
        position where the phase is extracted (default value = None)
    """
    if position is None:
        halfway_position = len(complex_array) // 2
        flattened_position = np.argmax(
            np.abs(complex_array[:halfway_position])
        )  # extract phase from element with largest amplitude modulus
        position = np.unravel_index(flattened_position, complex_array.shape)
    return cmath.phase(complex_array[position])


def standardize_phases(complex_array: np.ndarray) -> np.ndarray:
    """Uses `extract_phase` to obtain global phase from `array` and returns
    standardized array with global phase factor standardized.

    Parameters
    ----------
    complex_array:
        complex
    """
    phase = extract_phase(complex_array)
    std_array = complex_array * np.exp(-1j * phase)
    return std_array


def standardize_sign(real_array: np.ndarray) -> np.ndarray:
    """Standardizes the sign of a real-valued wavefunction by calculating the sign of
    the sum of all amplitudes up to the wavefunctions mid-position and making it
    positive.

    Summing up to the midpoint only is to address the  danger that the sum is
    actually zero, which may is the case for odd wavefunctions taken over an interval
    centered at zero.
    """
    halfway_position = len(real_array) // 2
    return np.sign(np.sum(real_array[:halfway_position])) * real_array


# -Matrix elements and operators (outside qutip) --------------------------------------


def matrix_element(
    state1: Union[np.ndarray, qt.Qobj],
    operator: Union[np.ndarray, csc_matrix, qt.Qobj],
    state2: Union[np.ndarray, qt.Qobj],
) -> Union[float, complex]:
    """Calculate the matrix element `<state1|operator|state2>`.

    Parameters
    ----------
    state1:
        state vector/ket
    state2:
        state vector/ket
    operator:
        representation of an operator

    Returns
    -------
        matrix element
    """
    if isinstance(operator, qt.Qobj):
        op_matrix = operator.data
    else:
        op_matrix = operator

    if isinstance(state1, qt.Qobj):
        vec1 = state1.data.toarray()
    else:
        vec1 = state1

    if isinstance(state2, qt.Qobj):
        vec2 = state2.data.toarray()
    else:
        vec2 = state2

    if isinstance(op_matrix, np.ndarray):  # Is operator given in dense form?
        # Yes - use numpy's 'vdot' and 'dot'.
        return np.vdot(vec1, np.dot(operator, vec2))
    # No, operator is sparse. Must use its own 'dot' method.
    return np.vdot(vec1, op_matrix.dot(vec2))


def get_matrixelement_table(
    operator: Union[np.ndarray, csc_matrix, dia_matrix, qt.Qobj],
    state_table: Union[np.ndarray, qt.Qobj],
) -> np.ndarray:
    """Calculates a table of matrix elements.

    Parameters
    ----------
    operator:
        operator with respect to which matrix elements are to be calculated
    state_table:
        list or array of numpy arrays representing the states `|v0>, |v1>, ...`
        Note: `state_table` is expected to be in scipy's `eigsh` transposed form.

    Returns
    -------
        table of matrix elements
    """
    if isinstance(operator, qt.Qobj):
        state_list = state_table
    else:
        state_list = state_table.T

    tablesize = len(state_list)
    mtable = [
        [
            matrix_element(state_list[n], operator, state_list[m])
            for m in range(tablesize)
        ]
        for n in range(tablesize)
    ]

    return np.asarray(mtable)


def closest_dressed_energy(
    bare_energy: float, dressed_energy_vals: np.ndarray
) -> float:
    """For a given bare energy value, this returns the closest lying dressed energy
    value from an array.

    Parameters
    ----------
    bare_energy:
        bare energy value
    dressed_energy_vals:
        array of dressed-energy values

    Returns
    -------
        element from `dressed_energy_vals` closest to `bare_energy`
    """
    index = (np.abs(dressed_energy_vals - bare_energy)).argmin()
    return dressed_energy_vals[index]


def get_eigenstate_index_maxoverlap(
    eigenstates_qobj: "QutipEigenstates",
    reference_state_qobj: qt.Qobj,
    return_overlap: bool = False,
) -> Union[int, Tuple[int, float], None]:
    """For given list of qutip states, find index of the state that has largest
    overlap with the qutip ket `reference_state_qobj`. If `|overlap|` is smaller than
    0.5, return None.

    Parameters
    ----------
    eigenstates_qobj:
        as obtained from qutip `.eigenstates()`
    reference_state_qobj:
        specific reference state
    return_overlap:
        set to true if the value of largest overlap should be also returned
        (default value = False)

    Returns
    -------
        index of eigenstate from `eigenstates_Qobj` with the largest overlap with the
        `reference_state_qobj`, None if `|overlap|<0.5`
    """
    overlaps = np.asarray(
        [
            eigenstates_qobj[j].overlap(reference_state_qobj)
            for j in range(len(eigenstates_qobj))
        ]
    )
    max_overlap = np.max(np.abs(overlaps))
    if max_overlap < 0.5:
        return None
    index = (np.abs(overlaps)).argmax()
    if return_overlap:
        return index, np.abs(overlaps[index])
    return index


def absorption_spectrum(spectrum_data: "SpectrumData") -> "SpectrumData":
    """Takes spectral data of energy eigenvalues and returns the absorption spectrum
    relative to a state of given index. Calculated by subtracting from eigenenergies
    the energy of the select state. Resulting negative frequencies, if the reference
    state is not the ground state, are omitted.
    """
    assert isinstance(spectrum_data.energy_table, ndarray)
    spectrum_data.energy_table = spectrum_data.energy_table.clip(min=0.0)  # type:ignore
    return spectrum_data


def emission_spectrum(spectrum_data: "SpectrumData") -> "SpectrumData":
    """Takes spectral data of energy eigenvalues and returns the emission spectrum
    relative to a state of given index. The resulting "upwards" transition
    frequencies are calculated by subtracting from eigenenergies the energy of the
    select state, and multiplying the result by -1. Resulting negative frequencies,
    corresponding to absorption instead, are omitted.
    """
    assert isinstance(spectrum_data.energy_table, ndarray)
    spectrum_data.energy_table *= -1.0
    spectrum_data.energy_table = spectrum_data.energy_table.clip(min=0.0)  # type:ignore
    return spectrum_data


def convert_evecs_to_ndarray(evecs_qutip: ndarray) -> np.ndarray:
    """Takes a qutip eigenstates array, as obtained with .eigenstates(), and converts
    it into a pure numpy array.

    Parameters
    ----------
    evecs_qutip:
        ndarray of eigenstates in qt.Qobj format

    Returns
    -------
        converted eigenstate data
    """
    evals_count = len(evecs_qutip)
    dimension = evecs_qutip[0].shape[0]
    evecs_ndarray = np.empty((evals_count, dimension), dtype=np.complex_)
    for index, eigenstate in enumerate(evecs_qutip):
        evecs_ndarray[index] = eigenstate.full()[:, 0]
    return evecs_ndarray


def convert_matrix_to_qobj(
    operator: Union[np.ndarray, csc_matrix, dia_matrix],
    subsystem: Union["QubitBaseClass", "Oscillator"],
    op_in_eigenbasis: bool,
    evecs: Optional[np.ndarray],
) -> qt.Qobj:
    dim = subsystem.truncated_dim

    if op_in_eigenbasis is False:
        if evecs is None:
            _, evecs = subsystem.eigensys(evals_count=dim)
        operator_matrixelements = get_matrixelement_table(operator, evecs)
        return qt.Qobj(inpt=operator_matrixelements)
    return qt.Qobj(inpt=operator[:dim, :dim])


def convert_opstring_to_qobj(
    operator: str,
    subsystem: Union["QubitBaseClass", "Oscillator"],
    evecs: Optional[np.ndarray],
) -> qt.Qobj:
    dim = subsystem.truncated_dim

    if evecs is None:
        _, evecs = subsystem.eigensys(evals_count=dim)
    operator_matrixelements = subsystem.matrixelement_table(operator, evecs=evecs)
    return qt.Qobj(inpt=operator_matrixelements)


def convert_operator_to_qobj(
    operator: Union[np.ndarray, csc_matrix, dia_matrix, qt.Qobj, str],
    subsystem: Union["QubitBaseClass", "Oscillator"],
    op_in_eigenbasis: bool,
    evecs: Optional[np.ndarray],
) -> qt.Qobj:
    if isinstance(operator, qt.Qobj):
        return operator
    if isinstance(operator, (np.ndarray, csc_matrix, dia_matrix)):
        return convert_matrix_to_qobj(operator, subsystem, op_in_eigenbasis, evecs)
    if isinstance(operator, str):
        return convert_opstring_to_qobj(operator, subsystem, evecs)
    raise TypeError("Unsupported operator type: ", type(operator))


def generate_target_states_list(
    sweep: "ParameterSweep", initial_state_labels: Tuple[int, ...]
) -> List[Tuple[int, ...]]:
    """Based on a bare state label (i1, i2, ...)  with i1 being the excitation level
    of subsystem 1, i2 the excitation level of subsystem 2 etc., generate a list of
    new bare state labels. These bare state labels correspond to target states
    reached from the given initial one by single-photon qubit transitions. These are
    transitions where one of the qubit excitation levels increases at a time. There
    are no changes in oscillator photon numbers.

    Parameters
    ----------
    sweep:
    initial_state_labels:
        bare-state labels of the initial state whose energy is supposed to be subtracted
        from the spectral data
    """
    target_states_list = []
    for qbt_subsys in sweep.qbt_subsys_list:  # iterate through qubit subsys_list
        assert qbt_subsys.truncated_dim is not None
        subsys_index = sweep._hilbertspace.get_subsys_index(qbt_subsys)
        initial_qbt_state = initial_state_labels[subsys_index]
        for state_label in range(initial_qbt_state + 1, qbt_subsys.truncated_dim):
            # for given qubit subsystem, generate target labels by increasing that qubit
            # excitation level
            target_labels = list(initial_state_labels)
            target_labels[subsys_index] = state_label
            target_states_list.append(tuple(target_labels))
    return target_states_list


def recast_esys_mapdata(
    esys_mapdata: List[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Takes data generated by a map of eigensystem calls and returns the eigenvalue and
    eigenstate tables

    Returns
    -------
        eigenvalues and eigenvectors
    """
    paramvals_count = len(esys_mapdata)
    eigenenergy_table = np.asarray(
        [esys_mapdata[index][0] for index in range(paramvals_count)]
    )
    eigenstate_table = [esys_mapdata[index][1] for index in range(paramvals_count)]
    return eigenenergy_table, eigenstate_table


def identity_wrap(
    operator: Union[str, ndarray, Qobj],
    subsystem: "QuantumSys",
    subsys_list: List["QuantumSys"],
    op_in_eigenbasis: bool = False,
    evecs: ndarray = None,
) -> Qobj:
    """Takes the `operator` belonging to `subsystem` and "wraps" it in identities.
    The full Hilbert space is taken to consist of all subsystems given as
    `subsys_list`. `subsystem` must be one element in that list. For each of the
    other subsystems in the list, an identity operator of the correct dimension is
    generated and inserted into the appropriate Kronecker product "sandwiching" the
    operator.

    Parameters
    ----------
    operator:
        operator acting in Hilbert space of `subsystem`; if str, then this should be an
        operator name in the subsystem, typically not in eigenbasis
    subsystem:
        subsystem where diagonal operator is defined
    subsys_list:
        list of all subsystems relevant to the Hilbert space.
    op_in_eigenbasis:
        whether `operator` is given in the `subsystem` eigenbasis; otherwise,
        `operator` is assumed to be in the internal QuantumSystem basis. This
        argument is ignored if `operator` is given as a Qobj.
    evecs:
        internal `QuantumSystem` eigenstates, used to convert `operator` into eigenbasis

    Returns
    -------
        operator in the full Hilbert space (as specified by `subsystem_list`). This
        operator is expressed in the bare product basis consisting of the energy
        eigenstates of each subsystem (unless `operator` is provided as a `Qobj`,
        in which case no conversion takes place).
    """
    subsys_operator = convert_operator_to_qobj(
        operator, subsystem, op_in_eigenbasis, evecs  # type:ignore
    )
    operator_identitywrap_list = [
        qt.operators.qeye(the_subsys.truncated_dim) for the_subsys in subsys_list
    ]
    subsystem_index = subsys_list.index(subsystem)
    operator_identitywrap_list[subsystem_index] = subsys_operator
    return qt.tensor(operator_identitywrap_list)
