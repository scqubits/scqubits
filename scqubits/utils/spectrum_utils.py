# spectrum_utils.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import cmath

import numpy as np
import qutip as qt


def order_eigensystem(evals, evecs):
    """Takes eigenvalues and corresponding eigenvectors and orders them (in place) according to the eigenvalues (from
    smallest to largest; real valued eigenvalues are assumed). Compare http://stackoverflow.com/questions/22806398.

    Parameters
    ----------
    evals: ndarray
        array of eigenvalues
    evecs: ndarray
        array containing eigenvectors; evecs[:, 0] is the first eigenvector etc.

    """
    ordered_evals_indices = evals.argsort()  # eigsh does not guarantee consistent ordering within result
    evals[:] = evals[ordered_evals_indices]
    evecs[:] = evecs[:, ordered_evals_indices]
    return evals, evecs


def extract_phase(complex_array, position=None):
    """Extracts global phase from `complex_array` at given `position`. If position is not specified, the `position` is
    set to to an intermediate position to avoid machine-precision problems with tails of wavefunctions at beginning
    or end of the array.

    Parameters
    ----------
    complex_array: ndarray
        complex-valued array
    position: int, optional
        position where the phase is extracted (default value = None)
    """
    if position is None:
        flattened_position = np.argmax(
            np.abs(complex_array))  # extract phase from element with largest amplitude modulus
        position = np.unravel_index(flattened_position, complex_array.shape)
    return cmath.phase(complex_array[position])


def standardize_phases(complex_array):
    """Uses `extract_phase` to obtain global phase from `array` and returns standardized array with global phase factor
    standardized.

    Parameters
    ----------
    complex_array: ndarray
        complex

    Returns
    -------
    ndarray (complex)
    """
    phase = extract_phase(complex_array)
    std_array = complex_array * np.exp(-1j * phase)
    return std_array


def standardize_sign(real_array):
    """Standardizes the sign of a real-valued wavefunction by calculating the sign of the sum of all amplitudes and
    making it positive.

    Parameters
    ----------
    real_array: ndarray

    Returns
    -------
    ndarray (float)
    """
    return np.sign(np.sum(real_array)) * real_array


# -Matrix elements and operators (outside qutip) ----------------------------------------------------------------------


def matrix_element(state1, operator, state2):
    """Calculate the matrix element `<state1|operator|state2>`.

    Parameters
    ----------
    state1: ndarray or qutip.Qobj
        state vector/ket
    state2: ndarray or qutip.Qobj
        state vector/ket
    operator: qutip.Qobj or numpy array or numpy sparse object
        representation of an operator

    Returns
    -------
    float or complex
        matrix element
    """
    if isinstance(operator, qt.Qobj):
        op_matrix = operator.data
    else:
        op_matrix = operator

    if isinstance(state1, qt.Qobj):
        vec1 = state1.data.toarray()
        vec2 = state2.data.toarray()
    else:
        vec1 = state1
        vec2 = state2

    if isinstance(op_matrix, np.ndarray):  # Is operator given in dense form?
        return np.vdot(vec1, np.dot(operator, vec2))  # Yes - use numpy's 'vdot' and 'dot'.
    return np.vdot(vec1, op_matrix.dot(vec2))  # No, operator is sparse. Must use its own 'dot' method.


def get_matrixelement_table(operator, state_table):
    """Calculates a table of matrix elements.

    Parameters
    ----------
    operator: ndarray or sparse matrix object
        operator with respect to which matrix elements are to be calculated
    state_table: list or ndarray
        list or array of numpy arrays representing the states `|v0>, |v1>, ...`
        Note: `state_table` is expected to be in scipy's `eigsh` transposed form.

    Returns
    -------
    ndarray
        table of matrix elements
    """
    if isinstance(operator, qt.Qobj):
        state_list = state_table
    else:
        state_list = state_table.T

    tablesize = len(state_list)
    mtable = [[matrix_element(state_list[n], operator, state_list[m]) for m in range(tablesize)]
              for n in range(tablesize)]

    return np.asarray(mtable)


def closest_dressed_energy(bare_energy, dressed_energy_vals):
    """For a given bare energy value, this returns the closest lying dressed energy value from an array.

    Parameters
    ----------
    bare_energy: float
        bare energy value
    dressed_energy_vals: ndarray
        array of dressed-energy values

    Returns
    -------
    float
        element from `dressed_energy_vals` closest to `bare_energy`
    """
    index = (np.abs(dressed_energy_vals - bare_energy)).argmin()
    return dressed_energy_vals[index]


def get_eigenstate_index_maxoverlap(eigenstates_qobj, reference_state_qobj, return_overlap=False):
    """For given list of qutip states, find index of the state that has largest overlap with the qutip ket
    `reference_state_qobj`. If `|overlap|` is smaller than 0.5, return None.

    Parameters
    ----------
    eigenstates_qobj: ndarray of qutip.Qobj
        as obtained from qutip `.eigenstates()`
    reference_state_qobj: qutip.Qobj ket
        specific reference state
    return_overlap: bool, optional
        set to true if the value of largest overlap should be also returned (default value = False)

    Returns
    -------
    int or None
        index of eigenstate from `eigenstates_Qobj` with the largest overlap with the `reference_state_qobj`;
        None if `|overlap|<0.5`
    """
    overlaps = np.asarray([eigenstates_qobj[j].overlap(reference_state_qobj) for j in range(len(eigenstates_qobj))])
    max_overlap = np.max(np.abs(overlaps))
    if max_overlap < 0.5:
        return None
    index = (np.abs(overlaps)).argmax()
    if return_overlap:
        return index, np.abs(overlaps[index])
    return index


def absorption_spectrum(spectrum_data):
    """Takes spectral data of energy eigenvalues and returns the absorption spectrum relative to a state
    of given index. Calculated by subtracting from eigenenergies the energy of the select state. Resulting negative
    frequencies, if the reference state is not the ground state, are omitted.

    Parameters
    ----------
    spectrum_data: SpectrumData

    Returns
    -------
    SpectrumData object
    """
    spectrum_data.energy_table = spectrum_data.energy_table.clip(min=0.0)
    return spectrum_data


def emission_spectrum(spectrum_data):
    """Takes spectral data of energy eigenvalues and returns the emission spectrum relative to a state
    of given index. The resulting "upwards" transition frequencies are calculated by subtracting from eigenenergies
    the energy of the select state, and multiplying the result by -1. Resulting negative
    frequencies, corresponding to absorption instead, are omitted.

    Parameters
    ----------
    spectrum_data: SpectrumData

    Returns
    -------
    SpectrumData object
    """
    spectrum_data.energy_table *= -1.0
    spectrum_data.energy_table = spectrum_data.energy_table.clip(min=0.0)
    return spectrum_data


def convert_esys_to_ndarray(esys_qutip):
    """Takes a qutip eigenstates array, as obtained with .eigenstates(), and converts it into a pure numpy array.

    Parameters
    ----------
    esys_qutip: ndarray of qutip.Qobj
        as obtained from qutip `.eigenstates()`

    Returns
    -------
    ndarray
        converted eigenstate data
    """
    evals_count = len(esys_qutip)
    dimension = esys_qutip[0].shape[0]
    esys_ndarray = np.empty((evals_count, dimension), dtype=np.complex_)
    for index, eigenstate in enumerate(esys_qutip):
        esys_ndarray[index] = eigenstate.full()[:, 0]
    return esys_ndarray


def convert_ndarray_to_qobj(operator, subsystem, op_in_eigenbasis, evecs):
    dim = subsystem.truncated_dim
    if op_in_eigenbasis is False:
        if evecs is None:
            _, evecs = subsystem.eigensys(evals_count=subsystem.truncated_dim)
        operator_matrixelements = get_matrixelement_table(operator, evecs)
        return qt.Qobj(inpt=operator_matrixelements)
    return qt.Qobj(inpt=operator[:dim, :dim])


def convert_opstring_to_qobj(operator, subsystem, evecs):
    if evecs is None:
        _, evecs = subsystem.eigensys(evals_count=subsystem.truncated_dim)
    operator_matrixelements = subsystem.matrixelement_table(operator, evecs=evecs)
    return qt.Qobj(inpt=operator_matrixelements)


def convert_operator_to_qobj(operator, subsystem, op_in_eigenbasis, evecs):
    if isinstance(operator, qt.Qobj):
        return operator
    if isinstance(operator, np.ndarray):
        return convert_ndarray_to_qobj(operator, subsystem, op_in_eigenbasis, evecs)
    if isinstance(operator, str):
        return convert_opstring_to_qobj(operator, subsystem, evecs)
    raise TypeError('Unsupported operator type: ', type(operator))


def generate_target_states_list(sweep, initial_state_labels):
    """Based on a bare state label (i1, i2, ...)  with i1 being the excitation level of subsystem 1, i2 the
    excitation level of subsystem 2 etc., generate a list of new bare state labels. These bare state labels
    correspond to target states reached from the given initial one by single-photon qubit transitions. These
    are transitions where one of the qubit excitation levels increases at a time. There are no changes in
    oscillator photon numbers.

    Parameters
    ----------
    sweep: ParameterSweep
    initial_state_labels: tuple(int1, int2, ...)
        bare-state labels of the initial state whose energy is supposed to be subtracted from the spectral data

    Returns
    -------
    list of tuple"""
    target_states_list = []
    for subsys_index, qbt_subsys in sweep.qbt_subsys_list:   # iterate through qubit subsys_list
        initial_qbt_state = initial_state_labels[subsys_index]
        for state_label in range(initial_qbt_state + 1, qbt_subsys.truncated_dim):
            # for given qubit subsystem, generate target labels by increasing that qubit excitation level
            target_labels = list(initial_state_labels)
            target_labels[subsys_index] = state_label
            target_states_list.append(tuple(target_labels))
    return target_states_list


def recast_esys_mapdata(esys_mapdata):
    """
    Takes data generated by a map of eigensystem calls and returns the eigenvalue and eigenstate tables

    Parameters
    ----------
    esys_mapdata: list of tuple of ndarray

    Returns
    -------
    (ndarray, ndarray)
        eigenvalues and eigenvectors
    """
    paramvals_count = len(esys_mapdata)
    eigenenergy_table = np.asarray([esys_mapdata[index][0] for index in range(paramvals_count)])
    eigenstate_table = [esys_mapdata[index][1] for index in range(paramvals_count)]
    return eigenenergy_table, eigenstate_table
