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

import numpy as np
import qutip as qt
import cmath


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
        position where the phase is extracted (Default value = None)
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
    array: ndarray
        complex

    Returns
    -------
    ndarray (complex)
    """
    phase = extract_phase(complex_array)
    std_array = complex_array * np.exp(-1j * phase)
    return std_array


# —Matrix elements and operators (outside qutip) ———————————————————————————————————————————————————————————————————————


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
    else:
        return np.vdot(vec1, op_matrix.dot(vec2))  # No, operator is sparse. Must use its own 'dot' method.


def get_matrixelement_table(operator, state_table, real_valued=False):
    """Calculates a table of matrix elements.

    Parameters
    ----------
    operator: ndarray or sparse matrix object
        operator with respect to which matrix elements are to be calculated
    state_table: list or ndarray
        list or array of numpy arrays representing the states `|v0>, |v1>, ...`
        Note: `state_table` is expected to be in scipy's `eigsh` transposed form.
    real_valued: bool, optional
        signals whether matrix elements are real valued (Default value = False)

    Returns
    -------
    ndarray
        table of matrix elements
    """
    if isinstance(operator, qt.Qobj):
        state_list = state_table
    else:
        state_list = state_table.T

    if real_valued:
        the_dtype = np.float_
    else:
        the_dtype = np.complex_

    tablesize = len(state_list)
    mtable = np.empty(shape=[tablesize, tablesize], dtype=the_dtype)
    for n in range(tablesize):
        for m in range(n + 1):
            mtable[n, m] = matrix_element(state_list[n], operator, state_list[m])
            if real_valued:
                mtable[m, n] = mtable[n, m]
            else:
                mtable[m, n] = np.conj(mtable[n, m])
    return mtable


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


def get_eigenstate_index_maxoverlap(eigenstates_Qobj, reference_state_Qobj, return_overlap=False):
    """For given list of qutip states, find index of the state that has largest overlap with the qutip ket
    `reference_state_Qobj`.

    Parameters
    ----------
    eigenstates_Qobj: ndarray of qutip.Qobj
        as obtained from qutip `.eigenstates()`
    reference_state_Qobj: qutip.Qobj ket
        specific reference state
    return_overlap: bool, optional
        set to true if the value of largest overlap should be also returned (Default value = False)

    Returns
    -------
    int
        index of eigenstate from `eigenstates_Qobj` with the largest overlap with the `reference_state_Qobj`
    """
    overlaps = np.asarray([eigenstates_Qobj[j].overlap(reference_state_Qobj) for j in range(len(eigenstates_Qobj))])
    index = (np.abs(overlaps)).argmax()
    if return_overlap:
        return index, np.abs(overlaps[index])
    else:
        return index


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
        esys_ndarray[index] = eigenstate.full().reshape(dimension)

    return esys_ndarray
