"""
A simple Python module to obtain energy levels of superconducting qubits by sparse Hamiltonian diagonalization.
"""

import numpy as np
import sympy
from scipy.sparse.linalg import *
from abc import ABCMeta
from abc import abstractmethod
import logging

import scqubits.core.constants as constants
import scqubits.core.descriptors as descriptors

import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.utils.plot_defaults as defaults
import scqubits.utils.plotting as plot
from typing import Tuple, List, Union
from .elements import CircuitElement
from .variable import Variable

from scqubits.core.circuit.variable import Variable


class CircuitNode:
    def __init__(self, name):
        self.name = name


class Circuit(base.QubitBaseClass, serializers.Serializable):
    """
    The class containing references to nodes, elements, variables, variable-to-node mappings.
    """

    def __init__(self, tolerance: float = 1e-18, real_mode: bool = False):
        """
        Abritrary quantum circuit class.

        Parameters
        ----------
        tolerance: float, optional
            roundoff error tolerance  (default value = 1e-18)
        real_mode: bool, optional
            assume Hamiltonian is real-valued; yields real-valued wavefunctions where possible (default value = False)

        """
        self.nodes = [CircuitNode('GND')]
        self.elements = []
        self.wires = []
        self.variables = []
        self.linear_coordinate_transform = np.asarray(0)
        self.invalidation_flag = True
        self.tolerance = tolerance
        self.best_permutation_cache = {}
        self.phase_potential = None
        self.charge_potential = None
        self.real_mode = real_mode
        self.nodes_graph = []

    # TODO: add something
    @staticmethod
    def default_params():
        return {
        }

    # TODO: add something
    @staticmethod
    def nonfit_params():
        return []

    def grid_shape(self) -> Tuple[int, ...]:
        """Returns Hilbert space dimension
        Returns
        -------
        tuple of ints
        """
        return tuple([v.pt_count for v in self.variables])

    def hilbertdim(self) -> int:
        """Returns Hilbert space dimension"""
        return int(np.prod(self.grid_shape()))

    def potential(self, *args) -> np.ndarray:
        """Circuit phase-basis potential evaluated at `*args`, in order of `variables`. Variables of parameter type
        are skipped.

        Parameters
        ----------
        *args: floats
            phase variables value

        Returns
        -------
        float
        """
        phase_values = []
        variable_values = args.__iter__()
        for variable in self.variables:
            if variable.variable_type == 'variable':
                phase_values.append(variable_values.__next__())
            else:
                phase_values.append(variable.get_phase_grid()[0])

        energy = 0
        for element in self.elements:
            if not element.is_external() and element.is_phase():
                element_node_ids = []
                for wire in self.wires:
                    if wire[0] == element.name:
                        for node_id, node in enumerate(self.nodes):
                            if wire[1] == node.name:
                                element_node_ids.append(node_id)
                for variable_id in range(len(self.variables)):
                    energy += element.energy_term(np.tensordot(np.asarray(self.linear_coordinate_transform)[
                                              element_node_ids, variable_id], phase_values[variable_id], axes=0), None)
        return energy

    def get_variable_index(self, variable: Union[str, int]) -> int:
        """
        Returns the variable index by name or id

        Parameters
        ----------
        variable: str or int
            variable name or id

        Returns
        -------
        int
        """
        if type(variable) is not int:
            for variable_id, variable_ in enumerate(self.variables):
                if variable_.name == variable:
                    return variable_id
        return variable

    def phase_operator(self, variable: Union[str, int]) -> np.ndarray:
        """
        Returns phi operator along given dimension in phi basis

        Parameters
        ----------
        variable: str or int
            variable name or id

        Returns
        -------
            phi operator in Phi basis
        """
        return self.get_phase_grid()[self.get_variable_index(variable)]

    def phase_operator_action(self, state_vector: np.ndarray, variable: Union[str, int]) -> np.ndarray:
        """
        Returns the action of the phase operator on the state vector describing the system in phase representation

        Parameters
        ----------
        state_vector: ndarray
            wavefunction to act upon
        variable: str or int
            variable name or id

        Returns
        -------
        ndarray
        """
        shape = state_vector.shape
        return np.reshape(self.phase_operator(variable).ravel() * state_vector.ravel(), shape)

    def charge_operator(self, variable: Union[str, int]) -> np.ndarray:
        """
        Returns the select charge operator along given dimension in charge basis

        Parameters
        ----------
        variable: str or int
            variable name or id

        Returns
        -------
            charge operator in charge basis
        """

        return self.get_charge_grid()[self.get_variable_index(variable)]

    def charge_operator_action(self, state_vector, variable: Union[str, int]) -> np.ndarray:
        """
        Returns the action of the charge operator on the state vector describing the system in phase representation

        Parameters
        ----------
        state_vector: ndarray
            wavefunction to act upon
        variable: str or int
            variable name or id

        Returns
        -------
        ndarray
        """
        index = self.get_variable_index(variable)
        charge_wave = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(state_vector, axes=index), axis=index), axes=index)
        w = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(self.charge_operator(variable) * charge_wave, axes=index),
                                          axis=index), axes=index)
        return w

    def wavefunction_charge(self, state_vector) -> np.ndarray:
        """
        Returns wavefunction in charge basis
        Parameters
        ----------
        state_vector: ndarray
            wavefunction in phase representation

        Returns
        -------
        ndarray
        """
        return np.fft.fftshift(np.fft.fft(np.fft.fftshift(state_vector), norm='ortho'))

    def exp_i_phi_operator(self, variable: Union[str, int]) -> np.ndarray:
        """
        Returns imaginary exponent of phi operator :math:`e^{i\\phi}` along given dimension in phi basis

        Parameters
        ----------
        variable: str or int
            variable name or id

        Returns
        -------
        ndarray
        """
        return np.exp(1j * self.phase_operator(self.get_variable_index(variable)))

    def cos_phi_operator(self, variable: Union[str, int]) -> np.ndarray:
        """
        Returns cosine of phi operator :math:`\\cos\\left(i\\phi\\right)` along given dimension in phi basis

        Parameters
        ----------
        variable: str or int
            variable name or id

        Returns
        -------
        ndarray
        """
        return np.cos(self.phase_operator(self.get_variable_index(variable)))

    def sin_phi_operator(self, variable: Union[str, int]) -> np.ndarray:
        """
        Returns sine of phi operator :math:`\\sin\\left(i\\phi\\right)` along given dimension in phi basis

        Parameters
        ----------
        variable: str or int
            variable name or id

        Returns
        -------
        ndarray
        """
        return np.sin(self.phase_operator(self.get_variable_index(variable)))

    def operator_matrix_elements(self, operator, state_vector1, state_vector2) -> Union[complex, float]:
        """
        Returns the matrix element :math:`\\langle \\psi_2(\\phi) | O(\\phi) | \\psi_1(\\phi) \\rangle` for the
        selected operator in phase representation.

        Parameters
        ----------
        operator: ndarray
            operator :math:`O(\\phi)` in phase representation
        state_vector1: ndarray
            ket vector :math:`\\psi_1(\\phi)` in phase representation
        state_vector2: ndarray
            bra vector :math:`\\psi_2(\\phi)` in phase representation

        Returns
        -------
        complex or float
        """
        return np.sum(np.conj(state_vector2) * operator * state_vector1)

    def hamiltonian(self) -> np.ndarray:
        """
        Returns Hamiltonian in charge basis.

        Returns
        -------
        ndarray
        """
        dim = len(self.variables)
        phase_grid = np.reshape(self.get_phase_grid(), (dim, 1, -1))
        charge_grid = np.reshape(self.get_charge_grid(), (dim, -1, 1))
        unitary = np.exp(1j * np.sum(phase_grid * charge_grid, axis=0)) / np.sqrt(self.hilbertdim())
        hamiltonian_mat = unitary @ np.diag(self.calculate_phase_potential().ravel()) @ np.conj(unitary.T)
        hamiltonian_mat += np.diag(self.calculate_charge_potential().ravel())
        return hamiltonian_mat

    def find_element(self, element_name: str) -> CircuitElement:
        """
        Returns an element inside the circuit with the specified name, if found.

        Parameters
        ----------
        element_name: str

        Returns
        -------
        CircuitElement object or None
        """
        for element in self.elements:
            if element.name == element_name:
                return element

    def find_variable(self, variable_name: str) -> Variable:
        """
        Returns a variable of the circuit with the specified name, if found.
        Parameters
        ----------
        variable_name: str

        Returns
        -------
        Variable object or None
        """
        for variable in self.variables:
            if variable.name == variable_name:
                return variable

    def add_element(self, element: CircuitElement, node_names: List[str]):
        """
        Connect an element to the circuit.

        Parameters
        ----------
        element: CircuitElement object
            circuit element to insert into the circuit
        node_names: list of str
            list of names of the nodes to which the element should be connected

        Returns
        -------
        None
        """
        self.elements.append(element)
        self.nodes_graph.append(tuple(node_names))
        for node_name in node_names:
            nodes_found = 0
            for node in self.nodes:
                if node.name == node_name:
                    self.wires.append((element.name, node.name))
                    nodes_found += 1
            if nodes_found == 0:
                self.nodes.append(CircuitNode(node_name))
                self.wires.append((element.name, node_name))
        if element.is_external():
            if element.is_phase() or element.is_charge():
                self.add_variable(Variable(element.name))
        self.invalidation_flag = True

    def add_variable(self, variable):
        """
        Add variable to circuit.
        Parameters
        ----------
        variable: Variable object

        Returns
        -------
        None
        """
        counter = 0
        for var_id, var in enumerate(self.variables):
            if var.name == variable.name:
                logging.warning('Variable {0} with name {1} already exists'.format(variable, variable.name))
                self.variables[var_id] = variable
                counter += 1
        if counter == 0:
            self.variables.append(variable)
        self.invalidation_flag = True

    def map_nodes_linear(self, node_names: List[str], variable_names: List[str], coefficients: np.ndarray):
        """
        Sets the value of node phases (and, respectively, their conjugate charges) as a linear combination of the
        circuit variables. Checks for external fluxes and charges variables.
        Parameters
        ----------
        node_names: list of str
            names of the nodes to be expressed through the variables, in the order of the coefficient matrix rows
        variable_names: list of str
            variables to express the node phases through, in the order of the coefficient matrix columns
        coefficients: ndarray
            transformation matrix


        Returns
        -------
        None
        """
        node_ids = []
        variable_ids = []
        for node_name in node_names:
            for node_id, node in enumerate(self.nodes):
                if node.name == node_name:
                    node_ids.append(node_id)
        for variable_name in variable_names:
            for variable_id, variable in enumerate(self.variables):
                if variable.name == variable_name:
                    variable_ids.append(variable_id)
        if len(variable_ids) != len(self.variables):
            raise Exception('VariableError',
                            'Wrong number of variables in variable list. Got {0}, expected {1}'.format(
                                len(variable_ids), len(self.variables)))
        if len(node_ids) != len(self.nodes):
            raise Exception('VariableError',
                            'Wrong number of nodes in node list. Got {0}, expected {1}'.format(
                                len(node_ids), len(self.nodes)))
        variable_idx, node_idx = np.meshgrid(variable_ids, node_ids)
        self.linear_coordinate_transform = np.zeros(coefficients.shape, coefficients.dtype)
        self.linear_coordinate_transform[node_idx, variable_idx] = coefficients
        """ External fluxes and(/or) charges checking. """
        for element_id, element in enumerate(self.elements):
            if element.is_external():  # Check is only for external flux/charge variables
                counter = 0
                element_node_ids = []  # node ids for element.
                var_id = None
                var_list = np.zeros(len(variable_ids), coefficients.dtype)
                for variable_id, variable in enumerate(self.variables):
                    if variable.name == element.name:
                        var_id = variable_id  # id for external variable corresponding ext_flux/charge in self.variables
                for variable_name in variable_names:
                    if element.name == variable_name:
                        counter += 1  # counter for variables with the same name as element.name. It should be equal 1!
                        for wire in self.wires:
                            if wire[0] == element.name:
                                for node_id, node in enumerate(self.nodes):
                                    if wire[1] == node.name:
                                        element_node_ids.append(node_id)  # we need to have two node
                if counter == 0:
                    if element.is_phase():
                        raise Exception('VariableError',
                                        'There is no variable {0} for external flux in variable list'.format(
                                            element.name))
                    if element.is_charge():
                        raise Exception('VariableError',
                                        'There is no variable {0} for external charge in variable list'.format(
                                            element.name))
                else:
                    var_list = self.linear_coordinate_transform[element_node_ids[1], :] - \
                               self.linear_coordinate_transform[element_node_ids[0], :]
                if var_list[var_id] != 1:
                    raise Exception('ExternalVariableError',
                                    'There is external flux(charge) = {0}*{1} between nodes {2} and {3}, but it should be'
                                    'equal 1*{4}.'.format(var_list[var_id], element.name, element_node_ids[1],
                                                          element_node_ids[0], element.name))
                element_var_list = np.zeros(len(variable_ids), coefficients.dtype)
                element_var_list[var_id] = 1
                if not (var_list == element_var_list).all():
                    raise Exception('ExternalVariableError',
                                    'For variable list: {0}, external flux(charge) {1} between nodes {2} and {3} '
                                    'should be equal {4}, but it equals {5}.'.format(self.variables, element.name,
                                                                                     element_node_ids[1],
                                                                                     element_node_ids[0],
                                                                                     element_var_list, var_list))
        self.invalidation_flag = True

    def get_phase_grid(self) -> Tuple[np.ndarray, ...]:
        """
        Returns a tuple of n-d grids of the phase variables, where n is the number of variables in the circuit, on which
        the circuit wavefunction depends.

        Returns
        -------
        tuple of ndarray
        """
        axes = []
        for variable in self.variables:
            axes.append(variable.get_phase_grid())
        return np.meshgrid(*tuple(axes), indexing='ij')

    def get_charge_grid(self) -> Tuple[np.ndarray, ...]:
        """
        Returns a tuple of n-d grids of the charge variables, where n is the number of variables in the circuit, on
        which the circuit wavefunction, when transformed into charge representation, depends.

        Returns
        -------
        tuple of ndarray
        """
        axes = [variable.get_charge_grid() for variable in self.variables]
        return np.meshgrid(*tuple(axes), indexing='ij')

    def hamiltonian_phase_action(self, state_vector) -> np.ndarray:
        """
        Returns the action of the hamiltonian on the state vector describing the system in phase representation.

        Parameters
        ----------
        state_vector: ndarray
            wavefunction to act upon

        Returns
        -------
        ndarray
        """
        psi = np.reshape(state_vector, self.charge_potential.shape)
        phi = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(psi)))
        u = self.phase_potential.ravel() * state_vector
        t = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(self.charge_potential * phi))).ravel()
        h = u + t
        if self.real_mode:
            h = np.real(h)
        return h

    def capacitance_matrix(self, symbolic: bool = False) -> Union[sympy.Matrix, np.ndarray]:
        """
        Returns the linear capacitance matrix of the circuit with respect to the circuit nodes from the capacitances
        between them. The rows and columns are sorted according to the order in which the nodes are in the nodes
        attribute.

        Parameters
        ----------
        symbolic: bool (default value=False)
            if true, return symbolic sympy matrix; otherwise return numpy ndarray

        Returns
        -------
        sympy.Matrix or ndarray
        """
        if symbolic:
            capacitance_matrix = sympy.Matrix(np.zeros((len(self.nodes), len(self.nodes))))
        else:
            capacitance_matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for element in self.elements:
            if not element.is_external():
                if element.is_charge():
                    element_node_ids = []
                    for wire in self.wires:
                        if wire[0] == element.name:
                            for node_id, node in enumerate(self.nodes):
                                if wire[1] == node.name:
                                    element_node_ids.append(node_id)
                    if len(element_node_ids) != 2:
                        raise Exception('VariableError',
                                        'Wrong number of ports on capacitance, expected 2, got {0}'.format(
                                            len(element_node_ids)))
                    capacitance_matrix[element_node_ids[0], element_node_ids[0]] += element.get_capacitance()
                    capacitance_matrix[element_node_ids[0], element_node_ids[1]] += -element.get_capacitance()
                    capacitance_matrix[element_node_ids[1], element_node_ids[0]] += -element.get_capacitance()
                    capacitance_matrix[element_node_ids[1], element_node_ids[1]] += element.get_capacitance()
        return capacitance_matrix

    def capacitance_matrix_variables(self, symbolic=False):
        """
        Calculates the capacitance matrix for the energy term of the qubit Lagrangian in the variable representation.
        """

        if symbolic:
            C = self.linear_coordinate_transform.T * self.capacitance_matrix(
                symbolic) * self.linear_coordinate_transform
            C = sympy.Matrix([sympy.nsimplify(sympy.ratsimp(x)) for x in C]).reshape(*(C.shape))
        else:
            C = np.einsum('ji,jk,kl->il', self.linear_coordinate_transform, self.capacitance_matrix(symbolic),
                          self.linear_coordinate_transform)
        return C

    def capacitance_matrix_legendre_transform(self, symbolic=False):
        """
        Calculates the principle pivot transform of the capacitance matrix in variable representation with respect to "variables" as opposed to "parameters" for the Legendre transform
        """
        inverted_indices = [variable_id for variable_id, variable in enumerate(self.variables) if
                            variable.variable_type == 'variable']
        noninverted_indices = [variable_id for variable_id, variable in enumerate(self.variables) if
                               variable.variable_type == 'parameter']
		inverted_indeces = np.asarray(inverted_indeces, dtype=np.int32)
		noninverted_indeces = np.asarray(noninverted_indeces, dtype=np.int32)
        if symbolic:
            aii = self.capacitance_matrix_variables(symbolic)[inverted_indices, inverted_indices]
            ain = self.capacitance_matrix_variables(symbolic)[inverted_indices, noninverted_indices]
            ani = self.capacitance_matrix_variables(symbolic)[noninverted_indices, inverted_indices]
            # Ann = self.capacitance_matrix_variables(symbolic)[noninverted_indices, noninverted_indices]
            bii = aii.inv()
            bin = sympy.Matrix(-aii.inv() * ain)
            bni = sympy.Matrix(-ani * aii.inv())
            bnn = ani * aii.inv() * ain  # -Ann
            B = sympy.Matrix(np.zeros(self.capacitance_matrix_variables(symbolic).shape))
        else:
            aii = self.capacitance_matrix_variables(symbolic)[tuple(np.meshgrid(inverted_indices, inverted_indices))].T
            ain = self.capacitance_matrix_variables(symbolic)[
                tuple(np.meshgrid(inverted_indices, noninverted_indices))].T
            ani = self.capacitance_matrix_variables(symbolic)[
                tuple(np.meshgrid(noninverted_indices, inverted_indices))].T
            # Ann = self.capacitance_matrix_variables(symbolic)[np.meshgrid(noninverted_indices, noninverted_indices)].T
            bii = np.linalg.inv(aii)
            bin = -np.dot(np.linalg.inv(aii), ain)
            bni = -np.dot(ani, np.linalg.inv(aii))
            bnn = np.einsum('ij,jk,kl->il', ani, np.linalg.inv(aii), ain)  # -Ann
            B = np.empty(self.capacitance_matrix_variables(symbolic).shape)
        # if sympy could do indexing properly, we would have 3 time less code!!
        for i1, i2 in enumerate(inverted_indices):
            for j1, j2 in enumerate(inverted_indices):
                B[j2, i2] = bii[j1, i1]
        for i1, i2 in enumerate(noninverted_indices):
            for j1, j2 in enumerate(inverted_indices):
                B[j2, i2] = bin[j1, i1]
        for i1, i2 in enumerate(inverted_indices):
            for j1, j2 in enumerate(noninverted_indices):
                B[j2, i2] = bni[j1, i1]
        for i1, i2 in enumerate(noninverted_indices):
            for j1, j2 in enumerate(noninverted_indices):
                B[j2, i2] = bnn[j1, i1]
        return B

    def calculate_ndiagonal_hamiltonian(self, d1scheme, d2scheme):
        """
        Calculates the hamiltonian in phase representation in n-diagonal form
        :param d1scheme: finite difference scheme for first order derivatives
        :param d2scheme: finite difference scheme for second order derivatives
        :returns: the m-ndiagonal kinetic operator
        """
        n = len(d1scheme)
        if len(d1scheme) != len(d2scheme):
            raise Exception('ValueError', 'd1scheme and d2scheme lengths are not equal')
        if n < 3:
            raise Exception('ValueError', 'dscheme length is less than 3')
        if (n - 1) % 2 > 0:
            raise Exception('ValueError', 'dscheme length is even')

        self.ndiagonal_operator = np.zeros(tuple(n*np.ones((len(self.variables),), dtype=int))+self.grid_shape())
        slice_diagonal = [(n-1)/2 for v in self.variables]+[slice(0, v.pt_count, 1) for v in self.variables]

        ECmat = -0.5 * self.capacitance_matrix_legendre_transform()
        # d^2/dxi^2 type elements (C*_ii)
        for i in range(len(self.variables)):
            EC = ECmat[i, i]
            for column_id in range(n):
                slice_column = list(slice_diagonal)
                slice_column[i] = column_id
                self.ndiagonal_operator[slice_column] += EC / (self.variables[i].get_phase_step() ** 2) * d2scheme[
                    column_id]
        # d^2/dxidxj type elements (C*_ij)
        for i in range(len(self.variables)):
            nondiagonal = (x for x in range(len(self.variables)) if x != i)
            for j in nondiagonal:
                EC = ECmat[i, j]
                for column_id_i in range(n):
                    for column_id_j in range(n):
                        slice_column = list(slice_diagonal)
                        slice_column[i] = column_id_i
                        slice_column[j] = column_id_j
                        self.ndiagonal_operator[slice_column] += EC / (
                                self.variables[i].get_phase_step() * self.variables[j].get_phase_step()) * (
                                                                         d1scheme[column_id_i] * d1scheme[
                                                                     column_id_j])

        self.ndiagonal_operator[slice_diagonal] += self.phase_potential

        self.hamiltonian_ndiagonal = LinearOperator((np.prod(self.grid_shape()), np.prod(self.grid_shape())),
                                                    matvec=self.ndiagonal_operator_action)
        return self.ndiagonal_operator

    def ndiagonal_operator_action(self, psi):
        diagonal_shape = tuple([1] * len(self.variables)) + self.grid_shape()
        psi = np.reshape(psi, diagonal_shape)
        action = self.ndiagonal_operator * psi
        ndiagonal_columns = np.meshgrid(
            *tuple([range(self.ndiagonal_operator.shape[v_id]) for v_id in range(len(self.variables))]), indexing='ij')
        ndiagonal_columns = np.reshape(ndiagonal_columns, (
            len(self.variables), np.prod(self.ndiagonal_operator.shape[0:len(self.variables)])))
        ndiagonal_shifts = np.meshgrid(*tuple([np.linspace(
            -(self.ndiagonal_operator.shape[v_id] - 1) / 2,
            (self.ndiagonal_operator.shape[v_id] - 1) / 2,
            self.ndiagonal_operator.shape[v_id], dtype=int) for v_id in range(len(self.variables))]), indexing='ij')
        ndiagonal_shifts = np.reshape(ndiagonal_shifts, ndiagonal_columns.shape)

        result = np.zeros(self.grid_shape(), dtype=np.complex)
        for i in range(np.prod(self.ndiagonal_operator.shape[0:len(self.variables)])):
            psii = action[tuple(ndiagonal_columns[:, i]) + tuple([slice(None, None, None)] * len(self.variables))]
            for v_id in range(len(self.variables)):
                psii = np.roll(psii, ndiagonal_shifts[v_id, i], axis=v_id)
            result += psii
        return result

    def calculate_phase_potential(self):
        """
        Calculates the potential landspace of the circuit phase-dependent energy in phase representation. 
        :returns: the phase potential landscape on the wavefunction grid.
        """
        grid_shape = self.grid_shape()
        grid_size = np.prod(grid_shape)
        phase_grid = self.get_phase_grid()
        self.phase_potential = np.zeros(grid_shape)
        for element in self.elements:
            if not element.is_external():
                element_node_ids = []
                for wire in self.wires:
                    if wire[0] == element.name:
                        for node_id, node in enumerate(self.nodes):
                            if wire[1] == node.name:
                                element_node_ids.append(node_id)
                phase_grid = np.reshape(np.asarray(phase_grid), (len(self.variables), grid_size))
                node_phases = np.einsum('ij,jk->ik', self.linear_coordinate_transform, phase_grid)[element_node_ids, :]
                node_phases = np.reshape(node_phases, (len(element_node_ids),) + grid_shape)
                if element.is_phase():
                    self.phase_potential += element.energy_term(node_phases=node_phases,
                                                                node_charges=np.zeros(node_phases.shape))
        return self.phase_potential

    def calculate_charge_potential(self):
        """
        Calculates the potential landspace of the circuit charge-dependent energy in charge representation. 
        :returns: the charge potential landscape on the wavefunction grid.
        """
        grid_shape = self.grid_shape()
        grid_size = np.prod(grid_shape)
        charge_grid = np.reshape(np.asarray(self.get_charge_grid()), (len(self.variables), grid_size))
        ECmat = 0.5 * self.capacitance_matrix_legendre_transform()
        self.charge_potential = np.einsum('ij,ik,kj->j', charge_grid, ECmat, charge_grid)
        self.charge_potential = np.reshape(self.charge_potential, grid_shape)
        return self.charge_potential

    def calculate_potentials(self):
        """
        Calculate potentials for Fourier-based hamiltonian action.
        """
        phase_potential = self.calculate_phase_potential()
        charge_potential = self.calculate_charge_potential()
        self.hamiltonian_Fourier = LinearOperator((np.prod(self.grid_shape()), np.prod(self.grid_shape())),
                                                  matvec=self.hamiltonian_phase_action)
        return self.charge_potential, self.phase_potential

    def diagonalize_phase(self, num_states=2, use_sparse=True, hamiltonian_type='Fourier', maxiter=1000):
        """Performs sparse diagonalization of the circuit hamiltonian.
        :param num_states: number of states, starting from the ground state, to be obtained.
        :returns: energies and wavefunctions of the first num_states states.
        """
        if hamiltonian_type == 'Fourier':
            energies, wavefunctions = eigs(self.hamiltonian_Fourier, k=num_states, which='SR', maxiter=maxiter)
        elif hamiltonian_type == 'ndiagonal':
            energies, wavefunctions = eigs(self.hamiltonian_ndiagonal, k=num_states, which='SR', maxiter=maxiter)
        energy_order = np.argsort(np.real(energies))
        energies = energies[energy_order]
        wavefunctions = wavefunctions[:, energy_order]
        wavefunctions = np.reshape(wavefunctions, self.charge_potential.shape + (num_states,))
        for state_id in range(num_states):
            wavefunction = wavefunctions[..., state_id]
            ind_max = np.unravel_index(np.argmax(np.abs(wavefunction), axis=None), wavefunction.shape)
            wavefunction *= np.exp(-1j * np.angle(wavefunction[ind_max]))
            wavefunctions[..., state_id] = wavefunction
        return energies, wavefunctions

    def symbolic_lagrangian(self):
        variable_phase_symbols = []
        variable_voltage_symbols = []
        for variable_id, variable in enumerate(self.variables):
            variable.phase_symbol = sympy.Symbol(variable.name)
            variable.voltage_symbol = sympy.Symbol('\\partial_t' + variable.name)
            variable_phase_symbols.append(variable.phase_symbol)
            variable_voltage_symbols.append(variable.voltage_symbol)
        variable_phase_symbols = sympy.Matrix(variable_phase_symbols)
        variable_voltage_symbols = sympy.Matrix(variable_voltage_symbols)
        node_phase_symbols = self.linear_coordinate_transform * variable_phase_symbols
        node_voltage_symbols = self.linear_coordinate_transform * variable_voltage_symbols
        for node_id, node in enumerate(self.nodes):
            node.phase_symbol = node_phase_symbols[node_id]
            node.voltage_symbol = node_voltage_symbols[node_id]
        kinetic_energy = sympy.nsimplify(
            (0.5 * node_voltage_symbols.T * self.capacitance_matrix(symbolic=True) * node_voltage_symbols)[0, 0])
        potential_energy = 0
        for element in self.elements:
            if not element.is_external():
                if element.is_phase():
                    element_node_phases = []
                    element_node_voltages = []
                    for wire in self.wires:
                        if wire[0] == element.name:
                            for node_id, node in enumerate(self.nodes):
                                if wire[1] == node.name:
                                    element_node_phases.append(sympy.nsimplify(node.phase_symbol))
                                    element_node_voltages.append(sympy.nsimplify(node.voltage_symbol))
                    potential_energy += element.symbolic_energy_term(element_node_phases, 0)
        return kinetic_energy - potential_energy

    def symbolic_hamiltonian(self):
        variable_phase_symbols = []
        variable_charge_symbols = []
        for variable_id, variable in enumerate(self.variables):
            variable.phase_symbol = sympy.Symbol(variable.name)
            if variable.variable_type == 'variable':
                variable.charge_symbol = -sympy.I * sympy.Symbol('\\partial_{' + variable.name + '}')
            else:
                variable.charge_symbol = sympy.Symbol('\\partial_t' + variable.name)
            variable_phase_symbols.append(variable.phase_symbol)
            variable_charge_symbols.append(variable.charge_symbol)
        variable_phase_symbols = sympy.Matrix(variable_phase_symbols)
        variable_charge_symbols = sympy.Matrix(variable_charge_symbols)

        node_phase_symbols = self.linear_coordinate_transform * variable_phase_symbols
        for node_id, node in enumerate(self.nodes):
            node.phase_symbol = node_phase_symbols[node_id]
        kinetic_energy = 0.5 * sympy.nsimplify((variable_charge_symbols.T * self.capacitance_matrix_legendre_transform(
            symbolic=True) * variable_charge_symbols)[0, 0])
        potential_energy = 0
        for element in self.elements:
            if not element.is_external():
                if element.is_phase():
                    element_node_phases = []
                    element_node_voltages = []
                    for wire in self.wires:
                        if wire[0] == element.name:
                            for node_id, node in enumerate(self.nodes):
                                if wire[1] == node.name:
                                    element_node_phases.append(sympy.nsimplify(node.phase_symbol))
                    potential_energy += element.symbolic_energy_term(element_node_phases, 0)
        return kinetic_energy + potential_energy

    def plot_potential(self, phi_grid=None, contour_vals=None, **kwargs):
        """
        Visualize the potential energy.

        Parameters
        ----------
        phi_grid: Grid1d, optional
            used for setting a custom grid for phi; if None use self._default_grid
        contour_vals: list of float, optional
            specific contours to draw
        **kwargs:
            plot options
        """
        num_variables = 0
        for v in self.variables:
            if v.variable_type == 'variable':
                num_variables += 1

        phi_grid = phi_grid or self._default_grid
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (5, 5)
        x_vals = phi_grid.make_linspace()

        if num_variables == 1:
            return plot.plot(x_vals, self.potential(x_vals), **kwargs)
        elif num_variables == 2:
            y_vals = phi_grid.make_linspace()
            return plot.contours(x_vals, y_vals, self.potential, contour_vals=contour_vals, **kwargs)
        elif num_variables == 3:
            raise ValueError('Dimension of potential higher than 2, plot_potential failed')