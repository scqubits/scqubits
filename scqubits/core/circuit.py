"""
A simple Python module to obtain energy levels of superconducting qubits by sparse Hamiltonian diagonalization.
"""

import numpy as np
import sympy
from scipy.sparse.linalg import *
from abc import ABCMeta
from abc import abstractmethod

import scqubits.core.constants as constants
import scqubits.core.descriptors as descriptors
import scqubits.core.discretization as discretization
import scqubits.core.qubit_base as base
import scqubits.core.storage as storage
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.utils.plot_defaults as defaults
import scqubits.utils.plotting as plot


class CircuitNode:
    def __init__(self, name):
        self.name = name


class Variable(discretization.Grid1d):
    """
    Represents a variable of the circuit wavefunction or an constant external bias flux or voltage.
    """
    
    def __init__(self, name):
        self.variable_type = 'parameter'
        super().__init__(0, 0, 1)
        self.name = name
        self.offset_charge = 0

    def set_variable(self, pt_count, periods, center=0):
        """Creates a discrete grid for phase wavefunction variables.

        Parameters
        ----------
        pt_count: int
            number of grid points
        periods: float
            number of 2pi intervals in the grid
        center: float
            phase grid centering
        """
        self.variable_type = 'variable'
        self.pt_count = pt_count

        self.min_val = -np.pi * periods + center
        self.max_val = self.min_val + 2 * np.pi * periods * (self.pt_count - 1) / self.pt_count
        self.offset_charge = 0 # set offset charge in variable to 0

    def set_parameter(self, phase_value, voltage_value):
        """Makes the parameter an external flux and/or charge bias.

        Parameters
        ----------
        phase_value: float
            external flux bias in flux quanta/(2pi)
        voltage_value: float
            external voltage bias
        """
        self.variable_type = 'parameter'
        self.pt_count = 1

        self.min_val = phase_value
        self.max_val = phase_value
        self.offset_charge = voltage_value

    def get_phase_grid(self):
        """Returns a numpy array of the grid points in phase representation

        Returns
        -------
        ndarray
        """
        return self.make_linspace()

    def get_charge_grid(self):
        """Returns a numpy array of the grid points in cooper pair number representation

        Returns
        -------
        ndarray
        """
        if self.pt_count >1:
            range = (self.max_val - self.min_val) * self.pt_count / (self.pt_count - 1)
            delta_n = 2*np.pi/range
            grid = np.arange(0, delta_n*self.pt_count, delta_n)
        else:
            grid = np.asarray([0.0])
        grid -= np.mean(grid)
        grid += self.offset_charge
        if self.pt_count % 2 == 0:
            grid -= 0.5
        return grid


class CircuitElement:
    """
    Abstract class for circuit elements. All circuit elements defined in the QCircuit library derive from this base class.
    """
    
    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def is_phase(self):
        pass

    @abstractmethod
    def is_charge(self):
        pass

    @abstractmethod
    def energy_term(self, node_phases, node_charges):
        return None

    @abstractmethod
    def symbolic_energy_term(self, node_phases, node_charges):
        return None


class Capacitance(CircuitElement):
    """
    Circuit element representing a capacitor.
    """
    
    def __init__(self, name, capacitance=0):
        super().__init__(name)
        self.capacitance = capacitance

    def set_capacitance(self, capacitance):
        self.capacitance = capacitance

    def get_capacitance(self):
        return self.capacitance

    def is_phase(self):
        return False

    def is_charge(self):
        return True

    def energy_term(self, node_phases, node_charges):
        return None

    def symbolic_energy_term(self, node_phases, node_charges):
        return None


class JosephsonJunction(CircuitElement):
    """
    Circuit element representing a Josephson junction.
    """
    def __init__(self, name, critical_current=0, use_offset=True):
        super().__init__(name)
        self.critical_current = critical_current
        self.use_offset = use_offset

    def set_critical_current(self, critical_current):
        self.critical_current = critical_current

    def get_critical_current(self):
        return self.critical_current

    def energy_term(self, node_phases, node_charges):
        if len(node_phases) != 2:
            raise Exception('ConnectionError', 
                            'Josephson junction {0} has {1} nodes connected instead of 2.'.format(self.name, len(node_phases)))
        if self.use_offset:
            return self.critical_current * (1 - np.cos(node_phases[0] - node_phases[1]))
        else:
            return self.critical_current * (-np.cos(node_phases[0] - node_phases[1]))

    def symbolic_energy_term(self, node_phases, node_charges):
        if len(node_phases) != 2:
            raise Exception('ConnectionError', 
                            'Josephson junction {0} has {1} nodes connected instead of 2.'.format(self.name, len(node_phases)))
        if self.use_offset:
            return self.critical_current * (1 - sympy.cos(node_phases[0] - node_phases[1]))
        else:
            return self.critical_current * (-sympy.cos(node_phases[0] - node_phases[1]))

    def is_phase(self):
        return True

    def is_charge(self):
        return False


class Inductance(CircuitElement):
    """
    Circuit element representing a linear inductor.
    """
    def __init__(self, name, inductance=0):
        super().__init__(name)
        self.inductance = inductance

    def set_inductance(self, inductance):
        self.inductance = inductance

    def get_inductance(self):
        return self.inductance

    def energy_term(self, node_phases, node_charges):
        if len(node_phases) != 2:
            raise Exception('ConnectionError', 
                            'Inductance {0} has {1} nodes connected instead of 2.'.format(self.name, len(node_phases)))
        return (node_phases[0]-node_phases[1])**2/(2*self.inductance)

    def symbolic_energy_term(self, node_phases, node_charges):
        if len(node_phases) != 2:
            raise Exception('ConnectionError', 
                            'Inductance {0} has {1} nodes connected instead of 2.'.format(self.name, len(node_phases)))
        return (node_phases[0]-node_phases[1])**2/(2*self.inductance)

    def is_phase(self):
        return True

    def is_charge(self):
        return False


class LagrangianCurrentSource(CircuitElement):
    """
    Circuit element representing a Josephson junction.
    """
    def __init__(self, name, current=0):
        super().__init__(name)
        self.current = current

    def set_current(self, current):
        self.current = current

    def get_current(self):
        return self.current

    def energy_term(self, node_phases, node_charges):
        if len(node_phases) != 2:
            raise Exception('ConnectionError', 
                            'Lagrangian current source {0} has {1} nodes connected instead of 2.'.format(self.name, len(node_phases)))
        return self.current*(node_phases[0]-node_phases[1])

    def symbolic_energy_term(self, node_phases, node_charges):
        return self.energy_term(node_phases, node_charges)

    def is_phase(self):
        return True

    def is_charge(self):
        return False

    
class Circuit(base.QubitBaseClass, serializers.Serializable):
    """
    The class containing references to nodes, elements, variables, variable-to-node mappings.
    """
    def __init__(self, tolerance: float = 1e-18):
        """
        Default constructor.
        :param tolerance: capacitances below this value are considered to be computational errors when determining the inverse capacitance matrix.
        :type tolerance: float
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

    def hilbertdim(self):
        """Returns Hilbert space dimension"""
        return np.prod(self.grid_shape())

    def potential(self, *args):
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
            if element.is_phase():
                element_node_ids = []
                for wire in self.wires:
                    if wire[0] == element.name:
                        for node_id, node in enumerate(self.nodes):
                            if wire[1] == node.name:
                                element_node_ids.append(node_id)
                energy += element.energy_term(np.asarray(self.linear_coordinate_transform)[
                                              element_node_ids, :]@phase_values, None)
        return energy

    def hamiltonian(self):
        """Returns Hamiltonian in charge basis"""
        dim = len(self.variables)
        phase_grid = np.reshape(self.create_phase_grid(), (dim, 1, -1))
        charge_grid = np.reshape(self.create_charge_grid(), (dim, -1, 1))
        unitary = np.exp(1j*np.sum(phase_grid*charge_grid, axis=0))/np.sqrt(self.hilbertdim())
        hamiltonian_mat = unitary@np.diag(self.calculate_phase_potential().ravel())@np.conj(unitary.T)
        hamiltonian_mat += np.diag(self.calculate_charge_potential().ravel())
        return hamiltonian_mat

    def find_element(self, element_name):
        """
        Find an element inside the circuit with the specified name.
        :returns: the element, if found, else None
        """
        for element in self.elements:
            if element.name == element_name:
                return element
            
    def find_variable(self, variable_name):
        """
        Find a variable of the circuit with the specified name.
        :returns: the variable, if found
        """
        for variable in self.variables:
            if variable.name == variable_name:
                return variable
        
    def add_element(self, element, node_names):
        """
        Connect an element to the circuit.
        :param element: circuit element to insert into the circuit
        :type element: CircuitElement
        :param node_names: list of names of the nodes to which the element should be connected
        :type node_names: list of str
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
        self.invalidation_flag = True
        
    def add_variable(self, variable):
        """
        Add variable to circuit.
        :param variable:
        :type variable: Variable
        """
        self.variables.append(variable)
        self.invalidation_flag = True
        
    def map_nodes_linear(self, node_names, variable_names, coefficients):
        """
        Sets the value of node phases (and, respectively, their conjugate charges) as a linear combination of the circuit variables.
        :param node_names: the names of the nodes to be expressed through the variables, in the order of the coefficient matrix rows.
        :param variable_names: the variables to express the node phases through, in the order of the coefficient matrix columns.
        :param coefficients: the transfrmation matrix
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
        self.invalidation_flag = True
        
    def grid_shape(self):
        return tuple([v.pt_count for v in self.variables])
    
    def create_phase_grid(self):
        """
        Creates a n-d grid of the phase variables, where n is the number of variables in the circuit, on which the circuit wavefunction depends.
        :returns: tuple of numpy ndarray
        """
        self.invalidation_flag = True
        axes = []
        for variable in self.variables:
            axes.append(variable.get_phase_grid())
        return np.meshgrid(*tuple(axes), indexing='ij')
        
    def create_charge_grid(self):
        """
        Creates a n-d grid of the charge variables, where n is the number of variables in the circuit, on which the circuit wavefunction, when transformed into charge representation, depends.
        :returns: tuple of numpy ndarray
        """
        self.invalidation_flag = True
        axes = []
        for variable in self.variables:
            axes.append(variable.get_charge_grid())
        return np.meshgrid(*tuple(axes), indexing='ij')
        
    def hamiltonian_phase_action(self, state_vector):
        """
        Implements the action of the hamiltonian on the state vector describing the system in phase representation.
        :param state_vector: wavefunction to act upon
        :type state_vector: ndarray
        :returns: wavefunction after action of the hamiltonian
        """
        psi = np.reshape(state_vector, self.charge_potential.shape)
        phi = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(psi)))
        Up = self.phase_potential.ravel()*state_vector
        Tp = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(self.charge_potential*phi))).ravel()
        return Up+Tp
    
    def capacitance_matrix(self, symbolic=False):
        """
        Calculates the linear capacitance matrix of the circuit with respect 
        to the circuit nodes from the capacitances between them.
        :returns: the capacitance matrix with respect to the nodes, where the rows and columns are sorted accoring to the order in which the nodes are in the nodes attribute.
        """
        if symbolic:
            capacitance_matrix = sympy.Matrix(np.zeros((len(self.nodes), len(self.nodes))))
        else:
            capacitance_matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for element in self.elements:
            if element.is_charge():
                element_node_ids = []
                for wire in self.wires:
                    if wire[0] == element.name:
                        for node_id, node in enumerate(self.nodes):
                            if wire[1] == node.name:
                                element_node_ids.append(node_id)
                if len(element_node_ids) != 2:
                    raise Exception('VariableError', 
                                    'Wrong number of ports on capacitance, expected 2, got {0}'.format(len(element_node_ids)))
                capacitance_matrix[element_node_ids[0], element_node_ids[0]] += element.get_capacitance()
                capacitance_matrix[element_node_ids[0], element_node_ids[1]] += -element.get_capacitance()
                capacitance_matrix[element_node_ids[1], element_node_ids[0]] += -element.get_capacitance()
                capacitance_matrix[element_node_ids[1], element_node_ids[1]] += element.get_capacitance()
        return capacitance_matrix
    
    def capacitance_matrix_variables(self, symbolic=False):
        """
        Calculates the capacitance matrix for the energy term of the qubit Lagrangian in the variable respresentation.
        """                        
        
        if symbolic:
            C = self.linear_coordinate_transform.T*self.capacitance_matrix(symbolic)*self.linear_coordinate_transform
            C = sympy.Matrix([sympy.nsimplify(sympy.ratsimp(x)) for x in C]).reshape(*(C.shape))
        else:
            C = np.einsum('ji,jk,kl->il', self.linear_coordinate_transform,self.capacitance_matrix(symbolic),self.linear_coordinate_transform)
        return C
    
    def capacitance_matrix_legendre_transform(self, symbolic=False):
        """
        Calculates the principle pivot transform of the capacitance matrix in variable representation with respect to "variables" as opposed to "parameters" for the Legendre transform
        """
        inverted_indices = [variable_id for variable_id, variable in enumerate(self.variables) if variable.variable_type=='variable' ]
        noninverted_indices = [variable_id for variable_id, variable in enumerate(self.variables) if variable.variable_type=='parameter' ]
        if symbolic:
            aii = self.capacitance_matrix_variables(symbolic)[inverted_indices, inverted_indices]
            ain = self.capacitance_matrix_variables(symbolic)[inverted_indices, noninverted_indices]
            ani = self.capacitance_matrix_variables(symbolic)[noninverted_indices, inverted_indices]
            # Ann = self.capacitance_matrix_variables(symbolic)[noninverted_indices, noninverted_indices]
            bii = aii.inv()
            bin = sympy.Matrix(-aii.inv()*ain)
            bni = sympy.Matrix(-ani*aii.inv())
            bnn = ani*aii.inv()*ain#-Ann
            B = sympy.Matrix(np.zeros(self.capacitance_matrix_variables(symbolic).shape))
        else:
            aii = self.capacitance_matrix_variables(symbolic)[tuple(np.meshgrid(inverted_indices, inverted_indices))].T
            ain = self.capacitance_matrix_variables(symbolic)[tuple(np.meshgrid(inverted_indices, noninverted_indices))].T
            ani = self.capacitance_matrix_variables(symbolic)[tuple(np.meshgrid(noninverted_indices, inverted_indices))].T
            # Ann = self.capacitance_matrix_variables(symbolic)[np.meshgrid(noninverted_indices, noninverted_indices)].T
            bii = np.linalg.inv(aii)
            bin = -np.dot(np.linalg.inv(aii), ain)
            bni = -np.dot(ani,np.linalg.inv(aii))
            bnn = np.einsum('ij,jk,kl->il', ani, np.linalg.inv(aii), ain)#-Ann
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
        if len(d1scheme)!=len(d2scheme):
            raise Exception('ValueError', 'd1scheme and d2scheme lengths are not equal')
        if n<3:
            raise Exception('ValueError', 'dscheme length is less than 3')
        if (n-1)%2>0:
            raise Exception('ValueError', 'dscheme length is even')
            
        self.ndiagonal_operator = np.zeros(tuple(n*np.ones((len(self.variables),), dtype=int))+self.grid_shape())
        slice_diagonal = [(n-1)/2 for v in self.variables]+[slice(0, v.pt_count, 1) for v in self.variables]

        ECmat = -0.5*self.capacitance_matrix_legendre_transform()
        # d^2/dxi^2 type elements (C*_ii)
        for i in range(len(self.variables)):
            EC = ECmat[i,i]
            for column_id in range(n):
                slice_column = list(slice_diagonal)
                slice_column[i] = column_id
                self.ndiagonal_operator[slice_column] += EC/(self.variables[i].get_phase_step()**2)*d2scheme[column_id]
        # d^2/dxidxj type elements (C*_ij)
        for i in range(len(self.variables)):
            nondiagonal = (x for x in range(len(self.variables)) if x!=i)
            for j in nondiagonal:
                EC = ECmat[i,j]
                for column_id_i in range(n):
                    for column_id_j in range(n):
                        slice_column = list(slice_diagonal)
                        slice_column[i] = column_id_i
                        slice_column[j] = column_id_j
                        self.ndiagonal_operator[slice_column] +=  EC/(self.variables[i].get_phase_step()*self.variables[j].get_phase_step())*(d1scheme[column_id_i]*d1scheme[column_id_j])
                        
        self.ndiagonal_operator[slice_diagonal] += self.phase_potential
        
        self.hamiltonian_ndiagonal = LinearOperator((np.prod(self.grid_shape()), np.prod(self.grid_shape())), matvec=self.ndiagonal_operator_action)
        return self.ndiagonal_operator
    
    def ndiagonal_operator_action(self, psi):
        diagonal_shape = tuple([1]*len(self.variables))+self.grid_shape()
        psi = np.reshape(psi, diagonal_shape)
        action = self.ndiagonal_operator*psi
        ndiagonal_columns = np.meshgrid(*tuple([range(self.ndiagonal_operator.shape[v_id]) for v_id in range(len(self.variables))]), indexing='ij')
        ndiagonal_columns = np.reshape(ndiagonal_columns, (len(self.variables), np.prod(self.ndiagonal_operator.shape[0:len(self.variables)])))
        ndiagonal_shifts = np.meshgrid(*tuple([np.linspace(
                        -(self.ndiagonal_operator.shape[v_id]-1)/2, 
                         (self.ndiagonal_operator.shape[v_id]-1)/2, 
                          self.ndiagonal_operator.shape[v_id], dtype=int) for v_id in range(len(self.variables))]), indexing='ij')
        ndiagonal_shifts = np.reshape(ndiagonal_shifts, ndiagonal_columns.shape)
        
        result = np.zeros(self.grid_shape(), dtype=np.complex)
        for i in range(np.prod(self.ndiagonal_operator.shape[0:len(self.variables)])):
            psii = action[tuple(ndiagonal_columns[:, i])+tuple([slice(None, None, None)]*len(self.variables))]
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
        phase_grid = self.create_phase_grid()
        self.phase_potential = np.zeros(grid_shape)
        for element in self.elements:
            element_node_ids = []
            for wire in self.wires:
                if wire[0] == element.name:
                    for node_id, node in enumerate(self.nodes):
                        if wire[1] == node.name:
                            element_node_ids.append(node_id)
            phase_grid = np.reshape(np.asarray(phase_grid), (len(self.variables), grid_size))
            node_phases  = np.einsum('ij,jk->ik', self.linear_coordinate_transform, phase_grid)[element_node_ids, :]
            node_phases = np.reshape(node_phases, (len(element_node_ids),)+grid_shape) 
            if element.is_phase():
                self.phase_potential += element.energy_term(node_phases=node_phases, node_charges=np.zeros(node_phases.shape))
        return self.phase_potential
    
    def calculate_charge_potential(self):
        """
        Calculates the potential landspace of the circuit charge-dependent energy in charge representation. 
        :returns: the charge potential landscape on the wavefunction grid.
        """
        grid_shape = self.grid_shape()
        grid_size = np.prod(grid_shape)
        charge_grid = np.reshape(np.asarray(self.create_charge_grid()), (len(self.variables), grid_size))
        ECmat = 0.5*self.capacitance_matrix_legendre_transform()
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
        wavefunctions = np.reshape(wavefunctions, self.charge_potential.shape+(num_states,))
        return energies, wavefunctions
    
    def symbolic_lagrangian(self):
        variable_phase_symbols = []
        variable_voltage_symbols = []
        for variable_id, variable in enumerate(self.variables):
            variable.phase_symbol = sympy.Symbol(variable.name)
            variable.voltage_symbol = sympy.Symbol('\\partial_t'+variable.name)
            variable_phase_symbols.append(variable.phase_symbol)
            variable_voltage_symbols.append(variable.voltage_symbol)
        variable_phase_symbols = sympy.Matrix(variable_phase_symbols)
        variable_voltage_symbols = sympy.Matrix(variable_voltage_symbols)
        node_phase_symbols = self.linear_coordinate_transform*variable_phase_symbols
        node_voltage_symbols = self.linear_coordinate_transform*variable_voltage_symbols
        for node_id, node in enumerate(self.nodes):
            node.phase_symbol = node_phase_symbols[node_id]
            node.voltage_symbol = node_voltage_symbols[node_id]
        kinetic_energy = sympy.nsimplify((0.5*node_voltage_symbols.T*self.capacitance_matrix(symbolic=True)*node_voltage_symbols)[0, 0])
        potential_energy = 0
        for element in self.elements:
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
            if variable.variable_type=='variable':
                variable.charge_symbol = -sympy.I*sympy.Symbol('\\partial_{'+variable.name+'}')
            else:
                variable.charge_symbol = sympy.Symbol('\\partial_t'+variable.name)
            variable_phase_symbols.append(variable.phase_symbol)
            variable_charge_symbols.append(variable.charge_symbol)
        variable_phase_symbols = sympy.Matrix(variable_phase_symbols)
        variable_charge_symbols = sympy.Matrix(variable_charge_symbols)

        node_phase_symbols = self.linear_coordinate_transform*variable_phase_symbols
        for node_id, node in enumerate(self.nodes):
            node.phase_symbol = node_phase_symbols[node_id]
        kinetic_energy = 0.5*sympy.nsimplify((variable_charge_symbols.T * self.capacitance_matrix_legendre_transform(symbolic=True) * variable_charge_symbols)[0, 0])
        potential_energy = 0
        for element in self.elements:
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