# hilbert_space.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

# TODO: Clean up this doc and then branch for the zombie version

# TODO: Add typing for all methods

import functools
import warnings
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, List, Union

import numpy as np
import qutip as qt
from numpy import ndarray
from qutip.qobj import Qobj
from scipy.sparse.csc import csc_matrix
from scipy.sparse.dia import dia_matrix
from collections import namedtuple

import scqubits.core.central_dispatch as dispatch
import scqubits.core.descriptors as descriptors
import scqubits.core.harmonic_osc as osc
import scqubits.core.spec_lookup as spec_lookup
import scqubits.core.storage as storage
import scqubits.io_utils.fileio_qutip
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.settings as settings
import scqubits.ui.hspace_widget
import scqubits.utils.cpu_switch as cpu_switch
import scqubits.utils.misc as utils
import scqubits.utils.spectrum_utils as spec_utils
from scqubits.core.harmonic_osc import Oscillator
from scqubits.core.qubit_base import QubitBaseClass
from scqubits.core.storage import SpectrumData
from scqubits.io_utils.fileio_qutip import QutipEigenstates

if settings.IN_IPYTHON:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

QuantumSys = Union[QubitBaseClass, Oscillator]
Op = namedtuple("Op", ['operator', 'subsystem'])


class InteractionTerm(dispatch.DispatchClient, serializers.Serializable):
    # TODO: GUI needs to be fixed before pushing
    """
    Class for specifying a term in the interaction Hamiltonian of a composite Hilbert space, and constructing
    the Hamiltonian in qutip.Qobj format. The expected form of the interaction term is of two possible types:
    1. V = g A B C ..., where A, B, C... are Hermitian operators in subsystems in subsystem_list,
    2. V = g A B C... + h.c., where A, B, C... may be non-Hermitian
# TODO: Fix subsystem list and operator list definitions

    Parameters
    ----------
    g_strength:
        coefficient parametrizing the interaction strength.
    operator_list:
        list of tuples of operators involved in the interaction paired with their subsystems eg. (operator, subsystem).
    subsystem_list:
        list of subsystems.
    add_hc:
        If set to True, the interaction Hamiltonian is of type 2, and the Hermitian conjugate is added.
    """
    g_strength = descriptors.WatchedProperty('INTERACTIONTERM_UPDATE')
    operator_list = descriptors.WatchedProperty('INTERACTIONTERM_UPDATE')
    subsystem_list = descriptors.WatchedProperty('INTERACTIONTERM_UPDATE')
    add_hc = descriptors.WatchedProperty('INTERACTIONTERM_UPDATE')


    def __init__(self,
                 g_strength: Union[float, complex],
                 operator_list: List[Tuple[Union[np.ndarray, qt.Qobj, csc_matrix, dia_matrix, str], QuantumSys]],
                 subsystem_list: List[QuantumSys],
                 add_hc: bool = False
                 ) -> None:
        self.g_strength = g_strength
        self.operator_list = [Op._make(operator) for operator in operator_list]
        self.subsystem_list = subsystem_list
        self.add_hc = add_hc
        hamiltonian = g_strength
        qoperator_list = self.idwrap(operator_list, subsystem_list)
        for op in qoperator_list:
            hamiltonian *= op
        if add_hc:
            self.hamiltonian = hamiltonian + hamiltonian.dag()
        else:
            self.hamiltonian = hamiltonian

    def __repr__(self) -> str:
        init_dict = {name: getattr(self, name) for name in self._init_params}
        return type(self).__name__ + f'(**{init_dict!r})'

    def __str__(self) -> str:
        output = type(self).__name__.upper() + '\n ———— PARAMETERS ————'
        for param_name in self._init_params:
            output += '\n' + str(param_name) + '\t: ' + str(getattr(self, param_name))
        return output + '\n'

    def __getitem__(self, index: int) -> QuantumSys:
        return self.subsystem_list[index]

    def convert_operators(self, op_list: list) -> list:
        new_operators = [spec_utils.convert_operator_to_qobj(item.operator, item.subsystem,
                                                             op_in_eigenbasis=False, evecs=None)
                         for item in op_list]
        return new_operators

    def idwrap(self, operator_list, subsystem_list):
        id_wrapped_operators = [spec_utils.identity_wrap(item.operator, item.subsystem, subsystem_list)
                                for item in operator_list]
        return id_wrapped_operators

class InteractionTermStr(dispatch.DispatchClient, serializers.Serializable):
    """
    Class for specifying a term in the interaction Hamiltonian of a composite Hilbert space, and constructing
    the Hamiltonian in qutip.Qobj format. The form of the interaction is defined using the str_expression string.
    Each operator must be hermitian, unless add_hc = True in which case each operator my be non-hermitian.
    Acceptable functions inside of str_expression string include: cos(), sin(), dag(), ()), exp(), sqrt(), trans(),
    cosm(), sinm(), expm(), and sqrtm() along with other operators included in python.

    Parameters
    ----------
    str_expression:
        string that defines the interaction.
    operator_dict:
        dictonary of names and tuples of operators and subsystems eg. {name: (operator, subsystem)}.
    subsystem_list:
        list of all subsystems.
    add_hc:
        If set to True, the interaction Hamiltonian is of type 2, and the Hermitian conjugate is added.
    """
    str_expression = descriptors.WatchedProperty('INTERACTIONTERM_UPDATE')
    operator_list = descriptors.WatchedProperty('INTERACTIONTERM_UPDATE')
    subsystem_list = descriptors.WatchedProperty('INTERACTIONTERM_UPDATE')
    add_hc = descriptors.WatchedProperty('INTERACTIONTERM_UPDATE')

    qutip_dict = {
        'cosm(': 'Qobj.cosm(',
        'expm(': 'Qobj.expm(',
        'sinm(': 'Qobj.sinm(',
        'sqrtm(': 'Qobj.sqrtm(',
        'cos(': 'Qobj.cosm(',
        'dag(': 'Qobj.dag(',
        'conj(': 'Qobj.conj(',
        'exp(': 'Qobj.expm(',
        'sin(': 'Qobj.sinm(',
        'sqrt(': 'Qobj.sqrtm(',
        'trans(': 'Qobj.trans('
    }

    def __init__(self,
                 str_expression: str,
                 operator_dict: Dict[str, Tuple[Union[np.ndarray, qt.Qobj, csc_matrix, dia_matrix, str], QuantumSys]],
                 subsystem_list: List[QuantumSys],
                 add_hc: bool = False
                 ) -> None:
        self.str_expression = str_expression
        self.operator_dict = {key: Op._make(value) for (key, value) in operator_dict.items()}
        self.subsystem_list = subsystem_list
        self.add_hc = add_hc
        qoperator_dict = self.id_wrap(operator_dict, subsystem_list)
        self.add_to_variables(qoperator_dict)
        hamiltonian = self.run_string_code(str_expression)
        if not add_hc:
            self.hamiltonian = hamiltonian
        else:
            self.hamiltonian = hamiltonian + hamiltonian.dag()

    def __repr__(self) -> str:
        init_dict = {name: getattr(self, name) for name in self._init_params}
        return type(self).__name__ + f'(**{init_dict!r})'

    def __str__(self) -> str:
        output = type(self).__name__.upper() + '\n ———— PARAMETERS ————'
        for param_name in self._init_params:
            output += '\n' + str(param_name) + '\t: ' + str(getattr(self, param_name))
        return output + '\n'

    def add_to_variables(self, op_dict: dict) -> None:
        for (key, value) in op_dict.items():
            globals()[key] = value

    def replace_string(self, string: str) -> str:
        for item, value in self.qutip_dict.items():
            if item in string:
                string = string.replace(item, value)
        return string

    def run_string_code(self, string: str) -> Qobj:
        string = self.replace_string(string)
        answer = eval(string)
        return answer

    def id_wrap(self, op_dict: dict, subsys_list: list):
        new_operators = {key: spec_utils.identity_wrap(value[0], value[1], subsys_list)
                         for (key, value) in op_dict.items()}
        return new_operators


class HilbertSpace(dispatch.DispatchClient, serializers.Serializable):
    """Class holding information about the full Hilbert space, usually composed of multiple subsys_list.
    The class provides methods to turn subsystem operators into operators acting on the full Hilbert space, and
    establishes the interface to qutip. Returned operators are of the `qutip.Qobj` type. The class also provides methods
    for obtaining eigenvalues, absorption and emission spectra as a function of an external parameter.
    """
    osc_subsys_list = descriptors.ReadOnlyProperty()
    qbt_subsys_list = descriptors.ReadOnlyProperty()
    lookup = descriptors.ReadOnlyProperty()
    interaction_list = descriptors.WatchedProperty('INTERACTIONLIST_UPDATE')

    def __init__(self,
                 subsystem_list: List[QuantumSys],
                 interaction_list: List[InteractionTerm] = None
                 ) -> None:
        self._subsystems: Tuple[QuantumSys, ...] = tuple(subsystem_list)
        self.subsys_list = subsystem_list
        if interaction_list:
            self.interaction_list = tuple(interaction_list)
        else:
            self.interaction_list = []

        self._lookup: Optional[spec_lookup.SpectrumLookup] = None
        self._osc_subsys_list = [(index, subsys) for (index, subsys) in enumerate(self)
                                 if isinstance(subsys, osc.Oscillator)]
        self._qbt_subsys_list = [(index, subsys) for (index, subsys) in enumerate(self)
                                 if not isinstance(subsys, osc.Oscillator)]

        dispatch.CENTRAL_DISPATCH.register('QUANTUMSYSTEM_UPDATE', self)
        dispatch.CENTRAL_DISPATCH.register('INTERACTIONTERM_UPDATE', self)
        dispatch.CENTRAL_DISPATCH.register('INTERACTIONLIST_UPDATE', self)

    @classmethod
    def create(cls) -> 'HilbertSpace':
        hilbertspace = cls([])
        scqubits.ui.hspace_widget.create_hilbertspace_widget(hilbertspace.__init__)  # type: ignore
        return hilbertspace

    def __getitem__(self, index: int) -> QuantumSys:
        return self._subsystems[index]

    def __iter__(self) -> Iterator[QuantumSys]:
        return iter(self._subsystems)

    def __repr__(self) -> str:
        init_dict = self.get_initdata()
        return type(self).__name__ + f'(**{init_dict!r})'

    def __str__(self) -> str:
        output = '====== HilbertSpace object ======\n'
        for subsystem in self:
            output += '\n' + str(subsystem) + '\n'
        if self.interaction_list:
            for interaction_term in self.interaction_list:
                output += '\n' + str(interaction_term) + '\n'
        return output

    def index(self, item: QuantumSys) -> int:
        return self._subsystems.index(item)

    def get_initdata(self) -> Dict[str, Any]:
        """Returns dict appropriate for creating/initializing a new HilbertSpace object.
        """
        return {'subsystem_list': self._subsystems, 'interaction_list': self.interaction_list}

    def receive(self,
                event: str,
                sender: Any,
                **kwargs
                ) -> None:
        if self._lookup is not None:
            if event == 'QUANTUMSYSTEM_UPDATE' and sender in self:
                self.broadcast('HILBERTSPACE_UPDATE')
                self._lookup._out_of_sync = True
            elif event == 'INTERACTIONTERM_UPDATE' and sender in self.interaction_list:
                self.broadcast('HILBERTSPACE_UPDATE')
                self._lookup._out_of_sync = True
            elif event == 'INTERACTIONLIST_UPDATE' and sender is self:
                self.broadcast('HILBERTSPACE_UPDATE')
                self._lookup._out_of_sync = True

    @property
    def subsystem_list(self) -> Tuple[QuantumSys, ...]:
        return self._subsystems

    @property
    def subsystem_dims(self) -> List[int]:
        """Returns list of the Hilbert space dimensions of each subsystem"""
        return [subsystem.truncated_dim for subsystem in self]

    @property
    def dimension(self) -> int:
        """Returns total dimension of joint Hilbert space"""
        return np.prod(np.asarray(self.subsystem_dims))

    @property
    def subsystem_count(self) -> int:
        """Returns number of subsys_list composing the joint Hilbert space"""
        return len(self._subsystems)

    def generate_lookup(self) -> None:
        bare_specdata_list = []
        for index, subsys in enumerate(self):
            evals, evecs = subsys.eigensys(evals_count=subsys.truncated_dim)
            bare_specdata_list.append(storage.SpectrumData(energy_table=[evals],
                                                           state_table=[evecs],
                                                           system_params=subsys.get_initdata()))

        evals, evecs = self.eigensys(evals_count=self.dimension)
        dressed_specdata = storage.SpectrumData(energy_table=[evals],
                                                state_table=[evecs],
                                                system_params=self.get_initdata())
        self._lookup = spec_lookup.SpectrumLookup(self,
                                                  bare_specdata_list=bare_specdata_list,
                                                  dressed_specdata=dressed_specdata)

    def eigenvals(self, evals_count: int = 6) -> ndarray:
        """Calculates eigenvalues of the full Hamiltonian using `qutip.Qob.eigenenergies()`.

        Parameters
        ----------
        evals_count:
            number of desired eigenvalues/eigenstates
        """
        hamiltonian_mat = self.hamiltonian()
        return hamiltonian_mat.eigenenergies(eigvals=evals_count)

    def eigensys(self, evals_count: int = 6) -> Tuple[ndarray, QutipEigenstates]:
        """Calculates eigenvalues and eigenvectore of the full Hamiltonian using `qutip.Qob.eigenstates()`.

        Parameters
        ----------
        evals_count:
            number of desired eigenvalues/eigenstates

        Returns
        -------
            eigenvalues and eigenvectors
        """
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = hamiltonian_mat.eigenstates(eigvals=evals_count)
        evecs = evecs.view(scqubits.io_utils.fileio_qutip.QutipEigenstates)
        return evals, evecs

    def diag_operator(self, diag_elements: ndarray, subsystem: QuantumSys) -> Qobj:
        """For given diagonal elements of a diagonal operator in `subsystem`, return the `Qobj` operator for the
        full Hilbert space (perform wrapping in identities for other subsys_list).

        Parameters
        ----------
        diag_elements:
            diagonal elements of subsystem diagonal operator
        subsystem:
            subsystem where diagonal operator is defined
        """
        dim = subsystem.truncated_dim
        index = range(dim)
        diag_matrix = np.zeros((dim, dim), dtype=np.float_)
        diag_matrix[index, index] = diag_elements
        return spec_utils.identity_wrap(diag_matrix, subsystem, self.subsys_list)

    def diag_hamiltonian(self, subsystem: QuantumSys, evals: ndarray = None) -> Qobj:
        """Returns a `qutip.Qobj` which has the eigenenergies of the object `subsystem` on the diagonal.

        Parameters
        ----------
        subsystem:
            Subsystem for which the Hamiltonian is to be provided.
        evals:
            Eigenenergies can be provided as `evals`; otherwise, they are calculated.
        """
        evals_count = subsystem.truncated_dim
        if evals is None:
            evals = subsystem.eigenvals(evals_count=evals_count)
        diag_qt_op = qt.Qobj(inpt=np.diagflat(evals[0:evals_count]))
        return spec_utils.identity_wrap(diag_qt_op, subsystem, self.subsys_list)

    def hubbard_operator(self, j: int, k: int, subsystem: QuantumSys) -> Qobj:
        """Hubbard operator :math:`|j\\rangle\\langle k|` for system `subsystem`

        Parameters
        ----------
        j,k:
            eigenstate indices for Hubbard operator
        subsystem:
            subsystem in which Hubbard operator acts
        """
        dim = subsystem.truncated_dim
        operator = (qt.states.basis(dim, j) * qt.states.basis(dim, k).dag())
        return spec_utils.identity_wrap(operator, subsystem, self.subsys_list)

    def annihilate(self, subsystem: QuantumSys) -> Qobj:
        """Annihilation operator a for `subsystem`

        Parameters
        ----------
        subsystem:
            specifies subsystem in which annihilation operator acts
        """
        dim = subsystem.truncated_dim
        operator = (qt.destroy(dim))
        return spec_utils.identity_wrap(operator, subsystem, self.subsys_list)

    def get_subsys_index(self, subsys: QuantumSys) -> int:
        """
        Return the index of the given subsystem in the HilbertSpace.
        """
        return self.index(subsys)

    def bare_hamiltonian(self) -> Qobj:
        """
        Returns
        -------
            composite Hamiltonian composed of bare Hamiltonians of subsys_list independent of the external parameter
        """
        bare_hamiltonian = 0
        for subsys in self:
            evals = subsys.eigenvals(evals_count=subsys.truncated_dim)
            bare_hamiltonian += self.diag_hamiltonian(subsys, evals)
        return bare_hamiltonian

    def get_bare_hamiltonian(self) -> Qobj:
        """Deprecated, use `bare_hamiltonian()` instead."""
        warnings.warn('get_bare_hamiltonian() is deprecated, use bare_hamiltonian() instead', FutureWarning)
        return self.bare_hamiltonian()

    def hamiltonian(self) -> Qobj:
        """

        Returns
        -------
            Hamiltonian of the composite system, including the interaction between components
        """
        return self.bare_hamiltonian() + self.interaction_hamiltonian()

    def get_hamiltonian(self):
        """Deprecated, use `hamiltonian()` instead."""
        return self.hamiltonian()

    def interaction_hamiltonian(self) -> Qobj:
        """
        Returns
        -------
            interaction Hamiltonian
        """
        if not self.interaction_list:
            return 0

        operator_list = [term.hamiltonian if isinstance(term, InteractionTerm)
                         else term for term in self.interaction_list]
        hamiltonian = sum(operator_list)
        return hamiltonian

    def _esys_for_paramval(self,
                           paramval: float,
                           update_hilbertspace: Callable,
                           evals_count: int
                           ) -> Tuple[ndarray, QutipEigenstates]:
        update_hilbertspace(paramval)
        return self.eigensys(evals_count)

    def _evals_for_paramval(self,
                            paramval: float,
                            update_hilbertspace: Callable,
                            evals_count: int
                            ) -> ndarray:
        update_hilbertspace(paramval)
        return self.eigenvals(evals_count)

    def get_spectrum_vs_paramvals(self,
                                  param_vals: ndarray,
                                  update_hilbertspace: Callable,
                                  evals_count: int = 10,
                                  get_eigenstates: bool = False,
                                  param_name: str = "external_parameter",
                                  num_cpus: int = settings.NUM_CPUS
                                  ) -> SpectrumData:
        """Return eigenvalues (and optionally eigenstates) of the full Hamiltonian as a function of a parameter.
        Parameter values are specified as a list or array in `param_vals`. The Hamiltonian `hamiltonian_func`
        must be a function of that particular parameter, and is expected to internally set subsystem parameters.
        If a `filename` string is provided, then eigenvalue data is written to that file.

        Parameters
        ----------
        param_vals:
            array of parameter values
        update_hilbertspace:
            update_hilbertspace(param_val) specifies how a change in the external parameter affects
            the Hilbert space components
        evals_count:
            number of desired energy levels (default value = 10)
        get_eigenstates:
            set to true if eigenstates should be returned as well (default value = False)
        param_name:
            name for the parameter that is varied in `param_vals` (default value = "external_parameter")
        num_cpus:
            number of cores to be used for computation (default value: settings.NUM_CPUS)
        """
        target_map = cpu_switch.get_map_method(num_cpus)
        if get_eigenstates:
            func = functools.partial(self._esys_for_paramval, update_hilbertspace=update_hilbertspace,
                                     evals_count=evals_count)
            with utils.InfoBar("Parallel computation of eigenvalues [num_cpus={}]".format(num_cpus), num_cpus):
                eigensystem_mapdata = list(target_map(func, tqdm(param_vals, desc='Spectral data', leave=False,
                                                                 disable=(num_cpus > 1))))
            eigenvalue_table, eigenstate_table = spec_utils.recast_esys_mapdata(eigensystem_mapdata)
        else:
            func = functools.partial(self._evals_for_paramval, update_hilbertspace=update_hilbertspace,
                                     evals_count=evals_count)
            with utils.InfoBar("Parallel computation of eigensystems [num_cpus={}]".format(num_cpus), num_cpus):
                eigenvalue_table = list(target_map(func, tqdm(param_vals, desc='Spectral data', leave=False,
                                                              disable=(num_cpus > 1))))
            eigenvalue_table = np.asarray(eigenvalue_table)
            eigenstate_table = None  # type: ignore

        return storage.SpectrumData(eigenvalue_table,
                                    self.get_initdata(),
                                    param_name,
                                    param_vals,
                                    state_table=eigenstate_table)
