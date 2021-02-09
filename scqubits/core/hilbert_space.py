# hilbert_space.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################


import functools
import importlib
import warnings
import weakref

from collections import namedtuple
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import qutip as qt

from numpy import ndarray
from qutip.qobj import Qobj
from scipy.sparse import csc_matrix, dia_matrix

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

if TYPE_CHECKING:
    from scqubits.io_utils.fileio import IOData


QuantumSys = Union[QubitBaseClass, Oscillator]
SubsysOperator = namedtuple("SubsysOperator", ["operator", "subsystem"])


class InteractionTermLegacy(dispatch.DispatchClient, serializers.Serializable):
    """
    Deprecated, will not work in future versions. Please look into InteractionTerm
    instead.

    Class for specifying a term in the interaction Hamiltonian of a composite Hilbert
    space, and constructing the Hamiltonian in qutip.Qobj format. The expected form
    of the interaction term is of two possible types: 1. V = g A B, where A,
    B are Hermitean operators in two specified subsys_list, 2. V = g A B + h.c.,
    where A, B may be non-Hermitean

    Parameters
    ----------
    g_strength:
        coefficient parametrizing the interaction strength
    hilbertspace:
        specifies the Hilbert space components
    subsys1, subsys2:
        the two subsys_list involved in the interaction
    op1, op2:
        names of operators in the two subsys_list
    add_hc:
        If set to True, the interaction Hamiltonian is of type 2, and the Hermitean
        conjugate is added.
    """

    g_strength = descriptors.WatchedProperty("INTERACTIONTERM_UPDATE")
    subsys1 = descriptors.WatchedProperty("INTERACTIONTERM_UPDATE")
    subsys2 = descriptors.WatchedProperty("INTERACTIONTERM_UPDATE")
    op1 = descriptors.WatchedProperty("INTERACTIONTERM_UPDATE")
    op2 = descriptors.WatchedProperty("INTERACTIONTERM_UPDATE")

    def __init__(
        self,
        g_strength: Union[float, complex],
        subsys1: QuantumSys,
        op1: Union[str, ndarray, csc_matrix, dia_matrix],
        subsys2: QuantumSys,
        op2: Union[str, ndarray, csc_matrix, dia_matrix],
        add_hc: bool = False,
        hilbertspace: "HilbertSpace" = None,
    ) -> None:
        warnings.warn(
            "Future use of InteractionTerm will require arguments in a different "
            "format, see help(InteractionTerm).",
            FutureWarning,
        )
        if hilbertspace:
            warnings.warn(
                "`hilbertspace` is no longer a parameter for initializing "
                "an InteractionTerm object.",
                FutureWarning,
            )
        self.g_strength = g_strength
        self.subsys1 = subsys1
        self.op1 = op1
        self.subsys2 = subsys2
        self.op2 = op2
        self.add_hc = add_hc

        self._init_params.remove("hilbertspace")

    def __repr__(self) -> str:
        init_dict = {name: getattr(self, name) for name in self._init_params}
        return type(self).__name__ + f"(**{init_dict!r})"

    def __str__(self) -> str:
        indent_length = 25
        name_prepend = "InteractionTermLegacy".ljust(indent_length, "-") + "|\n"

        output = ""
        for param_name in self._init_params:
            param_content = getattr(self, param_name).__repr__()
            param_content = param_content.strip("\n")
            if len(param_content) > 50:
                param_content = param_content[:50]
                param_content += " ..."

            output += "{0}| {1}: {2}\n".format(
                " " * indent_length, str(param_name), param_content
            )
        return name_prepend + output


class InteractionTerm(dispatch.DispatchClient, serializers.Serializable):
    """
    Class for specifying a term in the interaction Hamiltonian of a composite Hilbert
    space, and constructing the Hamiltonian in qutip.Qobj format. The expected form
    of the interaction term is of two possible types: 1. V = g A B C ..., where A, B,
    C... are Hermitian operators in subsystems in subsystem_list, 2. V = g A B C... +
    h.c., where A, B, C... may be non-Hermitian

    Parameters
    ----------
    g_strength:
        coefficient parametrizing the interaction strength.
    operator_list:
        list of tuples of operators and their corresponding subsystems involved in the
        interaction eg. (operator, subsystem).
    subsystem_list:
        list of all subsystems relevant to the Hilbert space.
    add_hc:
        If set to True, the interaction Hamiltonian is of type 2, and the Hermitian
        conjugate is added.
    """

    g_strength = descriptors.WatchedProperty("INTERACTIONTERM_UPDATE")
    operator_list = descriptors.WatchedProperty("INTERACTIONTERM_UPDATE")
    subsystem_list = descriptors.WatchedProperty("INTERACTIONTERM_UPDATE")
    add_hc = descriptors.WatchedProperty("INTERACTIONTERM_UPDATE")

    def __new__(
        cls,
        g_strength: Union[float, complex],
        operator_list: List[
            Tuple[Union[np.ndarray, qt.Qobj, csc_matrix, dia_matrix, str], QuantumSys]
        ] = None,
        subsystem_list: List[QuantumSys] = None,
        subsys1: QuantumSys = None,
        op1: Union[str, ndarray, csc_matrix, dia_matrix] = None,
        subsys2: QuantumSys = None,
        op2: Union[str, ndarray, csc_matrix, dia_matrix] = None,
        hilbertspace: "HilbertSpace" = None,
        add_hc: bool = False,
    ) -> Union["InteractionTerm", InteractionTermLegacy]:
        if subsys1:
            return InteractionTermLegacy(
                g_strength=g_strength,
                op1=op1,
                subsys1=subsys1,
                op2=op2,
                subsys2=subsys2,
                hilbertspace=hilbertspace,
                add_hc=add_hc,
            )
        else:
            return super().__new__(cls)

    def __init__(
        self,
        g_strength: Union[float, complex],
        operator_list: List[
            Tuple[Union[np.ndarray, qt.Qobj, csc_matrix, dia_matrix, str], QuantumSys]
        ],
        subsystem_list: List[QuantumSys],
        add_hc: bool = False,
    ) -> None:
        self.g_strength = g_strength
        self.operator_list = [
            (SubsysOperator(operator[0], operator[1])) for operator in operator_list
        ]
        self.subsystem_list = subsystem_list
        self.add_hc = add_hc
        self.qoperator_list = self.id_wrap_all_ops(self.operator_list, subsystem_list)

    def __repr__(self) -> str:
        init_dict = {name: getattr(self, name) for name in self._init_params}
        return type(self).__name__ + f"(**{init_dict!r})"

    def __str__(self) -> str:
        indent_length = 25
        name_prepend = "InteractionTerm".ljust(indent_length, "-") + "|\n"

        output = ""
        for param_name in self._init_params:
            param_content = getattr(self, param_name).__repr__()
            param_content = param_content.strip("\n")
            if len(param_content) > 50:
                param_content = param_content[:50]
                param_content += " ..."

            output += "{0}| {1}: {2}\n".format(
                " " * indent_length, str(param_name), param_content
            )
        return name_prepend + output

    def __getitem__(self, index: int) -> QuantumSys:
        return self.subsystem_list[index]

    def hamiltonian(self):
        hamiltonian = self.g_strength
        for op in self.qoperator_list:
            hamiltonian *= op
        if self.add_hc:
            return hamiltonian + hamiltonian.dag()
        else:
            return hamiltonian

    @staticmethod
    def convert_operators(op_list: list) -> list:
        new_operators = [
            spec_utils.convert_operator_to_qobj(
                item.operator, item.subsystem, op_in_eigenbasis=False, evecs=None
            )
            for item in op_list
        ]
        return new_operators

    @staticmethod
    def id_wrap_all_ops(operator_list: list, subsystem_list: list) -> list:
        id_wrapped_operators = [
            spec_utils.identity_wrap(item.operator, item.subsystem, subsystem_list)
            for item in operator_list
        ]
        return id_wrapped_operators


class InteractionTermStr(dispatch.DispatchClient, serializers.Serializable):
    """
    Class for specifying a term in the interaction Hamiltonian of a composite Hilbert
    space, and constructing the Hamiltonian in qutip.Qobj format. The form of the
    interaction is defined using the str_expression string. Each operator must be
    hermitian, unless add_hc = True in which case each operator my be non-hermitian.
    Acceptable functions inside of str_expression string include: cos(), sin(),
    dag(), conj(), exp(), sqrt(), trans(), cosm(), sinm(), expm(), and sqrtm() along
    with other operators included in python.

    Parameters
    ----------
    str_expression:
        string that defines the interaction.
    operator_dict:
        dictionary of names and tuples of operators and subsystems
        eg. {name: (operator, subsystem)}.
    subsystem_list:
        list of all subsystems.
    add_hc:
        If set to True, the interaction Hamiltonian is of type 2, and the Hermitian
        conjugate is added.
    """

    str_expression = descriptors.WatchedProperty("INTERACTIONTERM_UPDATE")
    operator_list = descriptors.WatchedProperty("INTERACTIONTERM_UPDATE")
    subsystem_list = descriptors.WatchedProperty("INTERACTIONTERM_UPDATE")
    add_hc = descriptors.WatchedProperty("INTERACTIONTERM_UPDATE")

    def __init__(
        self,
        str_expression: str,
        operator_dict: Dict[
            str,
            Tuple[Union[np.ndarray, qt.Qobj, csc_matrix, dia_matrix, str], QuantumSys],
        ],
        subsystem_list: List[QuantumSys],
        add_hc: bool = False,
    ) -> None:
        self.qutip_dict = {
            "cosm(": "Qobj.cosm(",
            "expm(": "Qobj.expm(",
            "sinm(": "Qobj.sinm(",
            "sqrtm(": "Qobj.sqrtm(",
            "cos(": "Qobj.cosm(",
            "dag(": "Qobj.dag(",
            "conj(": "Qobj.conj(",
            "exp(": "Qobj.expm(",
            "sin(": "Qobj.sinm(",
            "sqrt(": "Qobj.sqrtm(",
            "trans(": "Qobj.trans(",
        }
        self.str_expression = str_expression
        self.operator_dict = {
            key: (SubsysOperator(value[0], value[1]))
            for (key, value) in operator_dict.items()
        }
        self.subsystem_list = subsystem_list
        self.add_hc = add_hc
        self.qoperator_dict = self.id_wrap_all_ops(self.operator_dict, subsystem_list)
        # TODO: change add to variables
        self.add_to_variables(self.qoperator_dict)

    def __repr__(self) -> str:
        init_dict = {name: getattr(self, name) for name in self._init_params}
        return type(self).__name__ + f"(**{init_dict!r})"

    def __str__(self) -> str:
        indent_length = 25
        name_prepend = "InteractionTermStr".ljust(indent_length, "-") + "|\n"

        output = ""
        for param_name in self._init_params:
            param_content = getattr(self, param_name).__repr__()
            param_content = param_content.strip("\n")
            if len(param_content) > 50:
                param_content = param_content[:50]
                param_content += " ..."

            output += "{0}| {1}: {2}\n".format(
                " " * indent_length, str(param_name), param_content
            )
        return name_prepend + output

    def add_to_variables(self, op_dict: dict) -> None:
        for (key, value) in op_dict.items():
            self.qutip_dict[key] = "_op_dict['" + key + "']"

    def replace_string(self, string: str) -> str:
        for item, value in self.qutip_dict.items():
            if item in string:
                string = string.replace(item, value)
        return string

    def run_string_code(self, string: str, op_dict: dict) -> Qobj:
        main = importlib.import_module("__main__")
        string = self.replace_string(string)
        main.__dict__["_op_dict"] = op_dict
        answer = eval(string, main.__dict__)
        return answer

    @staticmethod
    def id_wrap_all_ops(op_dict: dict, subsys_list: list) -> dict:
        new_operators = {
            key: spec_utils.identity_wrap(value.operator, value.subsystem, subsys_list)
            for (key, value) in op_dict.items()
        }
        return new_operators

    def hamiltonian(self):
        hamiltonian = self.run_string_code(self.str_expression, self.qoperator_dict)
        if not self.add_hc:
            return hamiltonian
        else:
            return hamiltonian + hamiltonian.dag()


class HilbertSpace(dispatch.DispatchClient, serializers.Serializable):
    """Class holding information about the full Hilbert space, usually composed of
    multiple subsys_list. The class provides methods to turn subsystem operators into
    operators acting on the full Hilbert space, and establishes the interface to
    qutip. Returned operators are of the `qutip.Qobj` type. The class also provides
    methods for obtaining eigenvalues, absorption and emission spectra as a function
    of an external parameter.
    """

    osc_subsys_list = descriptors.ReadOnlyProperty()
    qbt_subsys_list = descriptors.ReadOnlyProperty()
    lookup = descriptors.ReadOnlyProperty()
    interaction_list = descriptors.WatchedProperty("INTERACTIONLIST_UPDATE")

    def __init__(
        self,
        subsystem_list: List[QuantumSys],
        interaction_list: List[InteractionTerm] = None,
    ) -> None:
        self._subsystems: Tuple[QuantumSys, ...] = tuple(subsystem_list)
        self.subsys_list = subsystem_list
        if interaction_list:
            self.interaction_list = tuple(interaction_list)
        else:
            self.interaction_list = []

        self._lookup: Optional[spec_lookup.SpectrumLookup] = None
        self._osc_subsys_list = [
            (index, subsys)
            for (index, subsys) in enumerate(self)
            if isinstance(subsys, osc.Oscillator)
        ]
        self._qbt_subsys_list = [
            (index, subsys)
            for (index, subsys) in enumerate(self)
            if not isinstance(subsys, osc.Oscillator)
        ]

        dispatch.CENTRAL_DISPATCH.register("QUANTUMSYSTEM_UPDATE", self)
        dispatch.CENTRAL_DISPATCH.register("INTERACTIONTERM_UPDATE", self)
        dispatch.CENTRAL_DISPATCH.register("INTERACTIONLIST_UPDATE", self)

    def __getitem__(self, index: int) -> QuantumSys:
        return self._subsystems[index]

    def __iter__(self) -> Iterator[QuantumSys]:
        return iter(self._subsystems)

    def __repr__(self) -> str:
        init_dict = self.get_initdata()
        return type(self).__name__ + f"(**{init_dict!r})"

    def __str__(self) -> str:
        output = "HilbertSpace:  subsystems\n"
        output += "-------------------------\n"
        for subsystem in self:
            output += "\n" + str(subsystem) + "\n"
        if self.interaction_list:
            output += "\n\n"
            output += "HilbertSpace:  interaction terms\n"
            output += "--------------------------------\n"
            for interaction_term in self.interaction_list:
                output += "\n" + str(interaction_term) + "\n"
        return output

    ###################################################################################
    # HilbertSpace: file IO methods
    ###################################################################################
    @classmethod
    def deserialize(cls, io_data: "IOData") -> "HilbertSpace":
        """
        Take the given IOData and return an instance of the described class,
        initialized with the data stored in io_data.
        """
        alldata_dict = io_data.as_kwargs()
        lookup = alldata_dict.pop("_lookup", None)
        new_hilbertspace = cls(**alldata_dict)
        new_hilbertspace._lookup = lookup
        if lookup is not None:
            new_hilbertspace._lookup._hilbertspace = weakref.proxy(new_hilbertspace)
        return new_hilbertspace

    def serialize(self) -> "IOData":
        """
        Convert the content of the current class instance into IOData format.
        """
        initdata = {name: getattr(self, name) for name in self._init_params}
        initdata["_lookup"] = self._lookup
        iodata = serializers.dict_serialize(initdata)
        iodata.typename = type(self).__name__
        return iodata

    def get_initdata(self) -> Dict[str, Any]:
        """Returns dict appropriate for creating/initializing a new HilbertSpace
        object."""
        return {
            "subsystem_list": self._subsystems,
            "interaction_list": self.interaction_list,
        }

    ###################################################################################
    # HilbertSpace: creation via GUI
    ###################################################################################
    @classmethod
    def create(cls) -> "HilbertSpace":
        hilbertspace = cls([])
        scqubits.ui.hspace_widget.create_hilbertspace_widget(hilbertspace.__init__)  # type: ignore
        return hilbertspace

    ###################################################################################
    # HilbertSpace: methods for CentralDispatch
    ###################################################################################
    def receive(self, event: str, sender: Any, **kwargs) -> None:
        if self._lookup is not None:
            if event == "QUANTUMSYSTEM_UPDATE" and sender in self:
                self.broadcast("HILBERTSPACE_UPDATE")
                self._lookup._out_of_sync = True
            elif event == "INTERACTIONTERM_UPDATE" and sender in self.interaction_list:
                self.broadcast("HILBERTSPACE_UPDATE")
                self._lookup._out_of_sync = True
            elif event == "INTERACTIONLIST_UPDATE" and sender is self:
                self.broadcast("HILBERTSPACE_UPDATE")
                self._lookup._out_of_sync = True

    ###################################################################################
    # HilbertSpace: subsystems, dimensions, etc.
    ###################################################################################
    def get_subsys_index(self, subsys: QuantumSys) -> int:
        """
        Return the index of the given subsystem in the HilbertSpace.
        """
        return self._subsystems.index(subsys)

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

    ###################################################################################
    # HilbertSpace: generate SpectrumLookup
    ###################################################################################
    def generate_lookup(self) -> None:
        bare_specdata_list = []
        for index, subsys in enumerate(self):
            evals, evecs = subsys.eigensys(evals_count=subsys.truncated_dim)
            bare_specdata_list.append(
                storage.SpectrumData(
                    energy_table=[evals],
                    state_table=[evecs],
                    system_params=subsys.get_initdata(),
                )
            )

        evals, evecs = self.eigensys(evals_count=self.dimension)
        dressed_specdata = storage.SpectrumData(
            energy_table=[evals], state_table=[evecs], system_params=self.get_initdata()
        )
        self._lookup = spec_lookup.SpectrumLookup(
            self,
            bare_specdata_list=bare_specdata_list,
            dressed_specdata=dressed_specdata,
        )

    ###################################################################################
    # HilbertSpace: energy spectrum
    ##################################################################################
    def eigenvals(self, evals_count: int = 6) -> ndarray:
        """Calculates eigenvalues of the full Hamiltonian using
        `qutip.Qob.eigenenergies()`.

        Parameters
        ----------
        evals_count:
            number of desired eigenvalues/eigenstates
        """
        hamiltonian_mat = self.hamiltonian()
        return hamiltonian_mat.eigenenergies(eigvals=evals_count)

    def eigensys(self, evals_count: int = 6) -> Tuple[ndarray, QutipEigenstates]:
        """Calculates eigenvalues and eigenvectors of the full Hamiltonian using
        `qutip.Qob.eigenstates()`.

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

    def _esys_for_paramval(
        self, paramval: float, update_hilbertspace: Callable, evals_count: int
    ) -> Tuple[ndarray, QutipEigenstates]:
        update_hilbertspace(paramval)
        return self.eigensys(evals_count)

    def _evals_for_paramval(
        self, paramval: float, update_hilbertspace: Callable, evals_count: int
    ) -> ndarray:
        update_hilbertspace(paramval)
        return self.eigenvals(evals_count)

    ###################################################################################
    # HilbertSpace: Hamiltonian (bare, interaction, full)
    #######################################################

    def hamiltonian(self) -> Qobj:
        """

        Returns
        -------
            Hamiltonian of the composite system, including the interaction between
            components
        """
        return self.bare_hamiltonian() + self.interaction_hamiltonian()

    def bare_hamiltonian(self) -> Qobj:
        """
        Returns
        -------
            composite Hamiltonian composed of bare Hamiltonians of subsys_list
            independent of the external parameter
        """
        bare_hamiltonian = 0
        for subsys in self:
            evals = subsys.eigenvals(evals_count=subsys.truncated_dim)
            bare_hamiltonian += self.diag_hamiltonian(subsys, evals)
        return bare_hamiltonian

    def interaction_hamiltonian(self) -> Qobj:
        """
        Deprecated, will be changed in future versions.

        Returns
        -------
            interaction Hamiltonian
        """
        if not self.interaction_list:
            return 0

        operator_list = []
        for term in self.interaction_list:
            if isinstance(term, InteractionTerm) or isinstance(
                term, InteractionTermStr
            ):
                operator_list.append(term.hamiltonian())
            elif isinstance(term, InteractionTermLegacy):
                interactionlegacy_hamiltonian = self.interactionterm_hamiltonian(term)
                operator_list.append(interactionlegacy_hamiltonian)
            elif isinstance(term, Qobj):
                operator_list.append(term)
            else:
                raise TypeError(
                    "Expected an instance of InteractionTerm, InteractionTermStr, "
                    "or Qobj."
                )
        hamiltonian = sum(operator_list)
        return hamiltonian

    def interactionterm_hamiltonian(
        self,
        interactionterm: InteractionTermLegacy,
        evecs1: ndarray = None,
        evecs2: ndarray = None,
    ) -> Qobj:
        """Deprecated, will not work in future versions."""
        interaction_op1 = spec_utils.identity_wrap(
            interactionterm.op1, interactionterm.subsys1, self.subsys_list, evecs=evecs1
        )
        interaction_op2 = spec_utils.identity_wrap(
            interactionterm.op2, interactionterm.subsys2, self.subsys_list, evecs=evecs2
        )
        hamiltonian = interactionterm.g_strength * interaction_op1 * interaction_op2
        if interactionterm.add_hc:
            return hamiltonian + hamiltonian.dag()
        return hamiltonian

    def diag_hamiltonian(self, subsystem: QuantumSys, evals: ndarray = None) -> Qobj:
        """Returns a `qutip.Qobj` which has the eigenenergies of the object `subsystem`
        on the diagonal.

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

    def get_bare_hamiltonian(self) -> Qobj:
        """Deprecated, use `bare_hamiltonian()` instead."""
        warnings.warn(
            "get_bare_hamiltonian() is deprecated, use bare_hamiltonian() instead",
            FutureWarning,
        )
        return self.bare_hamiltonian()

    def get_hamiltonian(self):
        """Deprecated, use `hamiltonian()` instead."""
        return self.hamiltonian()

    ###################################################################################
    # HilbertSpace: identity wrapping, operators
    ###################################################################################

    def diag_operator(self, diag_elements: ndarray, subsystem: QuantumSys) -> Qobj:
        """For given diagonal elements of a diagonal operator in `subsystem`, return
        the `Qobj` operator for the full Hilbert space (perform wrapping in
        identities for other subsys_list).

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
        operator = qt.states.basis(dim, j) * qt.states.basis(dim, k).dag()
        return spec_utils.identity_wrap(operator, subsystem, self.subsys_list)

    def annihilate(self, subsystem: QuantumSys) -> Qobj:
        """Annihilation operator a for `subsystem`

        Parameters
        ----------
        subsystem:
            specifies subsystem in which annihilation operator acts
        """
        dim = subsystem.truncated_dim
        operator = qt.destroy(dim)
        return spec_utils.identity_wrap(operator, subsystem, self.subsys_list)

    ###################################################################################
    # HilbertSpace: spectrum sweep
    ###################################################################################
    def get_spectrum_vs_paramvals(
        self,
        param_vals: ndarray,
        update_hilbertspace: Callable,
        evals_count: int = 10,
        get_eigenstates: bool = False,
        param_name: str = "external_parameter",
        num_cpus: int = settings.NUM_CPUS,
    ) -> SpectrumData:
        """Return eigenvalues (and optionally eigenstates) of the full Hamiltonian as
        a function of a parameter. Parameter values are specified as a list or array
        in `param_vals`. The Hamiltonian `hamiltonian_func` must be a function of
        that particular parameter, and is expected to internally set subsystem
        parameters. If a `filename` string is provided, then eigenvalue data is
        written to that file.

        Parameters
        ----------
        param_vals:
            array of parameter values
        update_hilbertspace:
            update_hilbertspace(param_val) specifies how a change in the external
            parameter affects the Hilbert space components
        evals_count:
            number of desired energy levels (default value = 10)
        get_eigenstates:
            set to true if eigenstates should be returned as well
            (default value = False)
        param_name:
            name for the parameter that is varied in `param_vals`
            (default value = "external_parameter")
        num_cpus:
            number of cores to be used for computation
            (default value: settings.NUM_CPUS)
        """
        target_map = cpu_switch.get_map_method(num_cpus)
        if get_eigenstates:
            func = functools.partial(
                self._esys_for_paramval,
                update_hilbertspace=update_hilbertspace,
                evals_count=evals_count,
            )
            with utils.InfoBar(
                "Parallel computation of eigenvalues [num_cpus={}]".format(num_cpus),
                num_cpus,
            ):
                eigensystem_mapdata = list(
                    target_map(
                        func,
                        tqdm(
                            param_vals,
                            desc="Spectral data",
                            leave=False,
                            disable=(num_cpus > 1),
                        ),
                    )
                )
            eigenvalue_table, eigenstate_table = spec_utils.recast_esys_mapdata(
                eigensystem_mapdata
            )
        else:
            func = functools.partial(
                self._evals_for_paramval,
                update_hilbertspace=update_hilbertspace,
                evals_count=evals_count,
            )
            with utils.InfoBar(
                "Parallel computation of eigensystems [num_cpus={}]".format(num_cpus),
                num_cpus,
            ):
                eigenvalue_table = list(
                    target_map(
                        func,
                        tqdm(
                            param_vals,
                            desc="Spectral data",
                            leave=False,
                            disable=(num_cpus > 1),
                        ),
                    )
                )
            eigenvalue_table = np.asarray(eigenvalue_table)
            eigenstate_table = None  # type: ignore

        return storage.SpectrumData(
            eigenvalue_table,
            self.get_initdata(),
            param_name,
            param_vals,
            state_table=eigenstate_table,
        )
