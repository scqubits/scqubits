# hilbert_space.py
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

from __future__ import annotations

import functools
import importlib
import re


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
    cast,
    overload,
)

import numpy as np
import qutip as qt

from numpy import ndarray
from qutip.qobj import Qobj
from scipy.sparse import csc_matrix, dia_matrix

import scqubits.core.central_dispatch as dispatch
import scqubits.core.descriptors as descriptors
import scqubits.core.oscillator as osc
import scqubits.core.spec_lookup as spec_lookup
import scqubits.core.storage as storage
import scqubits.io_utils.fileio_qutip
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.settings as settings
import scqubits.ui.hspace_widget
import scqubits.utils.cpu_switch as cpu_switch
import scqubits.utils.misc as utils
import scqubits.utils.spectrum_utils as spec_utils
from scqubits.core.namedslots_array import NamedSlotsNdarray, Parameters

from scqubits.core.storage import SpectrumData
from scqubits.io_utils.fileio_qutip import QutipEigenstates

if settings.IN_IPYTHON:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

if TYPE_CHECKING:
    from scqubits.io_utils.fileio import IOData

from scqubits.utils.typedefs import OscillatorList, QuantumSys, QubitList


def has_duplicate_id_str(subsystem_list: List[QuantumSys]):
    id_str_list = [obj._id_str for obj in subsystem_list]
    id_str_set = set(obj._id_str for obj in subsystem_list)
    return len(id_str_set) != len(id_str_list)


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
        list of tuples (subsys_index, operator)
    add_hc:
        If set to True, the interaction Hamiltonian is of type 2, and the Hermitian
        conjugate is added.
    """

    g_strength = descriptors.WatchedProperty(complex, "INTERACTIONTERM_UPDATE")
    operator_list = descriptors.WatchedProperty(
        List[Tuple[int, Union[ndarray, csc_matrix]]], "INTERACTIONTERM_UPDATE"
    )
    add_hc = descriptors.WatchedProperty(bool, "INTERACTIONTERM_UPDATE")

    def __init__(
        self,
        g_strength: Union[float, complex],
        operator_list: List[Tuple[int, Union[ndarray, csc_matrix]]],
        add_hc: bool = False,
    ) -> None:
        self.g_strength = g_strength
        self.operator_list = operator_list
        self.add_hc = add_hc

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

    def hamiltonian(
        self,
        subsystem_list: List[QuantumSys],
        bare_esys: Optional[Dict[int, ndarray]] = None,
    ) -> Qobj:
        """
        Returns the full Hamiltonian of the interacting quantum system described by the
        HilbertSpace object

        Parameters
        ----------
        subsystem_list:
            list of all quantum systems in HilbertSpace calling ``hamiltonian``,
            needed for identity wrapping
        bare_esys:
            optionally, the bare eigensystems for each subsystem can be provided to
            speed up computation; these are provided in dict form via <subsys>: esys)

        Returns
        -------
            Hamiltonian in `qutip.Qobj` format
        """
        hamiltonian = cast(Qobj, self.g_strength)
        id_wrapped_ops = self.id_wrap_all_ops(
            self.operator_list, subsystem_list, bare_esys=bare_esys
        )
        for op in id_wrapped_ops:
            hamiltonian *= op
        if self.add_hc:
            hamiltonian += hamiltonian.dag()
        return hamiltonian

    @staticmethod
    def id_wrap_all_ops(
        operator_list: List[Tuple[int, Union[ndarray, csc_matrix]]],
        subsystem_list: List[QuantumSys],
        bare_esys: Optional[Dict[int, ndarray]] = None,
    ) -> list:
        id_wrapped_operators = []
        for subsys_index, operator in operator_list:
            if bare_esys is not None and subsys_index in bare_esys:
                evecs = bare_esys[subsys_index][1]
            else:
                evecs = None
            id_wrapped_operators.append(
                spec_utils.identity_wrap(
                    operator, subsystem_list[subsys_index], subsystem_list, evecs=evecs
                )
            )
        return id_wrapped_operators


class InteractionTermStr(dispatch.DispatchClient, serializers.Serializable):
    """
    Class for specifying a term in the interaction Hamiltonian of a composite Hilbert
    space, and constructing the Hamiltonian in qutip.Qobj format. The form of the
    interaction is defined using the expr string. Each operator must be
    hermitian, unless add_hc = True in which case each operator my be non-hermitian.
    Acceptable functions inside of expr string include: cos(), sin(),
    dag(), conj(), exp(), sqrt(), trans(), cosm(), sinm(), expm(), and sqrtm() along
    with other operators allowed in Python expressions.

    Parameters
    ----------
    expr:
        string that defines the interaction.
    operator_list:
        list of tuples of operator names, operators, and subsystem indices
        eg. {name: (operator, subsystem)}.
    add_hc:
        If set to True, the interaction Hamiltonian is of type 2, and the Hermitian
        conjugate is added.

    """

    expr = descriptors.WatchedProperty(str, "INTERACTIONTERM_UPDATE")
    operator_list = descriptors.WatchedProperty(
        List[Tuple[int, str, Union[ndarray, csc_matrix, dia_matrix]]],
        "INTERACTIONTERM_UPDATE",
    )
    add_hc = descriptors.WatchedProperty(bool, "INTERACTIONTERM_UPDATE")

    def __init__(
        self,
        expr: str,
        operator_list: List[Tuple[int, str, Union[ndarray, csc_matrix, dia_matrix]]],
        const: Optional[Dict[str, Union[float, complex]]] = None,
        add_hc: bool = False,
    ) -> None:
        self.qutip_dict = {
            "cosm(": "Qobj.cosm(",
            "expm(": "Qobj.expm(",
            "sinm(": "Qobj.sinm(",
            "sqrtm(": "Qobj.sqrtm(",
            "cos(": "Qobj.cosm(",
            "exp(": "Qobj.expm(",
            "sin(": "Qobj.sinm(",
            "sqrt(": "Qobj.sqrtm(",
        }
        self.expr = expr
        self.operator_list = operator_list
        self.const = const or {}
        self.add_hc = add_hc

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

    def parse_qutip_functions(self, string: str) -> str:
        for item, value in self.qutip_dict.items():
            if item in string:
                string = string.replace(item, value)
        return string

    def run_string_code(
        self, expression: str, idwrapped_ops_by_name: Dict[str, Qobj]
    ) -> Qobj:
        expression = self.parse_qutip_functions(expression)
        idwrapped_ops_by_name["Qobj"] = Qobj

        main = importlib.import_module("__main__")
        answer = eval(
            expression, {**main.__dict__, **idwrapped_ops_by_name, **self.const}
        )
        return answer

    def id_wrap_all_ops(
        self,
        subsys_list: List[QuantumSys],
        bare_esys: Optional[Dict[int, ndarray]] = None,
    ) -> Dict[str, Qobj]:
        idwrapped_ops_by_name = {}
        for subsys_index, name, op in self.operator_list:
            if bare_esys and subsys_index in bare_esys:
                evecs = bare_esys[subsys_index][1]
            else:
                evecs = None
            idwrapped_ops_by_name[name] = spec_utils.identity_wrap(
                op, subsys_list[subsys_index], subsys_list, evecs=evecs
            )
        return idwrapped_ops_by_name

    def hamiltonian(
        self,
        subsystem_list: List[QuantumSys],
        bare_esys: Optional[Dict[int, ndarray]] = None,
    ) -> Qobj:
        """
        Parameters
        ----------
        subsystem_list:
            list of all quantum systems in HilbertSpace calling ``hamiltonian``,
            needed for identity wrapping
        bare_esys:
            optionally, the bare eigensystems for each subsystem can be provided to
            speed up computation; these are provided in dict form via <subsys>: esys)
        """
        idwrapped_ops_by_name = self.id_wrap_all_ops(
            subsystem_list, bare_esys=bare_esys
        )
        hamiltonian = self.run_string_code(self.expr, idwrapped_ops_by_name)
        if not self.add_hc:
            return hamiltonian
        else:
            return hamiltonian + hamiltonian.dag()


class HilbertSpace(
    spec_lookup.SpectrumLookupMixin, dispatch.DispatchClient, serializers.Serializable
):
    """Class holding information about the full Hilbert space, usually composed of
    multiple subsys_list. The class provides methods to turn subsystem operators into
    operators acting on the full Hilbert space, and establishes the interface to
    qutip. Returned operators are of the `qutip.Qobj` type. The class also provides
    methods for obtaining eigenvalues, absorption and emission spectra as a function
    of an external parameter.

    Parameters
    ----------
    subsystem_list:
        List of all quantum systems comprising the composite Hilbert space
    interaction_list:
        (optional) typically, interaction terms are added one by one by means of the
        `add_interaction` method. Alternatively, a list of interaction term objects
        can be supplied here upon initialization of a `HilbertSpace` instance.
    """

    osc_subsys_list = descriptors.ReadOnlyProperty(OscillatorList)
    qbt_subsys_list = descriptors.ReadOnlyProperty(QubitList)
    lookup = descriptors.ReadOnlyProperty(spec_lookup.SpectrumLookupAdapter)
    interaction_list = descriptors.WatchedProperty(
        Tuple[Union[InteractionTerm, InteractionTermStr], ...], "INTERACTIONLIST_UPDATE"
    )

    def __init__(
        self,
        subsystem_list: List[QuantumSys],
        interaction_list: List[Union[InteractionTerm, InteractionTermStr]] = None,
        ignore_low_overlap: bool = False,
    ) -> None:
        if has_duplicate_id_str(subsystem_list):
            raise ValueError(
                "Subsystem list must not contain multiple objects with "
                "the same `id_str` name."
            )
        self._subsystems: Tuple[QuantumSys, ...] = tuple(subsystem_list)
        self._subsys_by_id_str = {
            obj._id_str: self[index] for index, obj in enumerate(self)
        }
        if interaction_list:
            self.interaction_list = interaction_list
        else:
            self.interaction_list: Tuple[InteractionTerm, ...] = []
        self._interaction_term_by_id_str = {
            "InteractionTerm_{}".format(index): interaction_term
            for index, interaction_term in enumerate(self.interaction_list)
        }

        self._lookup: Optional[spec_lookup.SpectrumLookupAdapter] = None
        self._osc_subsys_list = [
            subsys for subsys in self if isinstance(subsys, osc.Oscillator)
        ]
        self._qbt_subsys_list = [
            subsys for subsys in self if not isinstance(subsys, osc.Oscillator)
        ]

        # The following attributes are for compatibility with SpectrumLookupMixin
        self._data: Dict[str, Any] = {}
        self._parameters = Parameters({"dummy_parameter": np.array([0])})
        self._ignore_low_overlap = ignore_low_overlap
        self._current_param_indices = 0
        self._evals_count = self.dimension
        self._out_of_sync = False
        # end attributes for compatibility with SpectrumLookupMixin

        dispatch.CENTRAL_DISPATCH.register("QUANTUMSYSTEM_UPDATE", self)
        dispatch.CENTRAL_DISPATCH.register("INTERACTIONTERM_UPDATE", self)
        dispatch.CENTRAL_DISPATCH.register("INTERACTIONLIST_UPDATE", self)

    @overload
    def __getitem__(self, key: int) -> QuantumSys:
        ...

    @overload
    def __getitem__(
        self, key: str
    ) -> Union[QuantumSys, InteractionTerm, InteractionTermStr]:
        ...

    def __getitem__(
        self, key: Union[int, str]
    ) -> Union[QuantumSys, InteractionTerm, InteractionTermStr]:
        if isinstance(key, int):
            return self._subsystems[key]
        if key in self._subsys_by_id_str:
            return self._subsys_by_id_str[key]
        if key in self._interaction_term_by_id_str:
            return self._interaction_term_by_id_str[key]
        if key in self._data.keys():
            return self._data[key]

        raise KeyError(
            "Unrecognized key: {}. Key must be an integer index or a "
            "string specifying a subsystem or interaction term part of "
            "HilbertSpace.".format(key)
        )

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

            for id_str, interaction_term in self._interaction_term_by_id_str.items():
                indent_length = 25
                term_output = "InteractionTerm".ljust(indent_length, "-")
                term_output += "| [{}]\n".format(id_str)
                term_output += "\n".join(str(interaction_term).splitlines()[1:])
                term_output += "\n\n"
                output += term_output
        return output

    def __len__(self):
        return len(self._subsystems)

    @property
    def lookup(self):
        """[Legacy] supporting old lookup interface."""
        return self._lookup

    @property
    def hilbertspace(self) -> "HilbertSpace":
        """[Legacy] Auxiliary reference to self for compatibility with
        SpectrumLookupMixin
        class."""
        return self

    @property
    def subsys_list(self) -> List[QuantumSys]:
        return list(self._subsystems)

    def subsys_by_id_str(self, id_str: str) -> QuantumSys:
        return self._subsys_by_id_str[id_str]

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
        alldata_dict["ignore_low_overlap"] = alldata_dict.pop("_ignore_low_overlap")
        data = alldata_dict.pop("_data", {})
        new_hilbertspace: "HilbertSpace" = cls(**alldata_dict)
        new_hilbertspace._data = data
        new_hilbertspace._lookup = spec_lookup.SpectrumLookupAdapter(new_hilbertspace)
        return new_hilbertspace

    def serialize(self) -> "IOData":
        """
        Convert the content of the current class instance into IOData format.
        """
        init_parameters = self._init_params
        init_parameters.remove("ignore_low_overlap")
        init_parameters.append("_ignore_low_overlap")
        initdata = {name: getattr(self, name) for name in init_parameters}
        if self._data:
            initdata = {**initdata, "_data": self._data}
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
        scqubits.ui.hspace_widget.create_hilbertspace_widget(hilbertspace.__init__)
        return hilbertspace

    ###################################################################################
    # HilbertSpace: methods for CentralDispatch
    ###################################################################################
    def receive(self, event: str, sender: Any, **kwargs) -> None:
        if event == "QUANTUMSYSTEM_UPDATE" and sender in self:
            self.broadcast("HILBERTSPACE_UPDATE")
            if self._lookup:
                self._lookup._out_of_sync = True
        elif event == "INTERACTIONTERM_UPDATE" and sender in self.interaction_list:
            self.broadcast("HILBERTSPACE_UPDATE")
            if self._lookup:
                self._lookup._out_of_sync = True
        elif event == "INTERACTIONLIST_UPDATE" and sender is self:
            self.broadcast("HILBERTSPACE_UPDATE")
            if self._lookup:
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
        return np.prod(np.asarray(self.subsystem_dims))  # type:ignore

    @property
    def subsystem_count(self) -> int:
        """Returns number of subsys_list composing the joint Hilbert space"""
        return len(self._subsystems)

    ###################################################################################
    # HilbertSpace: generate SpectrumLookup
    ###################################################################################
    def generate_lookup(self) -> None:
        bare_evals = np.empty((self.subsystem_count,), dtype=object)
        bare_evecs = np.empty((self.subsystem_count,), dtype=object)
        bare_esys_dict = {}

        dummy_params = self._parameters.paramvals_by_name

        for subsys_index, subsys in enumerate(self):
            bare_esys = subsys.eigensys(evals_count=subsys.truncated_dim)
            bare_esys_dict[subsys_index] = bare_esys
            bare_evals[subsys_index] = NamedSlotsNdarray(
                np.asarray([bare_esys[0].tolist()]),
                self._parameters.paramvals_by_name,
            )
            bare_evecs[subsys_index] = NamedSlotsNdarray(
                np.asarray([bare_esys[1].tolist()]),
                self._parameters.paramvals_by_name,
            )
        self._data["bare_evals"] = NamedSlotsNdarray(
            bare_evals, {"subsys": np.arange(self.subsystem_count)}
        )
        self._data["bare_evecs"] = NamedSlotsNdarray(
            bare_evecs, {"subsys": np.arange(self.subsystem_count)}
        )

        evals, evecs = self.eigensys(
            evals_count=self.dimension, bare_esys=bare_esys_dict
        )
        # The following workaround ensures that eigenvectors maintain QutipEigenstates
        # view when getting placed inside an outer array
        evecs_wrapped = np.empty(shape=1, dtype=object)
        evecs_wrapped[0] = evecs

        self._data["evals"] = NamedSlotsNdarray(np.array([evals]), dummy_params)
        self._data["evecs"] = NamedSlotsNdarray(evecs_wrapped, dummy_params)
        self._data["dressed_indices"] = spec_lookup.SpectrumLookupMixin.generate_lookup(
            self
        )
        self._lookup = spec_lookup.SpectrumLookupAdapter(self)

    ###################################################################################
    # HilbertSpace: energy spectrum
    ##################################################################################
    def eigenvals(
        self,
        evals_count: int = 6,
        bare_esys: Optional[Dict[int, Union[ndarray, List[ndarray]]]] = None,
    ) -> ndarray:
        """Calculates eigenvalues of the full Hamiltonian using
        `qutip.Qob.eigenenergies()`.

        Parameters
        ----------
        evals_count:
            number of desired eigenvalues/eigenstates
        bare_esys:
            optionally, the bare eigensystems for each subsystem can be provided to
            speed up computation; these are provided in dict form via <subsys>: esys
        """
        hamiltonian_mat = self.hamiltonian(bare_esys=bare_esys)  # type:ignore
        return hamiltonian_mat.eigenenergies(eigvals=evals_count)

    def eigensys(
        self,
        evals_count: int = 6,
        bare_esys: Optional[Dict[int, Union[ndarray, List[ndarray]]]] = None,
    ) -> Tuple[ndarray, QutipEigenstates]:
        """Calculates eigenvalues and eigenvectors of the full Hamiltonian using
        `qutip.Qob.eigenstates()`.

        Parameters
        ----------
        evals_count:
            number of desired eigenvalues/eigenstates
        bare_esys:
            optionally, the bare eigensystems for each subsystem can be provided to
            speed up computation; these are provided in dict form via <subsys>: esys

        Returns
        -------
            eigenvalues and eigenvectors
        """
        hamiltonian_mat = self.hamiltonian(bare_esys=bare_esys)  # type:ignore
        evals, evecs = hamiltonian_mat.eigenstates(eigvals=evals_count)
        evecs = evecs.view(scqubits.io_utils.fileio_qutip.QutipEigenstates)
        return evals, evecs

    def _esys_for_paramval(
        self,
        paramval: float,
        update_hilbertspace: Callable,
        evals_count: int,
        bare_esys: Optional[Dict[int, Union[ndarray, List[ndarray]]]] = None,
    ) -> Tuple[ndarray, QutipEigenstates]:
        update_hilbertspace(paramval)
        return self.eigensys(evals_count, bare_esys=bare_esys)

    def _evals_for_paramval(
        self,
        paramval: float,
        update_hilbertspace: Callable,
        evals_count: int,
        bare_esys: Optional[Dict[int, Union[ndarray, List[ndarray]]]] = None,
    ) -> ndarray:
        update_hilbertspace(paramval)
        return self.eigenvals(evals_count, bare_esys=bare_esys)

    ###################################################################################
    # HilbertSpace: Hamiltonian (bare, interaction, full)
    #######################################################

    def hamiltonian(
        self,
        bare_esys: Optional[Dict[int, ndarray]] = None,
    ) -> Qobj:
        """
        Parameters
        ----------
        bare_esys:
            optionally, the bare eigensystems for each subsystem can be provided to
            speed up computation; these are provided in dict form via <subsys>: esys

        Returns
        -------
            Hamiltonian of the composite system, including the interaction between
            components
        """
        hamiltonian = self.bare_hamiltonian(bare_esys=bare_esys)
        hamiltonian += self.interaction_hamiltonian(bare_esys=bare_esys)
        return hamiltonian

    def bare_hamiltonian(self, bare_esys: Optional[Dict[int, ndarray]] = None) -> Qobj:
        """
        Parameters
        ----------
        bare_esys:
            optionally, the bare eigensystems for each subsystem can be provided to
            speed up computation; these are provided in dict form via <subsys>: esys

        Returns
        -------
            composite Hamiltonian composed of bare Hamiltonians of subsys_list
            independent of the external parameter
        """
        bare_hamiltonian = 0
        for subsys_index, subsys in enumerate(self):
            if bare_esys is not None and subsys_index in bare_esys:
                evals = bare_esys[subsys_index][0]
            else:
                evals = subsys.eigenvals(evals_count=subsys.truncated_dim)
            bare_hamiltonian += self.diag_hamiltonian(subsys, evals)
        return bare_hamiltonian

    def interaction_hamiltonian(
        self, bare_esys: Optional[Dict[int, ndarray]] = None
    ) -> Qobj:
        """
        Returns the interaction Hamiltonian, based on the interaction terms specified
        for the current HilbertSpace object

        Parameters
        ----------
        bare_esys:
            optionally, the bare eigensystems for each subsystem can be provided to
            speed up computation; these are provided in dict form via <subsys>: esys

        Returns
        -------
            interaction Hamiltonian
        """
        if not self.interaction_list:
            return 0

        operator_list = []
        for term in self.interaction_list:
            if isinstance(term, Qobj):
                operator_list.append(term)
            elif isinstance(term, (InteractionTerm, InteractionTermStr)):
                operator_list.append(
                    term.hamiltonian(self.subsys_list, bare_esys=bare_esys)
                )
            else:
                raise TypeError(
                    "Expected an instance of InteractionTerm, InteractionTermStr, "
                    "or Qobj; got {} instead.".format(type(term))
                )
        hamiltonian = sum(operator_list)
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
        diag_qt_op = qt.Qobj(inpt=np.diagflat(evals[0:evals_count]))  # type:ignore
        return spec_utils.identity_wrap(diag_qt_op, subsystem, self.subsys_list)

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
        num_cpus: Optional[int] = None,
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
        num_cpus = num_cpus or settings.NUM_CPUS
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
                self._evals_for_paramval,  # type:ignore
                update_hilbertspace=update_hilbertspace,
                evals_count=evals_count,
            )
            with utils.InfoBar(
                "Parallel computation of eigensystems [num_cpus={}]".format(num_cpus),
                num_cpus,
            ):
                eigenvalue_table = np.asarray(
                    list(
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
                )
            eigenstate_table = None  # type: ignore

        return storage.SpectrumData(
            eigenvalue_table,
            self.get_initdata(),
            param_name,
            param_vals,
            state_table=eigenstate_table,
        )

    ###################################################################################
    # HilbertSpace: add interaction and parsing arguments to .add_interaction
    ###################################################################################
    def add_interaction(
        self, check_validity=True, id_str: Optional[str] = None, **kwargs
    ) -> None:
        """
        Specify the interaction between subsystems making up the `HilbertSpace`
        instance. `add_interaction(...)` offers three different interfaces:

        * Simple interface for operator products
        * String-based interface for more general interaction operator expressions
        * General Qobj interface

        1. Simple interface for operator products
            Specify `ndarray`, `csc_matrix`, or `dia_matrix` (subsystem operator in
            subsystem-internal basis) along with the corresponding subsystem

            signature::

                .add_interaction(g=<float>,
                                op1=(<ndarray>, <QuantumSystem>),
                                op2=(<csc_matrix>, <QuantumSystem>),
                                 …,
                                add_hc=<bool>)

            Alternatively, specify subsystem operators via callable methods.

            signature::

                .add_interaction(g=<float>,
                                 op1=<Callable>,
                                 op2=<Callable>,
                                 …,
                                 add_hc=<bool>)
        2. String-based interface for more general interaction operator expressions
                Specify a Python expression that generates the desired operator. The
                expression enables convenience use of basic qutip operations::

                    .add_interaction(expr=<str>,
                                     op1=(<str>, <ndarray>, <subsys>),
                                     op2=(<str>, <Callable>),
                                     …)
        3. General Qobj operator
            Specify a fully identity-wrapped `qutip.Qobj` operator. Signature::

                .add_interaction(qobj=<Qobj>)

        id_str:
            optional string by which this instance can be referred to in `HilbertSpace`
            and `ParameterSweep`. If not provided, an id is auto-generated.
        """
        if "expr" in kwargs:
            interaction: Union[
                InteractionTerm, InteractionTermStr
            ] = self._parse_interactiontermstr(**kwargs)
        elif "qobj" in kwargs:
            interaction = self._parse_qobj(**kwargs)
        elif "op1" in kwargs:
            interaction = self._parse_interactionterm(**kwargs)
        else:
            raise TypeError(
                "Invalid combination and/or types of arguments for `add_interaction`"
            )
        if self._lookup is not None:
            self._lookup._out_of_sync = True

        self.interaction_list.append(interaction)

        id_str = id_str or "Interaction_{}".format(len(self.interaction_list))
        self._interaction_term_by_id_str[id_str] = interaction

        if not check_validity:
            return None
        try:
            _ = self.interaction_hamiltonian()
        except:
            self.interaction_list.pop()
            del self._interaction_term_by_id_str[id_str]
            raise ValueError("Invalid Interaction Term")

    def _parse_interactiontermstr(self, **kwargs) -> InteractionTermStr:
        expr = kwargs.pop("expr")
        add_hc = kwargs.pop("add_hc", False)
        const = kwargs.pop("const", None)

        operator_list = []
        for key in kwargs.keys():
            if re.match(r"op\d+$", key) is None:
                raise TypeError("Unexpected keyword argument {}.".format(key))
            operator_list.append(self._parse_op_by_name(kwargs[key]))

        return InteractionTermStr(expr, operator_list, const=const, add_hc=add_hc)

    def _parse_interactionterm(self, **kwargs) -> InteractionTerm:
        g = kwargs.pop("g", None)
        if g is None:
            g = kwargs.pop("g_strength")
        add_hc = kwargs.pop("add_hc", False)

        operator_list = []
        for key in kwargs.keys():
            if re.match(r"op\d+$", key) is None:
                raise TypeError("Unexpected keyword argument {}.".format(key))
            subsys_index, op = self._parse_op(kwargs[key])
            operator_list.append((subsys_index, op))

        return InteractionTerm(g, operator_list, add_hc=add_hc)

    @staticmethod
    def _parse_qobj(**kwargs) -> Qobj:
        op = kwargs["qobj"]
        if len(kwargs) > 1 or not isinstance(op, Qobj):
            raise TypeError("Cannot interpret specified operator {}".format(op))
        return kwargs["qobj"]

    def _parse_op_by_name(
        self, op_by_name
    ) -> Tuple[int, str, Union[ndarray, csc_matrix, dia_matrix]]:
        if not isinstance(op_by_name, tuple):
            raise TypeError("Cannot interpret specified operator {}".format(op_by_name))
        if len(op_by_name) == 3:
            # format expected:  (<op name as str>, <op as array>, <subsys as QuantumSystem>)
            return self.get_subsys_index(op_by_name[2]), op_by_name[0], op_by_name[1]
        # format expected (<op name as str)>, <QuantumSystem.method callable>)
        return (
            self.get_subsys_index(op_by_name[1].__self__),
            op_by_name[0],
            op_by_name[1](),
        )

    def _parse_op(
        self, op: Union[Callable, Tuple[Union[ndarray, csc_matrix], QuantumSys]]
    ) -> Tuple[int, Union[ndarray, csc_matrix]]:
        if callable(op):
            return self.get_subsys_index(op.__self__), op()  # type:ignore
        if not isinstance(op, tuple):
            raise TypeError("Cannot interpret specified operator {}".format(op))
        if len(op) == 2:
            # format expected:  (<op as array>, <subsys as QuantumSystem>)
            return self.get_subsys_index(op[1]), op[0]
        raise TypeError("Cannot interpret specified operator {}".format(op))
