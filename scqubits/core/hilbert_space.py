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
import warnings
import weakref

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


class InteractionTerm(dispatch.DispatchClient, serializers.Serializable):
    """
    Class for specifying a term in the interaction Hamiltonian of a composite Hilbert space, and constructing
    the Hamiltonian in qutip.Qobj format. The expected form of the interaction term is of two possible types:
    1. V = g A B, where A, B are Hermitean operators in two specified subsys_list,
    2. V = g A B + h.c., where A, B may be non-Hermitean

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
        If set to True, the interaction Hamiltonian is of type 2, and the Hermitean conjugate is added.
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
    ) -> None:
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
        name_prepend = "InteractionTerm".ljust(indent_length, "-") + "|\n"

        output = ""
        for param_name in self._init_params:
            param_content = getattr(self, param_name).__repr__()
            if "\n" in param_content:
                length = min(param_content.rfind("\n") - 1, 30)
                param_content = param_content[:length]
                param_content += " ..."

            output += "{0}| {1}: {2}\n".format(
                " " * indent_length, str(param_name), param_content
            )
        return name_prepend + output


class HilbertSpace(dispatch.DispatchClient, serializers.Serializable):
    """Class holding information about the full Hilbert space, usually composed of multiple subsys_list.
    The class provides methods to turn subsystem operators into operators acting on the full Hilbert space, and
    establishes the interface to qutip. Returned operators are of the `qutip.Qobj` type. The class also provides methods
    for obtaining eigenvalues, absorption and emission spectra as a function of an external parameter.
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

    ###############################################################################################
    # HilbertSpace: file IO methods
    ###############################################################################################
    @classmethod
    def deserialize(cls, io_data: "IOData") -> "HilbertSpace":
        """
        Take the given IOData and return an instance of the described class, initialized with the data stored in
        io_data.
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
        """Returns dict appropriate for creating/initializing a new HilbertSpace object.
        """
        return {
            "subsystem_list": self._subsystems,
            "interaction_list": self.interaction_list,
        }

    ###############################################################################################
    # HilbertSpace: creation via GUI
    ###############################################################################################
    @classmethod
    def create(cls) -> "HilbertSpace":
        hilbertspace = cls([])
        scqubits.ui.hspace_widget.create_hilbertspace_widget(hilbertspace.__init__)  # type: ignore
        return hilbertspace

    ###############################################################################################
    # HilbertSpace: methods for CentralDispatch
    ###############################################################################################
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

    ###############################################################################################
    # HilbertSpace: subsystems, dimensions, etc.
    ###############################################################################################
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

    ###############################################################################################
    # HilbertSpace: generate SpectrumLookup
    ###############################################################################################
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

    ###############################################################################################
    # HilbertSpace: energy spectrum
    ###############################################################################################
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
        """Calculates eigenvalues and eigenvectors of the full Hamiltonian using `qutip.Qob.eigenstates()`.

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

    ###############################################################################################
    # HilbertSpace: Hamiltonian (bare, interaction, full)
    ###############################################################################################
    def hamiltonian(self) -> Qobj:
        """

        Returns
        -------
            Hamiltonian of the composite system, including the interaction between components
        """
        return self.bare_hamiltonian() + self.interaction_hamiltonian()

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

    def interaction_hamiltonian(self) -> Qobj:
        """
        Returns
        -------
            interaction Hamiltonian
        """
        if not self.interaction_list:
            return 0

        hamiltonian = [
            self.interactionterm_hamiltonian(term) for term in self.interaction_list
        ]
        return sum(hamiltonian)

    def interactionterm_hamiltonian(
        self,
        interactionterm: InteractionTerm,
        evecs1: ndarray = None,
        evecs2: ndarray = None,
    ) -> Qobj:
        interaction_op1 = self.identity_wrap(
            interactionterm.op1, interactionterm.subsys1, evecs=evecs1
        )
        interaction_op2 = self.identity_wrap(
            interactionterm.op2, interactionterm.subsys2, evecs=evecs2
        )
        hamiltonian = interactionterm.g_strength * interaction_op1 * interaction_op2
        if interactionterm.add_hc:
            return hamiltonian + hamiltonian.dag()
        return hamiltonian

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
        return self.identity_wrap(diag_qt_op, subsystem)

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

    ###############################################################################################
    # HilbertSpace: identity wrapping, operators
    ###############################################################################################
    def identity_wrap(
        self,
        operator: Union[str, ndarray, csc_matrix, dia_matrix, Qobj],
        subsystem: QuantumSys,
        op_in_eigenbasis: bool = False,
        evecs: ndarray = None,
    ) -> Qobj:
        """Wrap given operator in subspace `subsystem` in identity operators to form full Hilbert-space operator.

        Parameters
        ----------
        operator:
            operator acting in Hilbert space of `subsystem`; if str, then this should be an operator name in
            the subsystem, typically not in eigenbasis
        subsystem:
            subsystem where diagonal operator is defined
        op_in_eigenbasis:
            whether `operator` is given in the `subsystem` eigenbasis; otherwise, the internal QuantumSys basis is
            assumed
        evecs:
            internal QuantumSys eigenstates, used to convert `operator` into eigenbasis
        """
        subsys_operator = spec_utils.convert_operator_to_qobj(
            operator, subsystem, op_in_eigenbasis, evecs
        )
        operator_identitywrap_list = [
            qt.operators.qeye(the_subsys.truncated_dim) for the_subsys in self
        ]
        subsystem_index = self.get_subsys_index(subsystem)
        operator_identitywrap_list[subsystem_index] = subsys_operator
        return qt.tensor(operator_identitywrap_list)

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
        return self.identity_wrap(diag_matrix, subsystem)

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
        return self.identity_wrap(operator, subsystem)

    def annihilate(self, subsystem: QuantumSys) -> Qobj:
        """Annihilation operator a for `subsystem`

        Parameters
        ----------
        subsystem:
            specifies subsystem in which annihilation operator acts
        """
        dim = subsystem.truncated_dim
        operator = qt.destroy(dim)
        return self.identity_wrap(operator, subsystem)

    ###############################################################################################
    # HilbertSpace: spectrum sweep
    ###############################################################################################
    def get_spectrum_vs_paramvals(
        self,
        param_vals: ndarray,
        update_hilbertspace: Callable,
        evals_count: int = 10,
        get_eigenstates: bool = False,
        param_name: str = "external_parameter",
        num_cpus: int = settings.NUM_CPUS,
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
