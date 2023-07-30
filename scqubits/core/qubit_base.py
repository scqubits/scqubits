# qubit_base.py
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
"""
Provides the base classes for qubits
"""

import functools
import inspect
import os

from abc import ABC, ABCMeta, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    overload,
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from scipy.sparse import csc_matrix, dia_matrix

import scqubits.core.constants as constants
import scqubits.core.descriptors as descriptors
import scqubits.core.diag as diag
import scqubits.core.units as units
import scqubits.settings as settings
import scqubits.ui.qubit_widget as ui
import scqubits.utils.plotting as plot

from scqubits.core.central_dispatch import DispatchClient
from scqubits.core.discretization import Grid1d
from scqubits.core.storage import DataStore, SpectrumData
from scqubits.settings import IN_IPYTHON, matplotlib_settings
from scqubits.utils.cpu_switch import get_map_method
from scqubits.utils.misc import InfoBar, process_which
from scqubits.utils.spectrum_utils import (
    get_matrixelement_table,
    order_eigensystem,
    recast_esys_mapdata,
    standardize_sign,
)

if IN_IPYTHON:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

if TYPE_CHECKING:
    from typing_extensions import Literal

    from scqubits.core.storage import WaveFunction


LevelsTuple = Tuple[int, ...]
Transition = Tuple[int, int]
TransitionsTuple = Tuple[Transition, ...]

# -Generic quantum system container and Qubit base class------------------------------


class QuantumSystem(DispatchClient, ABC):
    """Generic quantum system class"""

    truncated_dim = descriptors.WatchedProperty(int, "QUANTUMSYSTEM_UPDATE")
    _init_params: List[str]
    _image_filename: str
    _sys_type: str

    # To facilitate warnings in set_units, introduce a counter keeping track of the
    # number of QuantumSystem instances
    _quantumsystem_counter: int = 0
    # To enable autogeneration of id_str, keep a record of all subclass types and
    # corresponding counts of instances
    _instance_counter: Dict[str, int] = {}

    _subclasses: List[ABCMeta] = []

    def __new__(cls, *args, **kwargs) -> "QuantumSystem":
        QuantumSystem._quantumsystem_counter += 1

        if cls.__name__ not in QuantumSystem._instance_counter:
            QuantumSystem._instance_counter[cls.__name__] = 1
        else:
            QuantumSystem._instance_counter[cls.__name__] += 1

        return super().__new__(cls)

    def __del__(self) -> None:
        # The following if clause mitigates an issue where upon program exit calls to
        # this destructor fail because `QuantumSystem` is of NoneType. (Upon program
        # exit, does the class itself get deleted before class instances are calling
        # their destructor?)
        try:
            QuantumSystem._quantumsystem_counter -= 1
        except (NameError, AttributeError):
            pass

    def __init__(self, id_str: Union[str, None]):
        self._sys_type = type(self).__name__
        self._id_str = id_str or self._autogenerate_id_str()
        self._image_filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "qubit_img",
            type(self).__name__ + ".jpg",
        )

    def __init_subclass__(cls):
        """Used to register all non-abstract _subclasses as a list in
        `QuantumSystem._subclasses`."""
        super().__init_subclass__()
        if not inspect.isabstract(cls):
            cls._subclasses.append(cls)

    def __repr__(self) -> str:
        if hasattr(self, "_init_params"):
            init_names = self._init_params
        else:
            init_names = list(inspect.signature(self.__init__).parameters.keys())[1:]  # type: ignore
        init_dict = {name: getattr(self, name) for name in init_names}
        return type(self).__name__ + f"(**{init_dict!r})"

    def __str__(self) -> str:
        indent_length = 20
        name_prepend = self._sys_type.ljust(indent_length, "-") + "| [{}]\n".format(
            self._id_str
        )

        output = ""
        for param_name in self.default_params().keys():
            output += "{0}| {1}: {2}\n".format(
                " " * indent_length, str(param_name), str(getattr(self, param_name))
            )
        output += "{0}|\n".format(" " * indent_length)
        output += "{0}| dim: {1}\n".format(" " * indent_length, str(self.hilbertdim()))

        return name_prepend + output

    def __eq__(self, other: Any):
        if not isinstance(other, type(self)):
            return False
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return super().__hash__()

    def _autogenerate_id_str(self):
        name = self._sys_type
        return "{}_{}".format(name, QuantumSystem._instance_counter[name])

    @property
    def id_str(self):
        return self._id_str

    def get_initdata(self) -> Dict[str, Any]:
        """Returns dict appropriate for creating/initializing a new Serializable
        object."""
        EXCLUDE = [
            "evals_method",
            "evals_method_options",
            "esys_method",
            "esys_method_options",
        ]
        initdata = {
            name: getattr(self, name)
            for name in self._init_params
            if name not in EXCLUDE
        }
        return initdata

    @abstractmethod
    def hilbertdim(self) -> int:
        """Returns dimension of Hilbert space"""

    @classmethod
    def get_operator_names(cls) -> List[str]:
        """Returns a list of all operator names for the quantum system.
        Note that this list omits any operators that start with "_".

        Parameters
        ----------
        subsys:
            Class instance of quantum system

        Returns
        -------
            list of operator names
        """
        operator_list = []
        for name, val in inspect.getmembers(cls):
            if "operator" in name and name[0] != "_" and name != "get_operator_names":
                operator_list.append(name)
        return operator_list

    @classmethod
    def create(cls) -> "QuantumSystem":
        """Use ipywidgets to create a new class instance"""
        init_params = cls.default_params()
        instance = cls(**init_params)
        instance.widget()
        return instance

    def widget(self, params: Optional[Dict[str, Any]] = None):
        """Use ipywidgets to modify parameters of class instance"""
        init_params = params or self.get_initdata()
        init_params.pop("id_str", None)
        ui.create_widget(
            self.set_params_from_gui, init_params, image_filename=self._image_filename
        )

    @staticmethod
    @abstractmethod
    def default_params() -> Dict[str, Any]:
        """Return dictionary with default parameter values for initialization of
        class instance"""

    def set_params_from_gui(self, change):
        """
        Set new parameters through the provided dictionary.
        """
        param_name = change["owner"].name
        param_val = change["owner"].num_value
        setattr(self, param_name, param_val)

    def set_params(self, **kwargs):
        """
        Set new parameters through the provided dictionary.
        """
        for param_name, param_val in kwargs.items():
            setattr(self, param_name, param_val)

    def supported_noise_channels(self) -> List:
        """
        Returns a list of noise channels this QuantumSystem supports. If none,
        return an empty list.
        """
        return []


# -QubitBaseClass-------------------------------------------------------------------------------------------------------


class QubitBaseClass(QuantumSystem, ABC):
    """Base class for superconducting qubit objects. Provide general mechanisms and
    routines for plotting spectra, matrix elements, and writing data to files
    """

    # see PEP 526 https://www.python.org/dev/peps/pep-0526/#class-and-instance-variable-annotations
    truncated_dim: int  # type:ignore
    _default_grid: Grid1d
    _sys_type: str
    _init_params: list

    def __init__(
        self,
        id_str: Union[str, None],
        evals_method: Union[str, None] = None,
        evals_method_options: Union[Dict, None] = None,
        esys_method: Union[str, None] = None,
        esys_method_options: Union[Dict, None] = None,
    ):
        super().__init__(id_str=id_str)
        self.evals_method = evals_method
        self.evals_method_options = evals_method_options
        self.esys_method = esys_method
        self.esys_method_options = esys_method_options

    @abstractmethod
    def hamiltonian(self):
        """Returns the Hamiltonian"""

    def _evals_calc(self, evals_count: int) -> ndarray:
        hamiltonian_mat = self.hamiltonian()
        evals = sp.linalg.eigh(
            hamiltonian_mat,
            eigvals_only=True,
            subset_by_index=(0, evals_count - 1),
            check_finite=False,
        )
        return np.sort(evals)

    def _esys_calc(self, evals_count: int) -> Tuple[ndarray, ndarray]:
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = sp.linalg.eigh(
            hamiltonian_mat,
            eigvals_only=False,
            subset_by_index=(0, evals_count - 1),
            check_finite=False,
        )
        evals, evecs = order_eigensystem(evals, evecs)
        return evals, evecs

    @overload
    def eigenvals(
        self,
        evals_count: int = 6,
        filename: Optional[str] = None,
        return_spectrumdata: "Literal[False]" = False,
    ) -> ndarray:
        ...

    @overload
    def eigenvals(
        self,
        evals_count: int,
        filename: str,
        return_spectrumdata: "Literal[True]",
    ) -> SpectrumData:
        ...

    def eigenvals(
        self,
        evals_count: int = 6,
        filename: Optional[str] = None,
        return_spectrumdata: bool = False,
    ) -> Union[SpectrumData, ndarray]:
        """Calculates eigenvalues using `scipy.linalg.eigh`, returns numpy array of
        eigenvalues.

        Parameters
        ----------
        evals_count:
            number of desired eigenvalues/eigenstates (default value = 6)
        filename:
            path and filename without suffix, if file output desired
            (default value = None)
        return_spectrumdata:
            if set to true, the returned data is provided as a SpectrumData object
            (default value = False)

        Returns
        -------
            eigenvalues as ndarray or in form of a SpectrumData object
        """
        if not hasattr(self, "evals_method") or self.evals_method is None:
            evals = self._evals_calc(evals_count)
        else:
            diagonalizer = (
                diag.DIAG_METHODS[self.evals_method]
                if isinstance(self.evals_method, str)
                else self.evals_method
            )
            options = (
                {} if self.esys_method_options is None else self.esys_method_options
            )
            evals = diagonalizer(self.hamiltonian(), evals_count, **options)

        if filename or return_spectrumdata:
            specdata = SpectrumData(
                energy_table=evals, system_params=self.get_initdata()
            )
        if filename:
            specdata.filewrite(filename)
        return specdata if return_spectrumdata else evals

    @overload
    def eigensys(
        self,
        evals_count: int = 6,
        filename: Optional[str] = None,
        return_spectrumdata: "Literal[False]" = False,
    ) -> Tuple[ndarray, ndarray]:
        ...

    @overload
    def eigensys(
        self,
        evals_count: int,
        filename: Optional[str],
        return_spectrumdata: "Literal[True]",
    ) -> SpectrumData:
        ...

    def eigensys(
        self,
        evals_count: int = 6,
        filename: Optional[str] = None,
        return_spectrumdata: bool = False,
    ) -> Union[Tuple[ndarray, ndarray], SpectrumData]:
        """Calculates eigenvalues and corresponding eigenvectors using
        `scipy.linalg.eigh`. Returns two numpy arrays containing the eigenvalues and
        eigenvectors, respectively.

        Parameters
        ----------
        evals_count:
            number of desired eigenvalues/eigenstates (default value = 6)
        filename:
            path and filename without suffix, if file output desired
            (default value = None)
        return_spectrumdata: if set to true, the returned data is provided as a SpectrumData object
            (default value = False)

        Returns
        -------
            eigenvalues, eigenvectors as numpy arrays or in form of a SpectrumData object
        """
        if not hasattr(self, "esys_method") or self.esys_method is None:
            evals, evecs = self._esys_calc(evals_count)
        else:
            diagonalizer = (
                diag.DIAG_METHODS[self.esys_method]
                if isinstance(self.esys_method, str)
                else self.esys_method
            )
            options = (
                {} if self.esys_method_options is None else self.esys_method_options
            )
            evals, evecs = diagonalizer(self.hamiltonian(), evals_count, **options)

        if filename or return_spectrumdata:
            specdata = SpectrumData(
                energy_table=evals, system_params=self.get_initdata(), state_table=evecs
            )
        if filename:
            specdata.filewrite(filename)
        return specdata if return_spectrumdata else (evals, evecs)

    def process_op(
        self,
        native_op: Union[ndarray, csc_matrix],
        energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False,
    ) -> Union[ndarray, csc_matrix]:
        """Processes the operator `native_op`: either hand back `native_op` unchanged, or transform it into the
        energy eigenbasis. (Native basis refers to the basis used internally by each qubit, e.g., charge basis in the
        case of `Transmon`.

        Parameters
        ----------
        native_op:
            operator in native basis
        energy_esys:
            If `False` (default), returns operator in the native basis
            If `True`, the energy eigenspectrum is computed, returns operator in the energy eigenbasis
            if energy_esys is the energy eigenspectrum, in the form of a tuple containing two ndarrays
            (eigenvalues and energy eigenvectors), returns operator in the energy eigenbasis,
            and does not have to recalculate eigenspectrum.

        Returns
        -------
            `native_op` either unchanged or transformed into eigenenergy basis
        """
        if isinstance(energy_esys, bool):
            if not energy_esys:
                return native_op
            esys = self.eigensys(evals_count=self.truncated_dim)
        else:
            esys = energy_esys
        evectors = esys[1][:, : self.truncated_dim]
        return get_matrixelement_table(native_op, evectors)

    def process_hamiltonian(
        self,
        native_hamiltonian: Union[ndarray, csc_matrix],
        energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False,
    ) -> Union[ndarray, csc_matrix]:
        """Return qubit Hamiltonian in chosen basis: either return unchanged (i.e., in native basis) or transform
        into eigenenergy basis

        Parameters
        ----------
        native_hamiltonian:
            Hamiltonian in native basis
        energy_esys:
            If `False` (default), returns Hamiltonian in the native basis
            If `True`, the energy eigenspectrum is computed, returns Hamiltonian in the energy eigenbasis
            if energy_esys is the energy eigenspectrum, in the form of a tuple containing two ndarrays
            (eigenvalues and energy eigenvectors), returns Hamiltonian in the energy eigenbasis,
            and does not have to recalculate eigenspectrum.

        Returns
        -------
            Hamiltonian, either unchanged in native basis, or transformed into eigenenergy basis
        """
        if isinstance(energy_esys, bool):
            if not energy_esys:
                return native_hamiltonian
            esys = self.eigensys(evals_count=self.truncated_dim)
        else:
            esys = energy_esys
        evals = esys[0][: self.truncated_dim]
        if isinstance(native_hamiltonian, ndarray):
            return np.diag(evals)
        return dia_matrix(evals).tocsc()

    def anharmonicity(self) -> float:
        """Returns the qubit's anharmonicity, (E_2 - E_1) - (E_1 - E_0)."""
        energies = self.eigenvals(evals_count=3)
        return energies[2] - 2 * energies[1] + energies[0]

    def E01(self) -> float:
        """Returns the qubit's fundamental energy splitting, E_1 - E_0."""
        energies = self.eigenvals(evals_count=2)
        return energies[1] - energies[0]

    @overload
    def matrixelement_table(
        self,
        operator: str,
        evecs: ndarray = None,
        evals_count: int = 6,
        filename: str = None,
        return_datastore: "Literal[False]" = False,
    ) -> ndarray:
        ...

    @overload
    def matrixelement_table(
        self,
        operator: str,
        evecs: ndarray,
        evals_count: int,
        filename: Optional[str],
        return_datastore: "Literal[True]",
    ) -> DataStore:
        ...

    def matrixelement_table(
        self,
        operator: str,
        evecs: ndarray = None,
        evals_count: int = 6,
        filename: str = None,
        return_datastore: bool = False,
    ) -> Union[DataStore, ndarray]:
        """Returns table of matrix elements for `operator` with respect to the
        eigenstates of the qubit. The operator is given as a string matching a class
        method returning an operator matrix. E.g., for an instance `trm` of Transmon,
        the matrix element table for the charge operator is given by
        `trm.op_matrixelement_table('n_operator')`. When `esys` is set to `None`,
        the eigensystem is calculated on-the-fly.

        Parameters
        ----------
        operator:
            name of class method in string form, returning operator matrix in
            qubit-internal basis.
        evecs:
            if not provided, then the necessary eigenstates are calculated on the fly
        evals_count:
            number of desired matrix elements, starting with ground state
            (default value = 6)
        filename:
            output file name
        return_datastore:
            if set to true, the returned data is provided as a DataStore object
            (default value = False)
        """
        if evecs is None:
            _, evecs = self.eigensys(evals_count=evals_count)
        operator_matrix = getattr(self, operator)()
        table = get_matrixelement_table(operator_matrix, evecs)
        if filename or return_datastore:
            data_store = DataStore(
                system_params=self.get_initdata(), matrixelem_table=table
            )
        if filename:
            data_store.filewrite(filename)
        return data_store if return_datastore else table

    def _esys_for_paramval(
        self, paramval: float, param_name: str, evals_count: int
    ) -> Tuple[ndarray, ndarray]:
        setattr(self, param_name, paramval)
        return self.eigensys(evals_count=evals_count)

    def _evals_for_paramval(
        self, paramval: float, param_name: str, evals_count: int
    ) -> ndarray:
        setattr(self, param_name, paramval)
        return self.eigenvals(evals_count)

    def get_spectrum_vs_paramvals(
        self,
        param_name: str,
        param_vals: ndarray,
        evals_count: int = 6,
        subtract_ground: bool = False,
        get_eigenstates: bool = False,
        filename: str = None,
        num_cpus: Optional[int] = None,
    ) -> SpectrumData:
        """Calculates eigenvalues/eigenstates for a varying system parameter,
        given an array of parameter values. Returns a `SpectrumData` object with
        `energy_data[n]` containing eigenvalues calculated for parameter value
        `param_vals[n]`.

        Parameters
        ----------
        param_name:
            name of parameter to be varied
        param_vals:
            parameter values to be plugged in
        evals_count:
            number of desired eigenvalues (sorted from smallest to largest)
            (default value = 6)
        subtract_ground:
            if True, eigenvalues are returned relative to the ground state eigenvalue
            (default value = False)
        get_eigenstates:
            return eigenstates along with eigenvalues (default value = False)
        filename:
            file name if direct output to disk is wanted
        num_cpus:
            number of cores to be used for computation
            (default value: settings.NUM_CPUS)
        """
        num_cpus = num_cpus or settings.NUM_CPUS
        previous_paramval = getattr(self, param_name)
        tqdm_disable = num_cpus > 1 or settings.PROGRESSBAR_DISABLED

        target_map = get_map_method(num_cpus)
        if not get_eigenstates:
            func_evals = functools.partial(
                self._evals_for_paramval, param_name=param_name, evals_count=evals_count
            )
            with InfoBar(
                "Parallel computation of eigensystems [num_cpus={}]".format(num_cpus),
                num_cpus,
            ):
                eigenvalue_table = np.asarray(
                    list(
                        target_map(
                            func_evals,
                            tqdm(
                                param_vals,
                                desc="Spectral data",
                                leave=False,
                                disable=tqdm_disable,
                            ),
                        )
                    )
                )
            eigenstate_table = None
        else:
            func_esys = functools.partial(
                self._esys_for_paramval, param_name=param_name, evals_count=evals_count
            )
            with InfoBar(
                "Parallel computation of eigenvalues [num_cpus={}]".format(num_cpus),
                num_cpus,
            ):
                # Note that it is useful here that the outermost eigenstate object is
                # a list, as for certain applications the necessary hilbert space
                # dimension can vary with paramvals
                eigensystem_mapdata = list(
                    target_map(
                        func_esys,
                        tqdm(
                            param_vals,
                            desc="Spectral data",
                            leave=False,
                            disable=tqdm_disable,
                        ),
                    )
                )
            eigenvalue_table, eigenstate_table = recast_esys_mapdata(
                eigensystem_mapdata
            )

        if subtract_ground:
            for param_index, _ in enumerate(param_vals):
                eigenvalue_table[param_index] -= eigenvalue_table[param_index][0]

        setattr(self, param_name, previous_paramval)
        specdata = SpectrumData(
            eigenvalue_table,
            self.get_initdata(),
            param_name,
            param_vals,
            state_table=eigenstate_table,
        )
        if filename:
            specdata.filewrite(filename)
        return specdata

    def _compute_dispersion(
        self,
        dispersion_name: str,
        param_name: str,
        param_vals: ndarray,
        transitions_tuple: TransitionsTuple = ((0, 1),),
        levels_tuple: Optional[LevelsTuple] = None,
        point_count: int = 50,
        num_cpus: Optional[int] = None,
    ) -> Tuple[ndarray, ndarray]:
        from scqubits import HilbertSpace, ParameterSweep

        hilbertspace = HilbertSpace(subsystem_list=[self])

        paramvals_by_name = {
            dispersion_name: np.linspace(0.0, 1.0, point_count),
            param_name: param_vals,
        }

        def update_func(disp_val, sweep_val):
            setattr(self, dispersion_name, disp_val)
            setattr(self, param_name, sweep_val)

        previous_dispval = getattr(self, dispersion_name)
        previous_paramval = getattr(self, param_name)
        max_level = (
            np.max(transitions_tuple) if not levels_tuple else np.max(levels_tuple)
        )
        sweep = ParameterSweep(
            hilbertspace,
            paramvals_by_name,
            update_func,
            evals_count=max_level + 1,
            bare_only=True,
            num_cpus=num_cpus,
        )
        eigenenergies = sweep["bare_evals"]["subsys":0].toarray()  # type:ignore

        if levels_tuple is None:
            dispersions = np.empty((len(transitions_tuple), len(param_vals)))
            for index, (i, j) in enumerate(transitions_tuple):
                energy_ij = eigenenergies[:, :, i] - eigenenergies[:, :, j]
                dispersions[index] = np.max(energy_ij, axis=0) - np.min(
                    energy_ij, axis=0
                )
        else:
            dispersions = np.empty((len(levels_tuple), len(param_vals)))
            for index, j in enumerate(levels_tuple):
                energy_j = eigenenergies[:, :, j]
                dispersions[index] = np.max(energy_j, axis=0) - np.min(energy_j, axis=0)

        setattr(self, param_name, previous_paramval)
        setattr(self, dispersion_name, previous_dispval)
        return eigenenergies, dispersions

    def get_dispersion_vs_paramvals(
        self,
        dispersion_name: str,
        param_name: str,
        param_vals: ndarray,
        ref_param: Optional[str] = None,
        transitions: Union[Transition, TransitionsTuple] = (0, 1),
        levels: Optional[Union[int, LevelsTuple]] = None,
        point_count: int = 50,
        num_cpus: Optional[int] = None,
    ) -> SpectrumData:
        """Calculates eigenvalues/eigenstates for a varying system parameter,
        given an array of parameter values. Returns a `SpectrumData` object with
        `energy_data[n]` containing eigenvalues calculated for parameter value
        `param_vals[n]`.

        Parameters
        ----------
        dispersion_name:
            parameter inducing the dispersion, typically 'ng' or 'flux' (will be
            scanned over range from 0 to 1)
        param_name:
            name of parameter to be varied
        param_vals:
            parameter values to be plugged in
        ref_param:
            optional, name of parameter to use as reference for the parameter value;
            e.g., to compute charge dispersion vs. EJ/EC, use EJ as param_name and
            EC as ref_param
        transitions:
            integer tuple or tuples specifying for which transitions dispersion is to
            be calculated
            (default: = (0,1))
        levels:
            tuple specifying levels (rather than transitions) for which dispersion
            should be plotted; overrides transitions parameter when given
        point_count:
            number of points scanned for the dispersion parameter for determining min
            and max values of transition energies (default: 50)
        num_cpus:
            number of cores to be used for computation
            (default value: settings.NUM_CPUS)
        """
        if levels is not None:
            if isinstance(levels, int):
                # presence of levels argument will overwrite `transitions`;
                # here: single level
                levels_tuple: Optional[LevelsTuple] = (levels,)
                transitions_tuple: TransitionsTuple = (transitions,)  # type:ignore
            elif isinstance(levels, tuple):
                # presence of levels argument will overwrite `transitions`;
                # here: multiple levels
                levels_tuple: Optional[LevelsTuple] = levels
                transitions_tuple: TransitionsTuple = (transitions,)  # type:ignore
            else:
                raise ValueError(
                    "Invalid `levels` specification: expect int or tuple " "of int"
                )
        elif isinstance(transitions[0], int):
            # transitions is inferred to be of form (i, j), so only a single one
            transitions_tuple = (transitions,)  # type:ignore
            levels_tuple = None
        elif isinstance(transitions[0], tuple):
            # transitions is inferred to be of form ((i1, j1), ...) ,
            # there are multiple transitions
            transitions_tuple = transitions
            levels_tuple = None
        else:
            raise ValueError(
                "Invalid `transitions` specification: expect either ("
                "int, int)  or ((int, int), ...)"
            )

        eigenenergies, dispersion = self._compute_dispersion(
            dispersion_name,
            param_name,
            param_vals,
            transitions_tuple=transitions_tuple,
            levels_tuple=levels_tuple,
            point_count=point_count,
            num_cpus=num_cpus,
        )

        if ref_param is not None:
            param_name += "/" + ref_param
            param_vals /= getattr(self, ref_param)

        specdata = SpectrumData(
            eigenenergies,
            self.get_initdata(),
            param_name,
            param_vals,
            labels=levels_tuple or transitions_tuple,
            dispersion=dispersion.T,
        )
        return specdata

    def get_matelements_vs_paramvals(
        self,
        operator: str,
        param_name: str,
        param_vals: ndarray,
        evals_count: int = 6,
        num_cpus: Optional[int] = None,
    ) -> SpectrumData:
        """Calculates matrix elements for a varying system parameter, given an array
        of parameter values. Returns a `SpectrumData` object containing matrix
        element data, eigenvalue data, and eigenstate data..

        Parameters
        ----------
        operator:
            name of class method in string form, returning operator matrix
        param_name:
            name of parameter to be varied
        param_vals:
            parameter values to be plugged in
        evals_count:
            number of desired eigenvalues (sorted from smallest to largest)
            (default value = 6)
        num_cpus:
            number of cores to be used for computation
            (default value: settings.NUM_CPUS)
        """
        num_cpus = num_cpus or settings.NUM_CPUS
        spectrumdata = self.get_spectrum_vs_paramvals(
            param_name,
            param_vals,
            evals_count=evals_count,
            get_eigenstates=True,
            num_cpus=num_cpus,
        )
        paramvals_count = len(param_vals)
        matelem_table = np.empty(
            shape=(paramvals_count, evals_count, evals_count), dtype=np.complex_
        )

        paramval_before = getattr(self, param_name)
        assert spectrumdata.state_table is not None
        for index, paramval in enumerate(param_vals):
            evecs = spectrumdata.state_table[index]
            setattr(self, param_name, paramval)
            matelem_table[index] = self.matrixelement_table(
                operator, evecs=evecs, evals_count=evals_count
            )
        setattr(self, param_name, paramval_before)

        spectrumdata.matrixelem_table = matelem_table
        return spectrumdata

    @mpl.rc_context(matplotlib_settings)
    def plot_evals_vs_paramvals(
        self,
        param_name: str,
        param_vals: ndarray,
        evals_count: int = 6,
        subtract_ground: bool = False,
        num_cpus: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """Generates a simple plot of a set of eigenvalues as a function of one
        parameter. The individual points correspond to the a provided array of
        parameter values.

        Parameters
        ----------
        param_name:
            name of parameter to be varied
        param_vals:
            parameter values to be plugged in
        evals_count:
            number of desired eigenvalues (sorted from smallest to largest)
            (default value = 6)
        subtract_ground:
            whether to subtract ground state energy from all eigenvalues
            (default value = False)
        num_cpus:
            number of cores to be used for computation
            (default value: settings.NUM_CPUS)
        **kwargs:
            standard plotting option (see separate documentation)
        """
        num_cpus = num_cpus or settings.NUM_CPUS
        specdata = self.get_spectrum_vs_paramvals(
            param_name,
            param_vals,
            evals_count=evals_count,
            subtract_ground=subtract_ground,
            num_cpus=num_cpus,
        )
        return plot.evals_vs_paramvals(specdata, which=range(evals_count), **kwargs)

    @mpl.rc_context(matplotlib_settings)
    def plot_dispersion_vs_paramvals(
        self,
        dispersion_name: str,
        param_name: str,
        param_vals: ndarray,
        ref_param: Optional[str] = None,
        transitions: Union[Transition, TransitionsTuple] = (0, 1),
        levels: Optional[Union[int, LevelsTuple]] = None,
        point_count: int = 50,
        num_cpus: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """Generates a simple plot of a set of curves representing the charge or flux
        dispersion of transition energies.

        Parameters
        ----------
        dispersion_name:
            parameter inducing the dispersion, typically 'ng' or 'flux' (will be
            scanned over range from 0 to 1)
        param_name:
            name of parameter to be varied
        param_vals:
            parameter values to be plugged in
        ref_param:
            optional, name of parameter to use as reference for the parameter value;
            e.g., to compute charge dispersion vs. EJ/EC, use EJ as param_name and
            EC as ref_param
        transitions:
            integer tuple or tuples specifying for which transitions dispersion is to
            be calculated
            (default: = (0,1))
        levels:
            int or tuple specifying level(s) (rather than transitions) for which
            dispersion should be plotted; overrides transitions parameter when given
        point_count:
            number of points scanned for the dispersion parameter for determining min
            and max values of transition energies (default: 50)
        num_cpus:
            number of cores to be used for computation
            (default value: settings.NUM_CPUS)
        **kwargs:
            standard plotting option (see separate documentation)
        """
        specdata = self.get_dispersion_vs_paramvals(
            dispersion_name,
            param_name,
            param_vals,
            ref_param=ref_param,
            transitions=transitions,
            levels=levels,
            point_count=point_count,
            num_cpus=num_cpus,
        )
        if levels is not None:
            levels_tuple = levels if isinstance(levels, tuple) else (levels,)
            label_list = [str(j) for j in levels_tuple]
        else:
            transitions_tuple: TransitionsTuple = (
                transitions  # type:ignore
                if isinstance(transitions[0], tuple)
                else (transitions,)
            )
            label_list = ["{}{}".format(i, j) for i, j in transitions_tuple]

        return plot.data_vs_paramvals(
            xdata=specdata.param_vals,  # type:ignore
            ydata=specdata.dispersion,  # type:ignore
            label_list=label_list,
            xlabel=specdata.param_name,
            ylabel="energy dispersion [{}]".format(units.get_units()),
            yscale="log",
            **kwargs,
        )

    @mpl.rc_context(matplotlib_settings)
    def plot_matrixelements(
        self,
        operator: str,
        evecs: ndarray = None,
        evals_count: int = 6,
        mode: str = "abs",
        show_numbers: bool = False,
        show3d: bool = True,
        **kwargs,
    ) -> Union[Tuple[Figure, Tuple[Axes, Axes]], Tuple[Figure, Axes]]:
        """Plots matrix elements for `operator`, given as a string referring to a
        class method that returns an operator matrix. E.g., for instance `trm` of
        Transmon, the matrix element plot for the charge operator `n` is obtained by
        `trm.plot_matrixelements('n')`. When `esys` is set to None, the eigensystem
        with `which` eigenvectors is calculated.

        Parameters
        ----------
        operator:
            name of class method in string form, returning operator matrix
        evecs:
            eigensystem data of evals, evecs; eigensystem will be calculated if set to
            None (default value = None)
        evals_count:
            number of desired matrix elements, starting with ground state
            (default value = 6)
        mode:
            idx_entry from MODE_FUNC_DICTIONARY, e.g., `'abs'` for absolute value (default)
        show_numbers:
            determines whether matrix element values are printed on top of the plot
            (default: False)
        show3d:
            whether to show a 3d skyscraper plot of the matrix alongside the 2d plot
            (default: True)
        **kwargs:
            standard plotting option (see separate documentation)
        """
        matrixelem_array = self.matrixelement_table(operator, evecs, evals_count)
        assert isinstance(matrixelem_array, np.ndarray)
        if not show3d:
            return plot.matrix2d(
                matrixelem_array,
                mode=mode,
                show_numbers=show_numbers,
                **kwargs,
            )
        return plot.matrix(
            matrixelem_array,
            mode=mode,
            show_numbers=show_numbers,
            **kwargs,
        )

    @mpl.rc_context(matplotlib_settings)
    def plot_matelem_vs_paramvals(
        self,
        operator: str,
        param_name: str,
        param_vals: ndarray,
        select_elems: Union[int, List[Tuple[int, int]]] = 4,
        mode: str = "abs",
        num_cpus: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """Generates a simple plot of a set of eigenvalues as a function of one
        parameter. The individual points correspond to the a provided array of
        parameter values.

        Parameters
        ----------
        operator:
            name of class method in string form, returning operator matrix
        param_name:
            name of parameter to be varied
        param_vals:
            parameter values to be plugged in
        select_elems:
            either maximum index of desired matrix elements, or
            list [(i1, i2), (i3, i4), ...] of index tuples
            for specific desired matrix elements (default value = 4)
        mode:
            idx_entry from MODE_FUNC_DICTIONARY, e.g., `'abs'` for absolute value
            (default value = 'abs')
        num_cpus:
            number of cores to be used for computation
            (default value: settings.NUM_CPUS)
        **kwargs:
            standard plotting option (see separate documentation)
        """
        num_cpus = num_cpus or settings.NUM_CPUS
        if isinstance(select_elems, int):
            evals_count = select_elems
        else:
            flattened_list = [index for tupl in select_elems for index in tupl]
            evals_count = max(flattened_list) + 1

        specdata = self.get_matelements_vs_paramvals(
            operator, param_name, param_vals, evals_count=evals_count, num_cpus=num_cpus
        )
        return plot.matelem_vs_paramvals(
            specdata, select_elems=select_elems, mode=mode, **kwargs
        )

    def set_and_return(self, attr_name: str, value: Any) -> "QubitBaseClass":
        """
        Allows to set an attribute after which self is returned. This is useful for
        doing something like example::

            qubit.set_and_return('flux', 0.23).some_method()

        instead of example::

            qubit.flux=0.23
            qubit.some_method()

        Parameters
        ----------
        attr_name:
            name of class attribute in string form
        value:
            value that the attribute is to be set to

        Returns
        -------
            self
        """
        setattr(self, attr_name, value)
        return self


# -QubitBaseClass1d------------------------------------------------------------------


class QubitBaseClass1d(QubitBaseClass):
    """Base class for superconducting qubit objects with one degree of freedom.
    Provide general mechanisms and routines for plotting spectra, matrix elements,
    and writing data to files.
    """

    # see PEP 526 https://www.python.org/dev/peps/pep-0526/#class-and-instance-variable-annotations
    _default_grid: Grid1d

    @abstractmethod
    def potential(self, phi: Union[float, ndarray]) -> Union[float, ndarray]:
        pass

    @abstractmethod
    def wavefunction(
        self,
        esys: Optional[Tuple[ndarray, ndarray]],
        which: int = 0,
        phi_grid: Grid1d = None,
    ) -> "WaveFunction":
        pass

    def wavefunction1d_defaults(
        self, mode: str, evals: ndarray, wavefunc_count: int
    ) -> Dict[str, Any]:
        """Plot defaults for plotting.wavefunction1d.

        Parameters
        ----------
        mode:
            amplitude modifier, needed to give the correct default y label
        evals:
            eigenvalues to include in plot
        wavefunc_count:
            number of wave functions to be plotted
        """
        ylabel = r"$\psi_j(\varphi)$"
        ylabel = constants.MODE_STR_DICT[mode](ylabel)
        ylabel += ",  energy [{}]".format(units.get_units())
        options = {"xlabel": r"$\varphi$", "ylabel": ylabel}
        return options

    @mpl.rc_context(matplotlib_settings)
    def plot_wavefunction(
        self,
        which: Union[int, Iterable[int]] = 0,
        mode: str = "real",
        esys: Tuple[ndarray, ndarray] = None,
        phi_grid: Grid1d = None,
        scaling: Optional[float] = None,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """Plot 1d phase-basis wave function(s). Must be overwritten by
        higher-dimensional qubits like FluxQubits and ZeroPi.

        Parameters
        ----------
        which:
            single index or tuple/list of integers indexing the wave function(s) to be
            plotted.
            If which is -1, all wavefunctions up to the truncation limit are plotted.
        mode:
            choices as specified in `constants.MODE_FUNC_DICT`
            (default value = 'abs_sqr')
        esys:
            eigenvalues, eigenvectors
        phi_grid:
            used for setting a custom grid for phi; if None use self._default_grid
        scaling:
            custom scaling of wave function amplitude/modulus
        **kwargs:
            standard plotting option (see separate documentation)
        """
        wavefunc_indices = process_which(which, self.truncated_dim)

        if esys is None:
            evals_count = max(wavefunc_indices) + 1
            esys = self.eigensys(evals_count=evals_count)
            evals, _ = esys
        else:
            evals, _ = esys

        energies = evals[list(wavefunc_indices)]

        phi_grid = phi_grid or self._default_grid
        potential_vals = self.potential(phi_grid.make_linspace())

        amplitude_modifier = constants.MODE_FUNC_DICT[mode]
        wavefunctions = []
        for wavefunc_index in wavefunc_indices:
            phi_wavefunc = self.wavefunction(
                esys, which=wavefunc_index, phi_grid=phi_grid
            )
            phi_wavefunc.amplitudes = standardize_sign(phi_wavefunc.amplitudes)
            phi_wavefunc.amplitudes = amplitude_modifier(phi_wavefunc.amplitudes)
            wavefunctions.append(phi_wavefunc)

        fig_ax = kwargs.get("fig_ax") or plt.subplots()
        kwargs["fig_ax"] = fig_ax
        kwargs = {
            **self.wavefunction1d_defaults(
                mode, evals, wavefunc_count=len(wavefunc_indices)  # type:ignore
            ),
            **kwargs,
        }
        # in merging the dictionaries in the previous line: if any duplicates,
        # later ones survive

        plot.wavefunction1d(
            wavefunctions,
            potential_vals=potential_vals,  # type:ignore
            offset=energies,
            scaling=scaling,
            **kwargs,
        )
        return fig_ax
