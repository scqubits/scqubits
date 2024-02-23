# zeropi_full.py
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

from typing import Any, Dict, List, Optional, Tuple, Union, Callable


import numpy as np

from numpy import ndarray
from scipy import sparse
from scipy.sparse import csc_matrix

import scqubits
import scqubits.core.central_dispatch as dispatch
import scqubits.core.descriptors as descriptors
import scqubits.core.discretization as discretization
import scqubits.core.operators as op
import scqubits.core.qubit_base as base
import scqubits.io_utils.fileio_serializers as serializers
import scqubits.ui.qubit_widget as ui
import scqubits.utils.spectrum_utils as spec_utils

from scqubits.core.discretization import Grid1d
from scqubits.core.noise import NoisySystem

# - ZeroPi noise class


class NoisyFullZeroPi(NoisySystem):
    pass


class FullZeroPi(base.QubitBaseClass, serializers.Serializable, NoisyFullZeroPi):
    r"""Zero-Pi qubit [Brooks2013]_ [Dempster2014]_ including coupling to the zeta mode.
    The circuit is described by the Hamiltonian
    :math:`H = H_{0-\pi} + H_\text{int} + H_\zeta`, where

    .. math::

        &H_{0-\pi} = -2E_\text{CJ}\partial_\phi^2+2E_{\text{C}\Sigma}(i\partial_\theta-n_g)^2
                     +2E_{C\Sigma}dC_J\,\partial_\phi\partial_\theta\\
        &\qquad\qquad\qquad+2E_{C\Sigma}(\delta C_J/C_J)\partial_\phi\partial_\theta
                     +2\,\delta E_J \sin\theta\sin(\phi-\varphi_\text{ext}/2)\\
        &H_\text{int} = 2E_{C\Sigma}dC\,\partial_\theta\partial_\zeta + E_L dE_L \phi\,\zeta\\
        &H_\zeta = E_{\zeta} a^\dagger a

    expressed in phase basis. The definition of the relevant charging energies
    :math:`E_\text{CJ}`, :math:`E_{\text{C}\Sigma}`,
    Josephson energies :math:`E_\text{J}`, inductive energies :math:`E_\text{L}`,
    and relative amounts of disorder :math:`dC_\text{J}`, :math:`dE_\text{J}`,
    :math:`dC`, :math:`dE_\text{L}` follows [Groszkowski2018]_. Internally,
    the ``FullZeroPi`` class formulates the Hamiltonian matrix via the
    product basis of the decoupled Zero-Pi qubit (see ``ZeroPi``)  on one hand, and the
    zeta LC oscillator on the other hand.

    Parameters
    ----------
    EJ:
        mean Josephson energy of the two junctions
    EL:
        inductive energy of the two (super-)inductors
    ECJ:
        charging energy associated with the two junctions
    EC:
        charging energy of the large shunting capacitances; set to `None` if `ECS` is
        provided instead
    dEJ:
        relative disorder in EJ, i.e., (EJ1-EJ2)/EJavg
    dEL:
        relative disorder in EL, i.e., (EL1-EL2)/ELavg
    dCJ:
        relative disorder of the junction capacitances, i.e., (CJ1-CJ2)/CJavg
    dC:
        relative disorder in large capacitances, i.e., (C1-C2)/Cavg
    ng:
        offset charge associated with theta
    zeropi_cutoff:
        cutoff in the number of states of the disordered zero-pi qubit
    zeta_cutoff:
        cutoff in the zeta oscillator basis (Fock state basis)
    flux:
        magnetic flux through the circuit loop, measured in units of flux quanta (h/2e)
    grid:
        specifies the range and spacing of the discretization lattice
    ncut:
        charge number cutoff for `n_theta`,  `n_theta = -ncut, ..., ncut`
    ECS:
        total charging energy including large shunting capacitances and junction
        capacitances; may be provided instead of EC
    truncated_dim:
        desired dimension of the truncated quantum system; expected: truncated_dim > 1
    id_str:
        optional string by which this instance can be referred to in `HilbertSpace`
        and `ParameterSweep`. If not provided, an id is auto-generated.
    esys_method: 
        method for esys diagonalization, callable or string representation 
    esys_method_options: 
        dictionary with esys diagonalization options 
    evals_method: 
        method for evals diagonalization, callable or string representation 
    evals_method_options: 
        dictionary with evals diagonalization options 
    """

    EJ = descriptors.WatchedProperty(
        float, "QUANTUMSYSTEM_UPDATE", inner_object_name="_zeropi"
    )
    EL = descriptors.WatchedProperty(
        float, "QUANTUMSYSTEM_UPDATE", inner_object_name="_zeropi"
    )
    ECJ = descriptors.WatchedProperty(
        float, "QUANTUMSYSTEM_UPDATE", inner_object_name="_zeropi"
    )
    EC = descriptors.WatchedProperty(
        float, "QUANTUMSYSTEM_UPDATE", inner_object_name="_zeropi"
    )
    ECS = descriptors.WatchedProperty(
        float, "QUANTUMSYSTEM_UPDATE", inner_object_name="_zeropi"
    )
    dEJ = descriptors.WatchedProperty(
        float, "QUANTUMSYSTEM_UPDATE", inner_object_name="_zeropi"
    )
    dCJ = descriptors.WatchedProperty(
        float, "QUANTUMSYSTEM_UPDATE", inner_object_name="_zeropi"
    )
    dC = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    dEL = descriptors.WatchedProperty(float, "QUANTUMSYSTEM_UPDATE")
    ng = descriptors.WatchedProperty(
        float, "QUANTUMSYSTEM_UPDATE", inner_object_name="_zeropi"
    )
    flux = descriptors.WatchedProperty(
        float, "QUANTUMSYSTEM_UPDATE", inner_object_name="_zeropi"
    )
    grid = descriptors.WatchedProperty(
        Grid1d, "QUANTUMSYSTEM_UPDATE", inner_object_name="_zeropi"
    )
    ncut = descriptors.WatchedProperty(
        int, "QUANTUMSYSTEM_UPDATE", inner_object_name="_zeropi"
    )
    zeropi_cutoff = descriptors.WatchedProperty(
        int,
        "QUANTUMSYSTEM_UPDATE",
        inner_object_name="_zeropi",
        attr_name="truncated_dim",
    )

    def __init__(
        self,
        EJ: float,
        EL: float,
        ECJ: float,
        EC: float,
        dEJ: float,
        dCJ: float,
        dC: float,
        dEL: float,
        flux: float,
        ng: float,
        zeropi_cutoff: int,
        zeta_cutoff: int,
        grid: Grid1d,
        ncut: int,
        ECS: float = None,
        truncated_dim: int = 6,
        id_str: Optional[str] = None,
        evals_method: Union[Callable, str, None] = None,
        evals_method_options: Union[dict, None] = None,
        esys_method: Union[Callable, str, None] = None,
        esys_method_options: Union[dict, None] = None,
    ) -> None:
        base.QuantumSystem.__init__(self, id_str=id_str)
        self._zeropi = scqubits.ZeroPi(
            EJ=EJ,
            EL=EL,
            ECJ=ECJ,
            EC=EC,
            ng=ng,
            flux=flux,
            grid=grid,
            ncut=ncut,
            dEJ=dEJ,
            dCJ=dCJ,
            ECS=ECS,
            # the zeropi_cutoff defines the truncated_dim of the "base" zeropi object
            truncated_dim=zeropi_cutoff,
            id_str=self._id_str + " [interior ZeroPi]",
            evals_method=evals_method,
            evals_method_options=evals_method_options,
            esys_method=esys_method,
            esys_method_options=esys_method_options,
        )
        self.dC = dC
        self.dEL = dEL
        self.zeta_cutoff = zeta_cutoff
        self.truncated_dim = truncated_dim
        self._init_params.remove(
            "ECS"
        )  # used for file IO Serializable purposes; remove ECS as init parameter

        dispatch.CENTRAL_DISPATCH.register("GRID_UPDATE", self)

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {
            "EJ": 10.0,
            "EL": 0.04,
            "ECJ": 20.0,
            "EC": 0.04,
            "dEJ": 0.05,
            "dCJ": 0.05,
            "dC": 0.08,
            "dEL": 0.05,
            "ng": 0.1,
            "flux": 0.23,
            "ncut": 30,
            "zeropi_cutoff": 10,
            "zeta_cutoff": 40,
            "truncated_dim": 10,
        }

    @classmethod
    def create(cls) -> "FullZeroPi":
        phi_grid = discretization.Grid1d(-25.0, 25.0, 360)
        init_params = cls.default_params()
        init_params["grid"] = phi_grid
        zeropi = cls(**init_params)
        zeropi.widget()
        return zeropi

    @classmethod
    def supported_noise_channels(cls) -> List[str]:
        """Return a list of supported noise channels"""
        return [
            "tphi_1_over_f_cc",
            "tphi_1_over_f_flux",
            "t1_flux_bias_line",
            # 't1_capacitive',
            "t1_inductive",
        ]

    def widget(self, params: Dict[str, Any] = None) -> None:
        init_params = params or self.get_initdata()
        init_params.pop("id_str", None)
        del init_params["grid"]
        init_params["grid_max_val"] = self.grid.max_val
        init_params["grid_min_val"] = self.grid.min_val
        init_params["grid_pt_count"] = self.grid.pt_count
        ui.create_widget(
            self.set_params_from_gui, init_params, image_filename=self._image_filename
        )

    def set_params_from_gui(self, **kwargs) -> None:
        for param_name, param_val in kwargs.items():
            if "grid_" in param_name:
                setattr(self.grid, param_name[5:], param_val)
            else:
                setattr(self, param_name, param_val)

    def receive(self, event: str, sender: object, **kwargs) -> None:
        if sender is self._zeropi.grid:
            self.broadcast("QUANTUMSYSTEM_UPDATE")

    def __str__(self) -> str:
        output = super().__str__()
        output = output[: output.rfind("\n")]
        output = output[: output.rfind("\n")]
        line_indent = output[output.rfind("\n") :]

        output += (
            line_indent
            + " dim: {0}   --[ (theta, phi): {1} total, {2} truncated;  (zeta): {3}"
            " ]--\n".format(
                self.hilbertdim(),
                self._zeropi.hilbertdim(),
                self.zeropi_cutoff,
                self.zeta_cutoff,
            )
        )
        return output

    def set_EC_via_ECS(self, ECS: float) -> None:
        """Helper function to set `EC` by providing `ECS`, keeping `ECJ` constant."""
        self._zeropi.set_EC_via_ECS(ECS)

    @property
    def E_zeta(self) -> float:
        """Returns energy quantum of the zeta mode"""
        return (8.0 * self.EL * self.EC) ** 0.5

    def set_E_zeta(self, value: float) -> None:
        raise ValueError(
            "Cannot directly set `E_zeta`. Instead one can set its value through `EL`"
            " or `EC`."
        )

    @property
    def esys_method(self) -> Union[Callable, str, None]:
        return self._zeropi.esys_method

    @esys_method.setter
    def esys_method(self, value: Union[Callable, str, None] = None) -> None:
        self._zeropi.esys_method = value

    @property
    def esys_method_options(self) -> Union[dict, None]:
        return self._zeropi.esys_method_options

    @esys_method_options.setter
    def esys_method_options(self, value: Union[dict, None] = None) -> None:
        self._zeropi.esys_merthod_options = value

    @property
    def evals_method(self) -> Union[Callable, str, None]:
        return self._zeropi.evals_method

    @evals_method.setter
    def evals_method(self, value: Union[Callable, str, None] = None) -> None:
        self._zeropi.evals_method = value

    @property
    def evals_method_options(self) -> Union[dict, None]:
        return self._zeropi.evals_method_options

    @evals_method_options.setter
    def evals_method_options(self, value: Union[dict, None] = None) -> None:
        self._zeropi.evals_merthod_options = value

    def hamiltonian(
        self,
        return_parts: bool = False,
        energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False,
    ) -> Union[csc_matrix, Tuple[csc_matrix, ndarray, ndarray, ndarray]]:
        r"""Returns Hamiltonian in basis obtained by discretizing :math:`\phi`, employing
        charge basis for :math:`\theta`, and Fock basis for :math:`\zeta`, or in the eigenenergy basis.

        Parameters
        ----------
        return_parts:
            If set to true, `hamiltonian` returns
            [hamiltonian, evals, evecs, g_coupling_matrix]
        energy_esys:
            If `False` (default), returns Hamiltonian in native Hamiltonian basis
            If `True`, the energy eigenspectrum is computed, returns Hamiltonian in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns Hamiltonian in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Hamiltonian in chosen basis as csc_matrix. If the eigenenergy basis is chosen,
            unless `energy_esys` is specified, the Hamiltonian has dimensions of `truncated_dim`
            x `truncated_dim`. Otherwise, if eigenenergy basis is chosen, Hamiltonian has dimensions of m x m,
            for m given eigenvectors.

        """
        zeropi_dim = self.zeropi_cutoff
        zeropi_evals, zeropi_evecs = self._zeropi.eigensys(evals_count=zeropi_dim)
        zeropi_diag_hamiltonian = sparse.dia_matrix(
            (zeropi_dim, zeropi_dim), dtype=np.complex_
        )
        zeropi_diag_hamiltonian.setdiag(zeropi_evals)

        zeta_dim = self.zeta_cutoff
        prefactor = self.E_zeta

        zeta_diag_hamiltonian = op.number_sparse(zeta_dim, prefactor)

        hamiltonian_mat = sparse.kron(
            zeropi_diag_hamiltonian,
            sparse.identity(zeta_dim, format="dia", dtype=np.complex_),
        )
        hamiltonian_mat += sparse.kron(
            sparse.identity(zeropi_dim, format="dia", dtype=np.complex_),
            zeta_diag_hamiltonian,
        )

        gmat = self.g_coupling_matrix(zeropi_evecs)
        zeropi_coupling = sparse.dia_matrix((zeropi_dim, zeropi_dim), dtype=np.complex_)
        for l1 in range(zeropi_dim):
            for l2 in range(zeropi_dim):
                zeropi_coupling += gmat[l1, l2] * op.hubbard_sparse(l1, l2, zeropi_dim)
        hamiltonian_mat += sparse.kron(
            zeropi_coupling, op.annihilation_sparse(zeta_dim)
        ) + sparse.kron(zeropi_coupling.conjugate().T, op.creation_sparse(zeta_dim))
        hmtocsc = hamiltonian_mat.tocsc()
        if return_parts:
            return (
                self.process_hamiltonian(
                    native_hamiltonian=hmtocsc, energy_esys=energy_esys
                ),
                zeropi_evals,
                zeropi_evecs,
                gmat,
            )
        return self.process_hamiltonian(
            native_hamiltonian=hmtocsc, energy_esys=energy_esys
        )

    def d_hamiltonian_d_flux(
        self,
        zeropi_evecs: ndarray = None,
        energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False,
    ) -> Union[ndarray, csc_matrix]:
        r"""
        Calculates a derivative of the Hamiltonian w.r.t flux, at the current value of flux,
        as stored in the object. The returned operator is in the product basis or eigenenergy basis.

        Helper method _zeropi_operator_in_product_basis is employed which converts
        a zeropi operator into one in the product basis. If user already has zeropi eigenvectors, user can input
        as zeropi_evecs=ndarray.

        Parameters
        ----------
        zeropi_evecs:
            If None (default), helper method _zeropi_operator_in_product_basis calculates zeropi eigenvectors and uses
            them to convert operator to the product basis, if product basis is chosen.
            If zeropi_evecs = zeropievecs, where zeropievecs is an ndarray, and product basis is chosen,
            helper method _zeropi_operator_in_product_basis uses zeropievecs to convert the operator
            to the product basis.
        energy_esys:
            If `False` (default), returns operator in the product basis.
            If `True`, the energy eigenspectrum is computed, returns operator in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator in chosen basis. If product basis is chosen, operator
            returned as a csc_matrix. If the eigenenergy basis is chosen,
            unless `energy_esys` is specified, operator has dimensions of `truncated_dim`
            x truncated_dim, and is returned as an ndarray. Otherwise, if eigenenergy basis is chosen,
            operator has dimensions of m x m, for m given eigenvectors, and is returned as an ndarray.
        """
        native = self._zeropi_operator_in_product_basis(
            self._zeropi.d_hamiltonian_d_flux(), zeropi_evecs=zeropi_evecs
        )
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def d_hamiltonian_d_EJ(
        self,
        zeropi_evecs: ndarray = None,
        energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False,
    ) -> Union[ndarray, csc_matrix]:
        r"""
        Calculates a derivative of the Hamiltonian w.r.t EJ. The returned operator is in the
        product basis or eigenenergy basis.

        Helper method _zeropi_operator_in_product_basis is employed which converts
        a zeropi operator into one in the product basis. If user already has zeropi eigenvectors, user can input
        as zeropi_evecs=ndarray.

        Parameters
        ----------
        zeropi_evecs:
            If None (default), helper method _zeropi_operator_in_product_basis calculates zeropi eigenvectors and uses
            them to convert operator to the product basis, if product basis is chosen.
            If zeropi_evecs = zeropievecs, where zeropievecs is an ndarray, and product basis is chosen,
            helper method _zeropi_operator_in_product_basis uses zeropievecs to convert the operator
            to the product basis.
        energy_esys:
            If `False` (default), returns operator in the product basis.
            If `True`, the energy eigenspectrum is computed, returns operator in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator in chosen basis. If product basis chosen, operator
            returned as a csc_matrix. If the eigenenergy basis is chosen,
            unless `energy_esys` is specified, operator has dimensions of `truncated_dim`
            x truncated_dim, and is returned as an ndarray. Otherwise, if eigenenergy basis is chosen,
            operator has dimensions of m x m, for m given eigenvectors, and is returned as an ndarray.
        """
        native = self._zeropi_operator_in_product_basis(
            self._zeropi.d_hamiltonian_d_EJ(), zeropi_evecs=zeropi_evecs
        )
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def d_hamiltonian_d_ng(
        self, energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False
    ) -> Union[ndarray, csc_matrix]:
        r"""
        Calculates a derivative of the Hamiltonian w.r.t ng.
        Returns matrix representing a derivative of the Hamiltonian in the native Hamiltonian basis
        or eigenenergy basis.

        Parameters
        ----------
        energy_esys:
            If `False` (default), returns operator in the native basis.
            If `True`, the energy eigenspectrum is computed, returns operator in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator in chosen basis. If native basis chosen, operator
            returned as a csc_matrix. If the eigenenergy basis is chosen,
            unless `energy_esys` is specified, operator has dimensions of `truncated_dim`
            x truncated_dim, and is returned as an ndarray. Otherwise, if eigenenergy basis is chosen,
            operator has dimensions of m x m, for m given eigenvectors, and is returned as an ndarray.
        """
        native = -8 * self.EC * self.n_theta_operator()
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def _zeropi_operator_in_product_basis(
        self, zeropi_operator, zeropi_evecs: ndarray = None
    ) -> csc_matrix:
        """Helper method that converts a zeropi operator into one in the product basis.

        Returns
        -------
            operator written in the product basis
        """
        zeropi_dim = self.zeropi_cutoff
        zeta_dim = self.zeta_cutoff

        if zeropi_evecs is None:
            _, zeropi_evecs = self._zeropi.eigensys(evals_count=zeropi_dim)

        op_eigen_basis = sparse.dia_matrix(
            (zeropi_dim, zeropi_dim), dtype=np.complex_
        )  # guaranteed to be zero?

        op_zeropi = spec_utils.get_matrixelement_table(zeropi_operator, zeropi_evecs)
        for n in range(zeropi_dim):
            for m in range(zeropi_dim):
                op_eigen_basis += op_zeropi[n, m] * op.hubbard_sparse(n, m, zeropi_dim)

        return sparse.kron(
            op_eigen_basis,
            sparse.identity(zeta_dim, format="csc", dtype=np.complex_),
            format="csc",
        )

    def i_d_dphi_operator(
        self,
        zeropi_evecs: ndarray = None,
        energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False,
    ) -> Union[ndarray, csc_matrix]:
        r"""
        Returns :math:`i d/d\phi` operator in the product or eigenenergy basis.

        Helper method _zeropi_operator_in_product_basis is employed which converts
        a zeropi operator into one in the product basis. If user already has zeropi eigenvectors, user can input
        as zeropi_evecs=ndarray.

        Parameters
        ----------
        zeropi_evecs:
            If None (default), helper method _zeropi_operator_in_product_basis calculates zeropi eigenvectors and uses
            them to convert operator to the product basis, if product basis is chosen.
            If zeropi_evecs = zeropievecs, where zeropievecs is an ndarray, and product basis is chosen,
            helper method _zeropi_operator_in_product_basis uses zeropievecs to convert the operator
            to the product basis.
        energy_esys:
            If `False` (default), returns operator in the product basis.
            If `True`, the energy eigenspectrum is computed, returns operator in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator in chosen basis. If product basis chosen, operator
            returned as a csc_matrix. If the eigenenergy basis is chosen,
            unless `energy_esys` is specified, operator has dimensions of `truncated_dim`
            x truncated_dim, and is returned as an ndarray. Otherwise, if eigenenergy basis is chosen,
            operator has dimensions of m x m, for m given eigenvectors, and is returned as an ndarray.
        """
        native = self._zeropi_operator_in_product_basis(
            self._zeropi.i_d_dphi_operator(), zeropi_evecs=zeropi_evecs
        )
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def n_theta_operator(
        self,
        zeropi_evecs: ndarray = None,
        energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False,
    ) -> Union[ndarray, csc_matrix]:
        r"""
        Returns :math:`n_\theta`  operator in the product or eigenenergy basis.

        Helper method _zeropi_operator_in_product_basis is employed which converts
        a zeropi operator into one in the product basis. If user already has zeropi eigenvectors, user can input
        as zeropi_evecs=ndarray.

        Parameters
        ----------
        zeropi_evecs:
            If None (default), helper method _zeropi_operator_in_product_basis calculates zeropi eigenvectors and uses
            them to convert operator to the product basis, if product basis is chosen.
            If zeropi_evecs = zeropievecs, where zeropievecs is an ndarray, and product basis is chosen,
            helper method _zeropi_operator_in_product_basis uses zeropievecs to convert the operator
            to the product basis.
        energy_esys:
            If `False` (default), returns operator in the product basis.
            If `True`, the energy eigenspectrum is computed, returns operator in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator in chosen basis. If product basis chosen, operator
            returned as a csc_matrix. If the eigenenergy basis is chosen,
            unless `energy_esys` is specified, operator has dimensions of `truncated_dim`
            x truncated_dim, and is returned as an ndarray. Otherwise, if eigenenergy basis is chosen,
            operator has dimensions of m x m, for m given eigenvectors, and is returned as an ndarray.
        """
        native = self._zeropi_operator_in_product_basis(
            self._zeropi.n_theta_operator(), zeropi_evecs=zeropi_evecs
        )
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def phi_operator(
        self,
        zeropi_evecs: ndarray = None,
        energy_esys: Union[bool, Tuple[ndarray, ndarray]] = False,
    ) -> Union[ndarray, csc_matrix]:
        r"""
        Returns :math:`\phi`  operator in the product or eigenenergy basis.

        Helper method _zeropi_operator_in_product_basis is employed which converts
        a zeropi operator into one in the product basis. If user already has zeropi eigenvectors, user can input
        as zeropi_evecs=ndarray.

        Parameters
        ----------
        zeropi_evecs:
            If None (default), helper method _zeropi_operator_in_product_basis calculates zeropi eigenvectors and uses
            them to convert operator to the product basis, if product basis is chosen.
            If zeropi_evecs = zeropievecs, where zeropievecs is an ndarray, and product basis is chosen,
            helper method _zeropi_operator_in_product_basis uses zeropievecs to convert the operator
            to the product basis.
        energy_esys:
            If `False` (default), returns operator in the product basis.
            If `True`, the energy eigenspectrum is computed, returns operator in the energy eigenbasis.
            If `energy_esys = esys`, where esys is a tuple containing two ndarrays (eigenvalues and energy eigenvectors),
            returns operator in the energy eigenbasis, and does not have to recalculate eigenspectrum.

        Returns
        -------
            Operator in chosen basis. If product basis chosen, operator
            returned as a csc_matrix. If the eigenenergy basis is chosen,
            unless `energy_esys` is specified, operator has dimensions of `truncated_dim`
            x truncated_dim, and is returned as an ndarray. Otherwise, if eigenenergy basis is chosen,
            operator has dimensions of m x m, for m given eigenvectors, and is returned as an ndarray.
        """
        native = self._zeropi_operator_in_product_basis(
            self._zeropi.phi_operator(), zeropi_evecs=zeropi_evecs
        )
        return self.process_op(native_op=native, energy_esys=energy_esys)

    def hilbertdim(self) -> int:
        """Returns Hilbert space dimension

        Returns
        -------
        int
        """
        return self.zeropi_cutoff * self.zeta_cutoff

    def _evals_calc(
        self, evals_count: int, hamiltonian_mat: csc_matrix = None
    ) -> ndarray:
        if hamiltonian_mat is None:
            hamiltonian_mat = self.hamiltonian()
        evals = spec_utils.eigsh_safe(
            hamiltonian_mat,
            k=evals_count,
            # sigma=0.0,
            which="SA",
            return_eigenvectors=False,
        )
        return np.sort(evals)

    def _esys_calc(
        self, evals_count: int, hamiltonian_mat: csc_matrix = None
    ) -> Tuple[ndarray, ndarray]:
        if hamiltonian_mat is None:
            hamiltonian_mat = self.hamiltonian()
        evals, evecs = spec_utils.eigsh_safe(
            hamiltonian_mat,
            k=evals_count,
            # sigma=0.0,
            which="SA",
            return_eigenvectors=True,
        )
        evals, evecs = spec_utils.order_eigensystem(evals, evecs)
        return evals, evecs

    def g_phi_coupling_matrix(self, zeropi_states: ndarray) -> ndarray:
        """Returns a matrix of coupling strengths g^\\phi_{ll'}
        [cmp. Dempster et al., Eq. (18)], using the states from the list
        `zeropi_states`. Most commonly, `zeropi_states` will contain eigenvectors of the
        `DisorderedZeroPi` type.
        """
        prefactor = self.EL * (self.dEL / 2.0) * (8.0 * self.EC / self.EL) ** 0.25
        return prefactor * spec_utils.get_matrixelement_table(
            self._zeropi.phi_operator(), zeropi_states
        )

    def g_theta_coupling_matrix(self, zeropi_states: ndarray) -> ndarray:
        """Returns a matrix of coupling strengths i*g^\\theta_{ll'}
        [cmp. Dempster et al., Eq. (17)], using the states from the list
        'zeropi_states'.
        """
        prefactor = 1j * self.ECS * (self.dC / 2.0) * (32.0 * self.EL / self.EC) ** 0.25
        return prefactor * spec_utils.get_matrixelement_table(
            self._zeropi.n_theta_operator(), zeropi_states
        )

    def g_coupling_matrix(
        self, zeropi_states: ndarray = None, evals_count: int = None
    ) -> ndarray:
        """Returns a matrix of coupling strengths g_{ll'} [cmp. Dempster et al., text
        above Eq. (17)], using the states from 'zeropi_states'. If
        `zeropi_states==None`, then a set of `self.zeropi` eigenstates is calculated.
        Only in that case is `which` used for the eigenstate number (and hence the
        coupling matrix size).
        """
        if evals_count is None:
            evals_count = self._zeropi.truncated_dim
        if zeropi_states is None:
            _, zeropi_states = self._zeropi.eigensys(evals_count=evals_count)
        return self.g_phi_coupling_matrix(zeropi_states) + self.g_theta_coupling_matrix(
            zeropi_states
        )
