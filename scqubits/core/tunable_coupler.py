from typing import Callable, Dict, Any

import numpy as np
from numpy import ndarray
from scipy.linalg import inv
from scipy.optimize import minimize
from sympy import Matrix, S, diff, hessian, simplify, solve, symbols

import scqubits.core.qubit_base as base
import scqubits.core.vtbbasemethods as vtb
import scqubits.io_utils.fileio_serializers as serializers
from scqubits.core.hashing import Hashing


class TunableCouplerVTB(
    vtb.VTBBaseMethods, base.QubitBaseClass, serializers.Serializable
):
    _check_if_new_minima: Callable
    _normalize_minimum_inside_pi_range: Callable

    def __init__(
        self,
        ECq: float,
        ECS: float,
        EC1: float,
        EC2: float,
        ECg1: float,
        ECg2: float,
        EJq: float,
        EJ1: float,
        EJ2: float,
        epsilon: float,
        nglist: ndarray,
        flux: float,
        num_exc: int,
        maximum_unit_cell_vector_length: int,
        truncated_dim: int = None,
        **kwargs
    ) -> None:
        vtb.VTBBaseMethods.__init__(
            self,
            num_exc,
            maximum_unit_cell_vector_length,
            number_degrees_freedom=6,
            number_periodic_degrees_freedom=6,
            number_junctions=7,
            **kwargs
        )
        self.ECq = ECq
        self.ECS = ECS
        self.EC1 = EC1
        self.EC2 = EC2
        self.ECg1 = ECg1
        self.ECg2 = ECg2
        self.epsilon = epsilon
        self.flux = flux
        self.truncated_dim = truncated_dim
        self._EJlist = np.array([EJ1, EJ2, EJq, EJq, EJq, EJq, epsilon*EJq])
        self._nglist = nglist
        self._stitching_coefficients = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    def EC_matrix(self):
        return 0.5 * inv(self.capacitance_matrix())

    def set_EJlist(self, EJlist) -> None:
        self.__dict__["EJlist"] = EJlist

    def get_EJlist(self) -> ndarray:
        return self._EJlist

    EJlist = property(get_EJlist, set_EJlist)

    def set_nglist(self, nglist) -> None:
        self.__dict__["nglist"] = nglist

    def get_nglist(self) -> ndarray:
        return self._nglist

    nglist = property(get_nglist, set_nglist)

    @staticmethod
    def default_params() -> Dict[str, Any]:
        return {
            "ECq": 0.4,
            "ECS": 10.0,
            "EC1": 0.205,
            "EC2": 0.205,
            "ECg1": 0.185,
            "ECg2": 0.175,
            "EJq": 90.0,
            "EJ1": 34.4,
            "EJ2": 19.65,
            "epsilon": 1./6.,
            "nglist": np.zeros(6),
            "flux": 0.0,
            "num_exc": 2,
            "maximum_unit_cell_vector_length": 8,
            "truncated_dim": 6,
        }

    def find_minima(self) -> ndarray:
        """Find all minima in the potential energy landscape of the current mirror.

        Returns
        -------
        ndarray
            Location of all minima, unsorted
        """
        minima_holder = []
        for m in range(4):
            guess_pos = np.array(
                [
                    2.0 * np.pi * (m - self.flux) / 5
                    for _ in range(4)
                ]
            )
            guess_neg = np.array(
                [
                    2.0 * np.pi * (-m - self.flux) / 5
                    for _ in range(4)
                ]
            )
            guess_pos = np.concatenate((np.zeros(2), guess_pos))
            guess_neg = np.concatenate((np.zeros(2), guess_neg))
            result_pos = minimize(
                self.vtb_potential, guess_pos, options={"disp": False}
            )
            result_neg = minimize(
                self.vtb_potential, guess_neg, options={"disp": False}
            )
            new_minimum_pos = self._check_if_new_minima(
                result_pos.x, minima_holder
            ) and self._check_if_second_derivative_potential_positive(result_pos.x)
            if new_minimum_pos and result_pos.success:
                minima_holder.append(
                    self._normalize_minimum_inside_pi_range(result_pos.x)
                )
            new_minimum_neg = self._check_if_new_minima(
                result_neg.x, minima_holder
            ) and self._check_if_second_derivative_potential_positive(result_neg.x)
            if new_minimum_neg and result_neg.success:
                minima_holder.append(
                    self._normalize_minimum_inside_pi_range(result_neg.x)
                )
        return np.array(minima_holder)

    def _check_if_second_derivative_potential_positive(
        self, phi_array: ndarray
    ) -> bool:
        """Helper method for determining whether the location specified by `phi_array` is a minimum."""
        second_derivative = np.round(
            -(self.vtb_potential(phi_array) - np.sum(self.EJlist)), decimals=3
        )
        return second_derivative > 0.0

    def vtb_potential(self, phi_array: ndarray) -> float:
        """Helper method for converting potential method arguments"""
        potential = np.sum(-self.EJlist[0:6] * np.cos(phi_array[0:6]))
        potential += -self.EJlist[-1] * np.cos(np.sum(self.stitching_coefficients * phi_array)
                                               + 2.0 * np.pi * self.flux)
        return potential

    def capacitance_matrix(self):
        U = Matrix(self._coordinate_transformation_matrix())
        U_inv = U ** -1
        phi1, phi2, phi3, phi4, phi5, phi6, phi7 = symbols("phi1 phi2 phi3 phi4 phi5 phi6 phi7")
        phi_vector = Matrix([phi1, phi2, phi3, phi4, phi5, phi6, phi7])
        Cq = 1 / S(2.0 * self.ECq)
        CS = 1 / S(2.0 * self.ECS)
        C1 = 1 / S(2.0 * self.EC1)
        C2 = 1 / S(2.0 * self.EC2)
        Cg1 = 1 / S(2.0 * self.ECg1)
        Cg2 = 1 / S(2.0 * self.ECg2)
        T = 0.5 * (
                C1 * phi1 ** 2
                + C2 * phi2 ** 2
                + CS * (phi1 - phi2) ** 2
                + Cg1 * (phi1 - phi3) ** 2
                + Cg2 * (phi7 - phi2) ** 2
                + Cq * (self.epsilon * (phi7 - phi3) ** 2
                        + (phi4 - phi3) ** 2
                        + (phi5 - phi4) ** 2
                        + (phi6 - phi5) ** 2
                        + (phi7 - phi6) ** 2
                        )
        )
        varphi1, varphi2, varphi3, varphi4, varphi5, varphi6, varphisum = symbols(
            "varphi1 varphi2 varphi3 varphi4 varphi5 varphi6 varphisum"
        )
        varphi_list = Matrix([varphi1, varphi2, varphi3, varphi4, varphi5, varphi6, varphisum])
        phi_subs = U_inv * varphi_list
        T = T.subs([(phival, phi_subs[j]) for j, phival in enumerate(phi_vector)])
        T = simplify(T.subs(varphisum, solve(diff(T, varphisum), varphisum)[0]))
        cap_mat = hessian(T, varphi_list)
        return np.array(cap_mat, dtype=np.float_)[:-1, :-1]

    @staticmethod
    def _coordinate_transformation_matrix() -> ndarray:
        """Builds the matrix necessary for the coordinate transformation"""
        coord_transform_matrix = np.array([[1.0, 0.0,  0.0,  0.0,  0.0,  0.0, 0.0],
                                           [0.0, 1.0,  0.0,  0.0,  0.0,  0.0, 0.0],
                                           [0.0, 0.0, -1.0,  1.0,  0.0,  0.0, 0.0],
                                           [0.0, 0.0,  0.0, -1.0,  1.0,  0.0, 0.0],
                                           [0.0, 0.0,  0.0,  0.0, -1.0,  1.0, 0.0],
                                           [0.0, 0.0,  0.0,  0.0,  0.0, -1.0, 1.0],
                                           [0.0, 0.0,  1.0,  1.0,  1.0,  1.0, 1.0]])
        return coord_transform_matrix


class TunableCouplerVTBGlobal(Hashing, TunableCouplerVTB):
    def __init__(
        self,
        ECq: float,
        ECS: float,
        EC1: float,
        EC2: float,
        ECg1: float,
        ECg2: float,
        EJq: float,
        EJ1: float,
        EJ2: float,
        epsilon: float,
        nglist: ndarray,
        flux: float,
        num_exc: int,
        maximum_unit_cell_vector_length: int,
        truncated_dim: int = None,
        **kwargs
    ):
        Hashing.__init__(self)
        TunableCouplerVTB.__init__(
            self,
            ECq,
            ECS,
            EC1,
            EC2,
            ECg1,
            ECg2,
            EJq,
            EJ1,
            EJ2,
            epsilon,
            nglist,
            flux,
            num_exc,
            maximum_unit_cell_vector_length,
            truncated_dim,
            **kwargs
        )
