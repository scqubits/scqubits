import os

import numpy as np
import pytest
from scipy.linalg import inv
from scipy.special import factorial
from scipy.special import comb
import math

import scqubits.settings
from scqubits.core.storage import SpectrumData
from scqubits.core.flux_qubit import FluxQubit
from scqubits.core.flux_qubit_vtb import FluxQubitVTB, FluxQubitVTBSqueezing
from scqubits.tests.conftest import VTBTestFunctions

TESTDIR, _ = os.path.split(scqubits.__file__)
TESTDIR = os.path.join(TESTDIR, "tests", "")
DATADIR = os.path.join(TESTDIR, "data", "")


class TestFluxQubitVTB(VTBTestFunctions):
    @classmethod
    def setup_class(cls):
        cls.qbt = None
        cls.qbt_type = FluxQubitVTB
        cls.file_str = "fluxqubitvtb"
        cls.op1_str = "n_operator"
        cls.op1_arg = {"dof_index": 0}
        cls.op2_str = "exp_i_phi_operator"
        cls.op2_arg = {"dof_index": 0}
        cls.param_name = "flux"
        cls.param_list = np.linspace(0.46, 0.54, 21)
        cls.compare_qbt_type = FluxQubit
        cls.compare_file_str = "fluxqubit"


def heat(x, y, n):
    heatsum = 0.0
    for k in range(math.floor(float(n) / 2.0) + 1):
        heatsum += x ** (n - 2 * k) * y ** (k) / (factorial(n - 2 * k) * factorial(k))
    return heatsum * factorial(n)


def Hmn(m, n, x, y, w, z, tau):
    Hmnsum = 0.0
    for i in range(min(m, n) + 1):
        Hmnsum += (
            factorial(m)
            * factorial(n)
            / (factorial(m - i) * factorial(n - i) * factorial(i))
            * heat(x, y, m - i)
            * heat(w, z, n - i)
            * tau ** i
        )
    return Hmnsum


def Imn(m, n, y, z, a, b, c, d, f, alpha):
    xbar = b + (a * alpha / (2.0 * f))
    ybar = y + (a ** 2) / (4.0 * f)
    wbar = d + (c * alpha) / (2.0 * f)
    zbar = z + (c ** 2) / (4.0 * f)
    tau = a * c / (2.0 * f)
    return (
        np.sqrt(np.pi / f)
        * np.exp(alpha ** 2 / (4.0 * f))
        * Hmn(m, n, xbar, ybar, wbar, zbar, tau)
        / (
            np.sqrt(np.sqrt(np.pi) * 2 ** n * factorial(n))
            * np.sqrt(np.sqrt(np.pi) * 2 ** m * factorial(m))
        )
    )


def Rmnk(m, n, y, z, a, b, c, d, f, alpha, k):
    xbar = b + (a * alpha / (2.0 * f))
    ybar = y + (a ** 2) / (4.0 * f)
    wbar = d + (c * alpha) / (2.0 * f)
    zbar = z + (c ** 2) / (4.0 * f)
    tau = a * c / (2.0 * f)
    Rmnksum = 0.0
    for l in range(k + 1):
        if m - l >= 0 and n - k + l >= 0:
            Rmnksum += (
                comb(k, l)
                * ((a / c) ** l)
                * factorial(m)
                * factorial(n)
                / (factorial(m - l) * factorial(n - k + l))
            ) * Hmn(m - l, n - k + l, xbar, ybar, wbar, zbar, tau)
    return Rmnksum * (c / (2.0 * f)) ** k


def pImn(p, m, n, y, z, a, b, c, d, f, alpha):
    pImnsum = 0.0
    if m < 0 or n < 0:
        return 0.0
    for k in range(p + 1):
        pImnsum += (
            comb(p, k)
            * heat(alpha / (2.0 * f), 1.0 / (4.0 * f), p - k)
            * Rmnk(m, n, y, z, a, b, c, d, f, alpha, k)
        )
    return (
        np.sqrt(np.pi / f)
        * np.exp(alpha ** 2 / (4.0 * f))
        * pImnsum
        / (
            np.sqrt(np.sqrt(np.pi) * (2 ** n) * factorial(n))
            * np.sqrt(np.sqrt(np.pi) * (2 ** m) * factorial(m))
        )
    )


class TestFluxQubitVTBSqueezing(VTBTestFunctions):
    @classmethod
    def setup_class(cls):
        cls.qbt = None
        cls.qbt_type = FluxQubitVTBSqueezing
        cls.file_str = "fluxqubitvtbsqueezing"
        #        cls.op1_str = 'n_operator'
        #        cls.op1_arg = {'j': 0}
        #        cls.op2_str = 'exp_i_phi_operator'
        #        cls.op2_arg = {'j': 0}
        cls.param_name = "flux"
        cls.param_list = np.linspace(0.46, 0.54, 21)
        cls.compare_qbt_type = FluxQubit
        cls.compare_file_str = "fluxqubit"

    def test_harmonic_length_optimization_gradient(self, io_type):
        pytest.skip("not implemented for squeezing")

    def test_print_matrixelements(self, io_type):
        pytest.skip("not implemented for squeezing")

    def test_plot_matrixelements(self, io_type):
        pytest.skip("not implemented for squeezing")

    def test_matrixelement_table(self, io_type):
        pytest.skip("not implemented for squeezing")

    def test_inner_product_matrix_against_babusci(self, io_type):
        testname = self.file_str + "_1." + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        inner_product_ladder_mat = self.qbt.inner_product_matrix()
        num_states_per_min = self.qbt.number_states_per_minimum()
        relevant_unit_cell_vectors = self.qbt.find_relevant_unit_cell_vectors()
        hilbertdim = self.qbt.hilbertdim()
        inner_product_babusci_mat = np.zeros((hilbertdim, hilbertdim))
        minima_dict = self.qbt.sorted_minima_dict
        for min_index, min_location in minima_dict.items():
            Xi = self.qbt.Xi_matrix(min_index)
            Xi_inv = inv(Xi)
            for sone in range(self.qbt.num_exc + 1):
                for stwo in range(self.qbt.num_exc + 1):
                    for soneprime in range(self.qbt.num_exc + 1):
                        for stwoprime in range(self.qbt.num_exc + 1):
                            matelem = 0.0
                            for jkvals in relevant_unit_cell_vectors[
                                (min_index, min_index)
                            ]:
                                phik = 2.0 * np.pi * np.array([jkvals[0], jkvals[1]])
                                zetaoneoffset = (
                                    Xi_inv[0, 0] * min_location[0]
                                    + Xi_inv[0, 1] * min_location[1]
                                )
                                zetatwooffset = (
                                    Xi_inv[1, 0] * min_location[0]
                                    + Xi_inv[1, 1] * min_location[1]
                                )
                                zetaoneprimeoffset = Xi_inv[0, 0] * (
                                    phik[0] + min_location[0]
                                ) + Xi_inv[0, 1] * (phik[1] + min_location[1])
                                zetatwoprimeoffset = Xi_inv[1, 0] * (
                                    phik[0] + min_location[0]
                                ) + Xi_inv[1, 1] * (phik[1] + min_location[1])
                                matelem += (
                                    np.exp(
                                        -0.5
                                        * (zetatwooffset ** 2 + zetatwoprimeoffset ** 2)
                                    )
                                    * pImn(
                                        p=0,
                                        m=stwo,
                                        n=stwoprime,
                                        y=-1,
                                        z=-1,
                                        a=2,
                                        b=-2 * zetatwooffset,
                                        c=2,
                                        d=-2 * zetatwoprimeoffset,
                                        f=1,
                                        alpha=zetatwooffset + zetatwoprimeoffset,
                                    )
                                    * pImn(
                                        p=0,
                                        m=sone,
                                        n=soneprime,
                                        y=-1,
                                        z=-1,
                                        a=2,
                                        b=-2 * zetaoneoffset,
                                        c=2,
                                        d=-2 * zetaoneprimeoffset,
                                        f=1,
                                        alpha=zetaoneoffset + zetaoneprimeoffset,
                                    )
                                    * np.exp(
                                        -0.5
                                        * (zetaoneoffset ** 2 + zetaoneprimeoffset ** 2)
                                    )
                                )
                            i = (
                                (self.qbt.num_exc + 1) * (sone)
                                + stwo
                                + min_index * (self.qbt.num_exc + 1) ** 2
                            )
                            j = (
                                (self.qbt.num_exc + 1) * (soneprime)
                                + stwoprime
                                + min_index * (self.qbt.num_exc + 1) ** 2
                            )
                            inner_product_babusci_mat[i, j] += matelem
        block1_babusci = inner_product_babusci_mat[
            0:num_states_per_min, 0:num_states_per_min
        ]
        block2_babusci = inner_product_babusci_mat[
            num_states_per_min : 2 * num_states_per_min,
            num_states_per_min : 2 * num_states_per_min,
        ]
        block1_ladder = inner_product_ladder_mat[
            0:num_states_per_min, 0:num_states_per_min
        ]
        block2_ladder = inner_product_ladder_mat[
            num_states_per_min : 2 * num_states_per_min,
            num_states_per_min : 2 * num_states_per_min,
        ]
        assert np.allclose(block1_babusci, block1_ladder, atol=1e-7)
        assert np.allclose(block2_babusci, block2_ladder, atol=1e-7)

    def test_kinetic_matrix_against_babusci(self, io_type):
        testname = self.file_str + "_1." + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        self.qbt.flux = 0.46
        kinetic_ladder_mat = self.qbt.kinetic_matrix()
        num_states_per_min = self.qbt.number_states_per_minimum()
        relevant_unit_cell_vectors = self.qbt.find_relevant_unit_cell_vectors()
        hilbertdim = self.qbt.hilbertdim()
        kinetic_babusci_mat = np.zeros((hilbertdim, hilbertdim))
        minima_dict = self.qbt.sorted_minima_dict
        EC_mat = self.qbt.EC_matrix()
        for min_index, min_location in minima_dict.items():
            Xi = self.qbt.Xi_matrix(min_index)
            Xi_inv = inv(Xi)
            EC_mat_t = Xi_inv @ EC_mat @ Xi_inv.T
            for sone in range(self.qbt.num_exc + 1):
                for stwo in range(self.qbt.num_exc + 1):
                    for soneprime in range(self.qbt.num_exc + 1):
                        for stwoprime in range(self.qbt.num_exc + 1):
                            matelem = 0.0
                            for jkvals in relevant_unit_cell_vectors[
                                (min_index, min_index)
                            ]:
                                phik = 2.0 * np.pi * np.array([jkvals[0], jkvals[1]])
                                zetaoneoffset = (
                                    Xi_inv[0, 0] * min_location[0]
                                    + Xi_inv[0, 1] * min_location[1]
                                )
                                zetatwooffset = (
                                    Xi_inv[1, 0] * min_location[0]
                                    + Xi_inv[1, 1] * min_location[1]
                                )
                                zetaoneprimeoffset = Xi_inv[0, 0] * (
                                    phik[0] + min_location[0]
                                ) + Xi_inv[0, 1] * (phik[1] + min_location[1])
                                zetatwoprimeoffset = Xi_inv[1, 0] * (
                                    phik[0] + min_location[0]
                                ) + Xi_inv[1, 1] * (phik[1] + min_location[1])

                                elem11 = (
                                    4.0
                                    * EC_mat_t[0, 0]
                                    * np.exp(
                                        -0.5
                                        * (zetatwooffset ** 2 + zetatwoprimeoffset ** 2)
                                    )
                                    * pImn(
                                        p=0,
                                        m=stwo,
                                        n=stwoprime,
                                        y=-1,
                                        z=-1,
                                        a=2,
                                        b=-2 * zetatwooffset,
                                        c=2,
                                        d=-2 * zetatwoprimeoffset,
                                        f=1,
                                        alpha=zetatwooffset + zetatwoprimeoffset,
                                    )
                                    * pImn(
                                        p=0,
                                        m=sone,
                                        n=soneprime,
                                        y=-1,
                                        z=-1,
                                        a=2,
                                        b=-2 * zetaoneoffset,
                                        c=2,
                                        d=-2 * zetaoneprimeoffset,
                                        f=1,
                                        alpha=zetaoneoffset + zetaoneprimeoffset,
                                    )
                                    * np.exp(
                                        -0.5
                                        * (zetaoneoffset ** 2 + zetaoneprimeoffset ** 2)
                                    )
                                )

                                elem12 = -(
                                    4.0
                                    * EC_mat_t[0, 0]
                                    * np.exp(
                                        -0.5
                                        * (zetatwooffset ** 2 + zetatwoprimeoffset ** 2)
                                    )
                                    * pImn(
                                        p=0,
                                        m=stwo,
                                        n=stwoprime,
                                        y=-1,
                                        z=-1,
                                        a=2,
                                        b=-2 * zetatwooffset,
                                        c=2,
                                        d=-2 * zetatwoprimeoffset,
                                        f=1,
                                        alpha=zetatwooffset + zetatwoprimeoffset,
                                    )
                                    * pImn(
                                        p=2,
                                        m=sone,
                                        n=soneprime,
                                        y=-1,
                                        z=-1,
                                        a=2,
                                        b=-2 * (zetaoneoffset - zetaoneprimeoffset),
                                        c=2,
                                        d=0,
                                        f=1,
                                        alpha=zetaoneoffset - zetaoneprimeoffset,
                                    )
                                    * np.exp(
                                        -0.5 * (zetaoneprimeoffset - zetaoneoffset) ** 2
                                    )
                                )

                                elem13 = elem14 = 0.0
                                if soneprime >= 1:
                                    elem13 += -(
                                        (
                                            4.0
                                            * EC_mat_t[0, 0]
                                            / (np.sqrt(soneprime * 2))
                                        )
                                        * np.exp(
                                            -0.5
                                            * (
                                                zetatwooffset ** 2
                                                + zetatwoprimeoffset ** 2
                                            )
                                        )
                                        * pImn(
                                            p=0,
                                            m=stwo,
                                            n=stwoprime,
                                            y=-1,
                                            z=-1,
                                            a=2,
                                            b=-2 * zetatwooffset,
                                            c=2,
                                            d=-2 * zetatwoprimeoffset,
                                            f=1,
                                            alpha=zetatwooffset + zetatwoprimeoffset,
                                        )
                                        * 4
                                        * zetaoneprimeoffset
                                        * soneprime
                                        * pImn(
                                            p=0,
                                            m=sone,
                                            n=soneprime - 1,
                                            y=-1,
                                            z=-1,
                                            a=2,
                                            b=-2 * zetaoneoffset,
                                            c=2,
                                            d=-2 * zetaoneprimeoffset,
                                            f=1,
                                            alpha=zetaoneoffset + zetaoneprimeoffset,
                                        )
                                        * np.exp(
                                            -0.5
                                            * (
                                                zetaoneoffset ** 2
                                                + zetaoneprimeoffset ** 2
                                            )
                                        )
                                    )

                                    elem14 += (
                                        (
                                            4.0
                                            * EC_mat_t[0, 0]
                                            / (np.sqrt(soneprime * 2))
                                        )
                                        * np.exp(
                                            -0.5
                                            * (
                                                zetatwooffset ** 2
                                                + zetatwoprimeoffset ** 2
                                            )
                                        )
                                        * pImn(
                                            p=0,
                                            m=stwo,
                                            n=stwoprime,
                                            y=-1,
                                            z=-1,
                                            a=2,
                                            b=-2 * zetatwooffset,
                                            c=2,
                                            d=-2 * zetatwoprimeoffset,
                                            f=1,
                                            alpha=zetatwooffset + zetatwoprimeoffset,
                                        )
                                        * 4
                                        * soneprime
                                        * pImn(
                                            p=1,
                                            m=sone,
                                            n=soneprime - 1,
                                            y=-1,
                                            z=-1,
                                            a=2,
                                            b=-2 * zetaoneoffset,
                                            c=2,
                                            d=-2 * zetaoneprimeoffset,
                                            f=1,
                                            alpha=zetaoneoffset + zetaoneprimeoffset,
                                        )
                                        * np.exp(
                                            -0.5
                                            * (
                                                zetaoneoffset ** 2
                                                + zetaoneprimeoffset ** 2
                                            )
                                        )
                                    )

                                elem15 = 0.0
                                if soneprime >= 2:
                                    elem15 += (
                                        -(
                                            4.0
                                            * EC_mat_t[0, 0]
                                            / (
                                                np.sqrt(
                                                    soneprime * (soneprime - 1) * 2 * 2
                                                )
                                            )
                                        )
                                        * np.exp(
                                            -0.5
                                            * (
                                                zetatwooffset ** 2
                                                + zetatwoprimeoffset ** 2
                                            )
                                        )
                                        * pImn(
                                            p=0,
                                            m=stwo,
                                            n=stwoprime,
                                            y=-1,
                                            z=-1,
                                            a=2,
                                            b=-2 * zetatwooffset,
                                            c=2,
                                            d=-2 * zetatwoprimeoffset,
                                            f=1,
                                            alpha=zetatwooffset + zetatwoprimeoffset,
                                        )
                                        * 4
                                        * soneprime
                                        * (soneprime - 1)
                                        * pImn(
                                            p=0,
                                            m=sone,
                                            n=soneprime - 2,
                                            y=-1,
                                            z=-1,
                                            a=2,
                                            b=-2 * zetaoneoffset,
                                            c=2,
                                            d=-2 * zetaoneprimeoffset,
                                            f=1,
                                            alpha=zetaoneoffset + zetaoneprimeoffset,
                                        )
                                        * np.exp(
                                            -0.5
                                            * (
                                                zetaoneoffset ** 2
                                                + zetaoneprimeoffset ** 2
                                            )
                                        )
                                    )

                                #########

                                elem21 = (
                                    4.0
                                    * EC_mat_t[1, 1]
                                    * np.exp(
                                        -0.5
                                        * (zetaoneoffset ** 2 + zetaoneprimeoffset ** 2)
                                    )
                                    * pImn(
                                        p=0,
                                        m=sone,
                                        n=soneprime,
                                        y=-1,
                                        z=-1,
                                        a=2,
                                        b=-2 * zetaoneoffset,
                                        c=2,
                                        d=-2 * zetaoneprimeoffset,
                                        f=1,
                                        alpha=zetaoneoffset + zetaoneprimeoffset,
                                    )
                                    * pImn(
                                        p=0,
                                        m=stwo,
                                        n=stwoprime,
                                        y=-1,
                                        z=-1,
                                        a=2,
                                        b=-2 * zetatwooffset,
                                        c=2,
                                        d=-2 * zetatwoprimeoffset,
                                        f=1,
                                        alpha=zetatwooffset + zetatwoprimeoffset,
                                    )
                                    * np.exp(
                                        -0.5
                                        * (zetatwooffset ** 2 + zetatwoprimeoffset ** 2)
                                    )
                                )

                                elem22 = -(
                                    4.0
                                    * EC_mat_t[1, 1]
                                    * np.exp(
                                        -0.5
                                        * (zetaoneoffset ** 2 + zetaoneprimeoffset ** 2)
                                    )
                                    * pImn(
                                        p=0,
                                        m=sone,
                                        n=soneprime,
                                        y=-1,
                                        z=-1,
                                        a=2,
                                        b=-2 * zetaoneoffset,
                                        c=2,
                                        d=-2 * zetaoneprimeoffset,
                                        f=1,
                                        alpha=zetaoneoffset + zetaoneprimeoffset,
                                    )
                                    * pImn(
                                        p=2,
                                        m=stwo,
                                        n=stwoprime,
                                        y=-1,
                                        z=-1,
                                        a=2,
                                        b=-2 * (zetatwooffset - zetatwoprimeoffset),
                                        c=2,
                                        d=0,
                                        f=1,
                                        alpha=zetatwooffset - zetatwoprimeoffset,
                                    )
                                    * np.exp(
                                        -0.5 * (zetatwoprimeoffset - zetatwooffset) ** 2
                                    )
                                )

                                elem23 = elem24 = 0.0
                                if stwoprime >= 1:
                                    elem23 += -(
                                        (
                                            4.0
                                            * EC_mat_t[1, 1]
                                            / (np.sqrt(stwoprime * 2))
                                        )
                                        * np.exp(
                                            -0.5
                                            * (
                                                zetaoneoffset ** 2
                                                + zetaoneprimeoffset ** 2
                                            )
                                        )
                                        * pImn(
                                            p=0,
                                            m=sone,
                                            n=soneprime,
                                            y=-1,
                                            z=-1,
                                            a=2,
                                            b=-2 * zetaoneoffset,
                                            c=2,
                                            d=-2 * zetaoneprimeoffset,
                                            f=1,
                                            alpha=zetaoneoffset + zetaoneprimeoffset,
                                        )
                                        * 4
                                        * zetatwoprimeoffset
                                        * stwoprime
                                        * pImn(
                                            p=0,
                                            m=stwo,
                                            n=stwoprime - 1,
                                            y=-1,
                                            z=-1,
                                            a=2,
                                            b=-2 * zetatwooffset,
                                            c=2,
                                            d=-2 * zetatwoprimeoffset,
                                            f=1,
                                            alpha=zetatwooffset + zetatwoprimeoffset,
                                        )
                                        * np.exp(
                                            -0.5
                                            * (
                                                zetatwooffset ** 2
                                                + zetatwoprimeoffset ** 2
                                            )
                                        )
                                    )

                                    elem24 += (
                                        (
                                            4.0
                                            * EC_mat_t[1, 1]
                                            / (np.sqrt(stwoprime * 2))
                                        )
                                        * np.exp(
                                            -0.5
                                            * (
                                                zetaoneoffset ** 2
                                                + zetaoneprimeoffset ** 2
                                            )
                                        )
                                        * pImn(
                                            p=0,
                                            m=sone,
                                            n=soneprime,
                                            y=-1,
                                            z=-1,
                                            a=2,
                                            b=-2 * zetaoneoffset,
                                            c=2,
                                            d=-2 * zetaoneprimeoffset,
                                            f=1,
                                            alpha=zetaoneoffset + zetaoneprimeoffset,
                                        )
                                        * 4
                                        * stwoprime
                                        * pImn(
                                            p=1,
                                            m=stwo,
                                            n=stwoprime - 1,
                                            y=-1,
                                            z=-1,
                                            a=2,
                                            b=-2 * zetatwooffset,
                                            c=2,
                                            d=-2 * zetatwoprimeoffset,
                                            f=1,
                                            alpha=zetatwooffset + zetatwoprimeoffset,
                                        )
                                        * np.exp(
                                            -0.5
                                            * (
                                                zetatwooffset ** 2
                                                + zetatwoprimeoffset ** 2
                                            )
                                        )
                                    )

                                elem25 = 0.0
                                if stwoprime >= 2:
                                    elem25 += (
                                        -(
                                            4.0
                                            * EC_mat_t[1, 1]
                                            / (
                                                np.sqrt(
                                                    stwoprime * (stwoprime - 1) * 2 * 2
                                                )
                                            )
                                        )
                                        * np.exp(
                                            -0.5
                                            * (
                                                zetaoneoffset ** 2
                                                + zetaoneprimeoffset ** 2
                                            )
                                        )
                                        * pImn(
                                            p=0,
                                            m=sone,
                                            n=soneprime,
                                            y=-1,
                                            z=-1,
                                            a=2,
                                            b=-2 * zetaoneoffset,
                                            c=2,
                                            d=-2 * zetaoneprimeoffset,
                                            f=1,
                                            alpha=zetaoneoffset + zetaoneprimeoffset,
                                        )
                                        * 4
                                        * stwoprime
                                        * (stwoprime - 1)
                                        * pImn(
                                            p=0,
                                            m=stwo,
                                            n=stwoprime - 2,
                                            y=-1,
                                            z=-1,
                                            a=2,
                                            b=-2 * zetatwooffset,
                                            c=2,
                                            d=-2 * zetatwoprimeoffset,
                                            f=1,
                                            alpha=zetatwooffset + zetatwoprimeoffset,
                                        )
                                        * np.exp(
                                            -0.5
                                            * (
                                                zetatwooffset ** 2
                                                + zetatwoprimeoffset ** 2
                                            )
                                        )
                                    )

                                matelem += (
                                    elem11
                                    + elem12
                                    + elem13
                                    + elem14
                                    + elem15
                                    + elem21
                                    + elem22
                                    + elem23
                                    + elem24
                                    + elem25
                                )
                            i = (
                                (self.qbt.num_exc + 1) * (sone)
                                + stwo
                                + min_index * (self.qbt.num_exc + 1) ** 2
                            )
                            j = (
                                (self.qbt.num_exc + 1) * (soneprime)
                                + stwoprime
                                + min_index * (self.qbt.num_exc + 1) ** 2
                            )
                            kinetic_babusci_mat[i, j] += matelem
        block1_babusci = kinetic_babusci_mat[0:num_states_per_min, 0:num_states_per_min]
        block2_babusci = kinetic_babusci_mat[
            num_states_per_min : 2 * num_states_per_min,
            num_states_per_min : 2 * num_states_per_min,
        ]
        block1_ladder = kinetic_ladder_mat[0:num_states_per_min, 0:num_states_per_min]
        block2_ladder = kinetic_ladder_mat[
            num_states_per_min : 2 * num_states_per_min,
            num_states_per_min : 2 * num_states_per_min,
        ]
        assert np.allclose(block1_babusci, block1_ladder, atol=1e-7)
        assert np.allclose(block2_babusci, block2_ladder, atol=1e-7)

    def test_potential_matrix_against_babusci(self, io_type):
        testname = self.file_str + "_1." + io_type
        specdata = SpectrumData.create_from_file(DATADIR + testname)
        self.qbt = self.qbt_type(**specdata.system_params)
        self.qbt.flux = 0.46
        EJlist = np.array([1.0, 1.0, 0.8])
        self.qbt.EJlist = EJlist
        potential_ladder_mat = self.qbt.potential_matrix()
        num_states_per_min = self.qbt.number_states_per_minimum()
        relevant_unit_cell_vectors = self.qbt.find_relevant_unit_cell_vectors()
        hilbertdim = self.qbt.hilbertdim()
        potential_babusci_mat = np.zeros((hilbertdim, hilbertdim))
        minima_dict = self.qbt.sorted_minima_dict
        for min_index, min_location in minima_dict.items():
            Xi = self.qbt.Xi_matrix(min_index)
            Xi_inv = inv(Xi)
            for sone in range(self.qbt.num_exc + 1):
                for stwo in range(self.qbt.num_exc + 1):
                    for soneprime in range(self.qbt.num_exc + 1):
                        for stwoprime in range(self.qbt.num_exc + 1):
                            matelem = 0.0
                            for jkvals in relevant_unit_cell_vectors[
                                (min_index, min_index)
                            ]:
                                phik = 2.0 * np.pi * np.array([jkvals[0], jkvals[1]])
                                zetaoneoffset = (
                                    Xi_inv[0, 0] * min_location[0]
                                    + Xi_inv[0, 1] * min_location[1]
                                )
                                zetatwooffset = (
                                    Xi_inv[1, 0] * min_location[0]
                                    + Xi_inv[1, 1] * min_location[1]
                                )
                                zetaoneprimeoffset = Xi_inv[0, 0] * (
                                    phik[0] + min_location[0]
                                ) + Xi_inv[0, 1] * (phik[1] + min_location[1])
                                zetatwoprimeoffset = Xi_inv[1, 0] * (
                                    phik[0] + min_location[0]
                                ) + Xi_inv[1, 1] * (phik[1] + min_location[1])

                                potential1pos = (
                                    -0.5
                                    * EJlist[0]
                                    * (
                                        np.exp(
                                            -0.5
                                            * (
                                                zetatwooffset ** 2
                                                + zetatwoprimeoffset ** 2
                                            )
                                        )
                                        * pImn(
                                            p=0,
                                            m=stwo,
                                            n=stwoprime,
                                            y=-1,
                                            z=-1,
                                            a=2,
                                            b=-2 * zetatwooffset,
                                            c=2,
                                            d=-2 * zetatwoprimeoffset,
                                            f=1,
                                            alpha=(
                                                zetatwooffset
                                                + zetatwoprimeoffset
                                                + 1j * Xi[0, 1]
                                            ),
                                        )
                                        * pImn(
                                            p=0,
                                            m=sone,
                                            n=soneprime,
                                            y=-1,
                                            z=-1,
                                            a=2,
                                            b=-2 * zetaoneoffset,
                                            c=2,
                                            d=-2 * zetaoneprimeoffset,
                                            f=1,
                                            alpha=(
                                                zetaoneoffset
                                                + zetaoneprimeoffset
                                                + 1j * Xi[0, 0]
                                            ),
                                        )
                                        * np.exp(
                                            -0.5
                                            * (
                                                zetaoneoffset ** 2
                                                + zetaoneprimeoffset ** 2
                                            )
                                        )
                                    )
                                )

                                potential1neg = (
                                    -0.5
                                    * EJlist[0]
                                    * (
                                        np.exp(
                                            -0.5
                                            * (
                                                zetatwooffset ** 2
                                                + zetatwoprimeoffset ** 2
                                            )
                                        )
                                        * pImn(
                                            p=0,
                                            m=stwo,
                                            n=stwoprime,
                                            y=-1,
                                            z=-1,
                                            a=2,
                                            b=-2 * zetatwooffset,
                                            c=2,
                                            d=-2 * zetatwoprimeoffset,
                                            f=1,
                                            alpha=(
                                                zetatwooffset
                                                + zetatwoprimeoffset
                                                - 1j * Xi[0, 1]
                                            ),
                                        )
                                        * pImn(
                                            p=0,
                                            m=sone,
                                            n=soneprime,
                                            y=-1,
                                            z=-1,
                                            a=2,
                                            b=-2 * zetaoneoffset,
                                            c=2,
                                            d=-2 * zetaoneprimeoffset,
                                            f=1,
                                            alpha=(
                                                zetaoneoffset
                                                + zetaoneprimeoffset
                                                - 1j * Xi[0, 0]
                                            ),
                                        )
                                        * np.exp(
                                            -0.5
                                            * (
                                                zetaoneoffset ** 2
                                                + zetaoneprimeoffset ** 2
                                            )
                                        )
                                    )
                                )

                                potential2pos = (
                                    -0.5
                                    * EJlist[1]
                                    * (
                                        np.exp(
                                            -0.5
                                            * (
                                                zetatwooffset ** 2
                                                + zetatwoprimeoffset ** 2
                                            )
                                        )
                                        * pImn(
                                            p=0,
                                            m=stwo,
                                            n=stwoprime,
                                            y=-1,
                                            z=-1,
                                            a=2,
                                            b=-2 * zetatwooffset,
                                            c=2,
                                            d=-2 * zetatwoprimeoffset,
                                            f=1,
                                            alpha=(
                                                zetatwooffset
                                                + zetatwoprimeoffset
                                                + 1j * Xi[1, 1]
                                            ),
                                        )
                                        * pImn(
                                            p=0,
                                            m=sone,
                                            n=soneprime,
                                            y=-1,
                                            z=-1,
                                            a=2,
                                            b=-2 * zetaoneoffset,
                                            c=2,
                                            d=-2 * zetaoneprimeoffset,
                                            f=1,
                                            alpha=(
                                                zetaoneoffset
                                                + zetaoneprimeoffset
                                                + 1j * Xi[1, 0]
                                            ),
                                        )
                                        * np.exp(
                                            -0.5
                                            * (
                                                zetaoneoffset ** 2
                                                + zetaoneprimeoffset ** 2
                                            )
                                        )
                                    )
                                )

                                potential2neg = (
                                    -0.5
                                    * EJlist[1]
                                    * (
                                        np.exp(
                                            -0.5
                                            * (
                                                zetatwooffset ** 2
                                                + zetatwoprimeoffset ** 2
                                            )
                                        )
                                        * pImn(
                                            p=0,
                                            m=stwo,
                                            n=stwoprime,
                                            y=-1,
                                            z=-1,
                                            a=2,
                                            b=-2 * zetatwooffset,
                                            c=2,
                                            d=-2 * zetatwoprimeoffset,
                                            f=1,
                                            alpha=(
                                                zetatwooffset
                                                + zetatwoprimeoffset
                                                - 1j * Xi[1, 1]
                                            ),
                                        )
                                        * pImn(
                                            p=0,
                                            m=sone,
                                            n=soneprime,
                                            y=-1,
                                            z=-1,
                                            a=2,
                                            b=-2 * zetaoneoffset,
                                            c=2,
                                            d=-2 * zetaoneprimeoffset,
                                            f=1,
                                            alpha=(
                                                zetaoneoffset
                                                + zetaoneprimeoffset
                                                - 1j * Xi[1, 0]
                                            ),
                                        )
                                        * np.exp(
                                            -0.5
                                            * (
                                                zetaoneoffset ** 2
                                                + zetaoneprimeoffset ** 2
                                            )
                                        )
                                    )
                                )

                                potential3pos = -(
                                    0.5
                                    * EJlist[2]
                                    * np.exp(-1j * 2.0 * np.pi * self.qbt.flux)
                                    * np.exp(
                                        -0.5
                                        * (zetatwooffset ** 2 + zetatwoprimeoffset ** 2)
                                    )
                                    * pImn(
                                        p=0,
                                        m=stwo,
                                        n=stwoprime,
                                        y=-1,
                                        z=-1,
                                        a=2,
                                        b=-2 * zetatwooffset,
                                        c=2,
                                        d=-2 * zetatwoprimeoffset,
                                        f=1,
                                        alpha=(
                                            zetatwooffset
                                            + zetatwoprimeoffset
                                            + 1j * (Xi[1, 1] - Xi[0, 1])
                                        ),
                                    )
                                    * pImn(
                                        p=0,
                                        m=sone,
                                        n=soneprime,
                                        y=-1,
                                        z=-1,
                                        a=2,
                                        b=-2 * zetaoneoffset,
                                        c=2,
                                        d=-2 * zetaoneprimeoffset,
                                        f=1,
                                        alpha=(
                                            zetaoneoffset
                                            + zetaoneprimeoffset
                                            + 1j * (Xi[1, 0] - Xi[0, 0])
                                        ),
                                    )
                                    * np.exp(
                                        -0.5
                                        * (zetaoneoffset ** 2 + zetaoneprimeoffset ** 2)
                                    )
                                )

                                potential3neg = -(
                                    0.5
                                    * EJlist[2]
                                    * np.exp(1j * 2.0 * np.pi * self.qbt.flux)
                                    * np.exp(
                                        -0.5
                                        * (zetatwooffset ** 2 + zetatwoprimeoffset ** 2)
                                    )
                                    * pImn(
                                        p=0,
                                        m=stwo,
                                        n=stwoprime,
                                        y=-1,
                                        z=-1,
                                        a=2,
                                        b=-2 * zetatwooffset,
                                        c=2,
                                        d=-2 * zetatwoprimeoffset,
                                        f=1,
                                        alpha=(
                                            zetatwooffset
                                            + zetatwoprimeoffset
                                            - 1j * (Xi[1, 1] - Xi[0, 1])
                                        ),
                                    )
                                    * pImn(
                                        p=0,
                                        m=sone,
                                        n=soneprime,
                                        y=-1,
                                        z=-1,
                                        a=2,
                                        b=-2 * zetaoneoffset,
                                        c=2,
                                        d=-2 * zetaoneprimeoffset,
                                        f=1,
                                        alpha=(
                                            zetaoneoffset
                                            + zetaoneprimeoffset
                                            - 1j * (Xi[1, 0] - Xi[0, 0])
                                        ),
                                    )
                                    * np.exp(
                                        -0.5
                                        * (zetaoneoffset ** 2 + zetaoneprimeoffset ** 2)
                                    )
                                )

                                potentialconst = (
                                    np.sum(EJlist)
                                    * np.exp(
                                        -0.5
                                        * (zetatwooffset ** 2 + zetatwoprimeoffset ** 2)
                                    )
                                    * pImn(
                                        p=0,
                                        m=stwo,
                                        n=stwoprime,
                                        y=-1,
                                        z=-1,
                                        a=2,
                                        b=-2 * zetatwooffset,
                                        c=2,
                                        d=-2 * zetatwoprimeoffset,
                                        f=1,
                                        alpha=(zetatwooffset + zetatwoprimeoffset),
                                    )
                                    * pImn(
                                        p=0,
                                        m=sone,
                                        n=soneprime,
                                        y=-1,
                                        z=-1,
                                        a=2,
                                        b=-2 * zetaoneoffset,
                                        c=2,
                                        d=-2 * zetaoneprimeoffset,
                                        f=1,
                                        alpha=(zetaoneoffset + zetaoneprimeoffset),
                                    )
                                    * np.exp(
                                        -0.5
                                        * (zetaoneoffset ** 2 + zetaoneprimeoffset ** 2)
                                    )
                                )

                                matelem += (
                                    potential1pos
                                    + potential1neg
                                    + potential2pos
                                    + potential2neg
                                    + potential3pos
                                    + potential3neg
                                    + potentialconst
                                )
                            i = (
                                (self.qbt.num_exc + 1) * (sone)
                                + stwo
                                + min_index * (self.qbt.num_exc + 1) ** 2
                            )
                            j = (
                                (self.qbt.num_exc + 1) * (soneprime)
                                + stwoprime
                                + min_index * (self.qbt.num_exc + 1) ** 2
                            )
                            potential_babusci_mat[i, j] += matelem
        block1_babusci = potential_babusci_mat[
            0:num_states_per_min, 0:num_states_per_min
        ]
        block2_babusci = potential_babusci_mat[
            num_states_per_min : 2 * num_states_per_min,
            num_states_per_min : 2 * num_states_per_min,
        ]
        block1_ladder = potential_ladder_mat[0:num_states_per_min, 0:num_states_per_min]
        block2_ladder = potential_ladder_mat[
            num_states_per_min : 2 * num_states_per_min,
            num_states_per_min : 2 * num_states_per_min,
        ]
        assert np.allclose(block1_babusci, block1_ladder, atol=1e-7)
        assert np.allclose(block2_babusci, block2_ladder, atol=1e-7)
