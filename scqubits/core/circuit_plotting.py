from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from scqubits.core.circuit import Subsystem

import numpy as np
import sympy as sm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray

import scqubits.core.discretization as discretization
import scqubits.core.oscillator as osc
import scqubits.core.storage as storage
import scqubits.utils.plot_defaults as defaults
import scqubits.utils.plotting as plot
from scqubits import get_units
from scqubits.io_utils.fileio_serializers import dict_serialize
from scqubits.core.circuit_utils import (
    sawtooth_potential,
    get_trailing_number,
)
from scqubits.utils.misc import (
    flatten_list_recursive,
    list_intersection,
    unique_elements_in_list,
)
from scqubits.utils.plot_utils import _process_options
from abc import ABC


class CircuitPlot(ABC):
    """Mixin providing wave-function and potential plotting for circuit subsystems."""

    # The following attributes and methods are provided by sibling mixins
    # (CircuitRoutines, CircuitSymMethods) when composed into Subsystem/Circuit.
    # They are declared here under TYPE_CHECKING so that mypy understands the
    # mixin's references to them without affecting runtime behavior.
    if TYPE_CHECKING:
        dynamic_var_indices: list[int]
        var_categories: dict[str, list[int]]
        cutoff_names: list[str]
        ext_basis: str
        hierarchical_diagonalization: bool
        subsystems: list["Subsystem"]
        external_fluxes: list[sm.Symbol]
        symbolic_params: dict[sm.Symbol, float]
        potential_symbolic: sm.Expr

        def get_osc_param(self, var_index: int, which_param: str = ...) -> float:
            """Stub: oscillator parameter for ``var_index`` provided by sibling mixin.

            Parameters
            ----------
            var_index:
                index of the variable whose oscillator parameter is requested.
            which_param:
                key identifying which oscillator parameter to retrieve.
            """
            ...

        def get_subsystem_index(self, var_index: int) -> int:
            """Stub: subsystem index for ``var_index`` provided by sibling mixin.

            Parameters
            ----------
            var_index:
                index of the variable whose subsystem index is requested.
            """
            ...

        def cutoffs_dict(self) -> dict[int, int]:
            """Stub: cutoff dictionary provided by sibling mixin."""
            ...

        def discretized_grids_dict_for_vars(
            self,
        ) -> dict[int, discretization.Grid1d]:
            """Stub: discretized grids dictionary provided by sibling mixin."""
            ...

        def eigensys(self, evals_count: int = ...) -> tuple[ndarray, ndarray]:
            """Stub: eigensystem accessor provided by sibling mixin.

            Parameters
            ----------
            evals_count:
                see :meth:`scqubits.core.qubit_base.QubitBaseClass.eigensys`
            """
            ...

    # ****************************************************************
    # ************* Functions for plotting wave function *************
    # ****************************************************************
    def _recursive_basis_change(
        self, wf_reshaped, wf_dim: int, subsystem, relevant_indices=None
    ):
        """Change the basis recursively, reversing hierarchical diagonalization.

        Parameters
        ----------
        wf_reshaped:
            Wave function array reshaped to expose subsystem dimensions.
        wf_dim:
            The dimension of the wave function which needs to be rewritten in
            terms of the initial basis.
        subsystem:
            Subsystem whose eigenbasis is being undone along ``wf_dim``.
        relevant_indices:
            Variable indices that should still be resolved in their original
            basis after the recursion.
        """
        U_subsys = subsystem.eigensys(evals_count=subsystem.truncated_dim)[
            1
        ]  # eigensys(evals_count=subsystem.truncated_dim)
        wf_sublist = list(range(len(wf_reshaped.shape)))
        U_sublist = [wf_dim, len(wf_sublist)]
        target_sublist = wf_sublist.copy()
        target_sublist[wf_dim] = len(wf_sublist)
        wf_new_basis = np.einsum(
            wf_reshaped, wf_sublist, U_subsys.T, U_sublist, target_sublist
        )
        if subsystem.hierarchical_diagonalization:
            wf_shape = list(wf_new_basis.shape)
            wf_shape[wf_dim] = [
                sub_subsys.truncated_dim for sub_subsys in subsystem.subsystems
            ]
            wf_new_basis = wf_new_basis.reshape(flatten_list_recursive(wf_shape))
            for sub_subsys_index, sub_subsys in enumerate(subsystem.subsystems):
                if len(set(relevant_indices) & set(sub_subsys.dynamic_var_indices)) > 0:
                    wf_new_basis = self._recursive_basis_change(
                        wf_new_basis,
                        wf_dim + sub_subsys_index,
                        sub_subsys,
                        relevant_indices=relevant_indices,
                    )
        else:
            if len(set(relevant_indices) & set(subsystem.dynamic_var_indices)) > 0:
                wf_shape = list(wf_new_basis.shape)
                wf_shape[wf_dim] = [
                    (
                        getattr(subsystem, cutoff_attrib)
                        if "ext" in cutoff_attrib
                        else (2 * getattr(subsystem, cutoff_attrib) + 1)
                    )
                    for cutoff_attrib in subsystem.cutoff_names
                ]
                wf_new_basis = wf_new_basis.reshape(flatten_list_recursive(wf_shape))
        return wf_new_basis

    def _basis_change_harm_osc_to_n(
        self, wf_original_basis, wf_dim, var_index, grid_n: discretization.Grid1d
    ):
        """Change the basis from harmonic oscillator to charge (``n``) basis.

        Parameters
        ----------
        wf_original_basis:
            Wave function array in the harmonic-oscillator basis.
        wf_dim:
            Dimension of ``wf_original_basis`` along which the change of basis
            is applied.
        var_index:
            Index of the circuit variable being transformed.
        grid_n:
            Grid used for the target charge basis.
        """
        U_ho_n = np.array(
            [
                osc.harm_osc_wavefunction(
                    n,
                    grid_n.make_linspace(),
                    abs(self.get_osc_param(var_index, which_param="length")),
                )
                for n in range(getattr(self, "cutoff_ext_" + str(var_index)))
            ]
        )
        wf_sublist = [idx for idx, _ in enumerate(wf_original_basis.shape)]
        U_sublist = [wf_dim, len(wf_sublist)]
        target_sublist = wf_sublist.copy()
        target_sublist[wf_dim] = len(wf_sublist)
        wf_new_basis = np.einsum(
            wf_original_basis, wf_sublist, U_ho_n.T, U_sublist, target_sublist
        )
        return wf_new_basis

    def _basis_change_harm_osc_to_phi(
        self, wf_original_basis, wf_dim, var_index, grid_phi: discretization.Grid1d
    ):
        """Change the basis from harmonic oscillator to discretized phi basis.

        Parameters
        ----------
        wf_original_basis:
            Wave function array in the harmonic-oscillator basis.
        wf_dim:
            Dimension of ``wf_original_basis`` along which the change of basis
            is applied.
        var_index:
            Index of the circuit variable being transformed.
        grid_phi:
            Grid used for the target discretized phi basis.
        """
        U_ho_phi = np.array(
            [
                osc.harm_osc_wavefunction(
                    n,
                    grid_phi.make_linspace(),
                    abs(self.get_osc_param(var_index, which_param="length")),
                )
                for n in range(getattr(self, "cutoff_ext_" + str(var_index)))
            ]
        )
        wf_sublist = [idx for idx, _ in enumerate(wf_original_basis.shape)]
        U_sublist = [wf_dim, len(wf_sublist)]
        target_sublist = wf_sublist.copy()
        target_sublist[wf_dim] = len(wf_sublist)
        wf_ext_basis = np.einsum(
            wf_original_basis, wf_sublist, U_ho_phi, U_sublist, target_sublist
        )
        return wf_ext_basis

    def _basis_change_n_to_phi(
        self, wf_original_basis, wf_dim, var_index, grid_phi: discretization.Grid1d
    ):
        """Change the basis from charge (``n``) to discretized phi basis.

        Parameters
        ----------
        wf_original_basis:
            Wave function array in the charge basis.
        wf_dim:
            Dimension of ``wf_original_basis`` along which the change of basis
            is applied.
        var_index:
            Index of the circuit variable being transformed.
        grid_phi:
            Grid used for the target discretized phi basis.
        """
        U_n_phi = np.array(
            [
                np.exp(n * grid_phi.make_linspace() * 1j)
                for n in range(
                    -getattr(self, "cutoff_n_" + str(var_index)),
                    getattr(self, "cutoff_n_" + str(var_index)) + 1,
                )
            ]
        )
        wf_sublist = list(range(len(wf_original_basis.shape)))
        U_sublist = [wf_dim, len(wf_sublist)]
        target_sublist = wf_sublist.copy()
        target_sublist[wf_dim] = len(wf_sublist)
        wf_ext_basis = np.einsum(
            wf_original_basis, wf_sublist, U_n_phi, U_sublist, target_sublist
        )
        return wf_ext_basis

    def _get_var_dim_for_reshaped_wf(self, wf_var_indices, var_index):
        """Return the axis of the reshaped wave function corresponding to ``var_index``.

        Parameters
        ----------
        wf_var_indices:
            Variable indices kept as separate axes in the reshaped wave function.
        var_index:
            Variable index whose axis position is requested.
        """
        wf_dim = 0
        if not self.hierarchical_diagonalization:
            return self.dynamic_var_indices.index(var_index)
        for subsys in self.subsystems:
            intersection = list_intersection(subsys.dynamic_var_indices, wf_var_indices)
            if len(intersection) > 0 and var_index not in intersection:
                if subsys.hierarchical_diagonalization:
                    wf_dim += subsys._get_var_dim_for_reshaped_wf(
                        wf_var_indices, var_index
                    )
                else:
                    wf_dim += len(subsys.dynamic_var_indices)
            elif len(intersection) > 0 and var_index in intersection:
                if subsys.hierarchical_diagonalization:
                    wf_dim += subsys._get_var_dim_for_reshaped_wf(
                        wf_var_indices, var_index
                    )
                else:
                    wf_dim += subsys.dynamic_var_indices.index(var_index)
                break
            else:
                wf_dim += 1
        return wf_dim

    def _dims_to_be_summed(self, var_indices: tuple[int], num_wf_dims) -> list[int]:
        """Return the wave-function axes to sum over for a marginal in ``var_indices``.

        Parameters
        ----------
        var_indices:
            Variable indices that should be retained as plotting axes.
        num_wf_dims:
            Total number of dimensions of the reshaped wave function.
        """
        all_var_indices = self.dynamic_var_indices
        non_summed_dims = []
        for var_index in all_var_indices:
            if var_index in var_indices:
                non_summed_dims.append(
                    self._get_var_dim_for_reshaped_wf(var_indices, var_index)
                )
        return [dim for dim in range(num_wf_dims) if dim not in non_summed_dims]

    def _reshape_and_change_to_variable_basis(
        self, wf: ndarray, var_indices: tuple[int]
    ) -> ndarray:
        """Reshape the wave function and change to the variable basis.

        When hierarchical diagonalization is used, the wave function is first
        rotated back to the basis in which the variables were originally
        defined. The result is then reshaped so that each variable index is
        represented as a separate dimension.

        Parameters
        ----------
        wf:
            Wave function vector to be reshaped.
        var_indices:
            Variable indices to be exposed as separate axes.
        """
        if self.hierarchical_diagonalization:
            subsys_index_for_var_index = unique_elements_in_list(
                [self.get_subsystem_index(index) for index in var_indices]
            )  # getting the subsystem index for each of the variable indices
            subsys_index_for_var_index.sort()

            subsys_trunc_dims = [sys.truncated_dim for sys in self.subsystems]
            # reshaping the wave functions to truncated dims of subsystems
            wf_hd_reshaped = wf.reshape(*subsys_trunc_dims)

            # **** Converting to the basis in which the variables are defined *****
            wf_original_basis = wf_hd_reshaped
            for subsys_index in subsys_index_for_var_index:
                wf_dim = 0
                for sys_index in range(subsys_index):
                    if sys_index in subsys_index_for_var_index:
                        wf_dim += len(self.subsystems[sys_index].dynamic_var_indices)
                    else:
                        wf_dim += 1
                wf_original_basis = self._recursive_basis_change(
                    wf_original_basis,
                    wf_dim,
                    self.subsystems[subsys_index],
                    relevant_indices=var_indices,
                )
        else:
            wf_original_basis = wf.reshape(
                *[
                    (
                        getattr(self, cutoff_attrib)
                        if "ext" in cutoff_attrib
                        else (2 * getattr(self, cutoff_attrib) + 1)
                    )
                    for cutoff_attrib in self.cutoff_names
                ]
            )
        return wf_original_basis

    def _basis_for_var_index(self, var_index: int) -> str:
        """Return the ``ext_basis`` of the leaf subsystem owning ``var_index``.

        Parameters
        ----------
        var_index:
            Variable index whose owning leaf subsystem's basis is queried.
        """
        if self.hierarchical_diagonalization:
            subsys = self.subsystems[self.get_subsystem_index(var_index)]
            return subsys._basis_for_var_index(var_index)
        else:
            if var_index in self.var_categories["extended"]:
                return self.ext_basis
            else:
                return "periodic"

    def _change_to_phi_basis(
        self,
        wf_original_basis: ndarray,
        var_indices: tuple[int],
        grids_dict: dict[int, discretization.Grid1d | ndarray],
        change_discrete_charge_to_phi: bool,
    ):
        """Change the basis of the variable indices to the discretized phi basis.

        Parameters
        ----------
        wf_original_basis:
            Wave function in the basis in which the variables were originally
            defined.
        var_indices:
            Variable indices whose axes are converted to the phi basis.
        grids_dict:
            Mapping from variable index to the grid used for the target basis.
        change_discrete_charge_to_phi:
            If True, periodic (charge-basis) variables are also converted to
            the discretized phi basis; otherwise they are left in the charge
            basis.
        """
        wf_ext_basis = wf_original_basis
        for var_index in var_indices:
            # finding the dimension corresponding to the var_index
            if not self.hierarchical_diagonalization:
                wf_dim = self.dynamic_var_indices.index(var_index)
            else:
                wf_dim = self._get_var_dim_for_reshaped_wf(var_indices, var_index)

            var_basis = self._basis_for_var_index(var_index)

            if var_basis == "harmonic":
                wf_ext_basis = self._basis_change_harm_osc_to_phi(
                    wf_ext_basis,
                    wf_dim,
                    var_index,
                    cast(discretization.Grid1d, grids_dict[var_index]),
                )
            elif var_basis == "periodic" and change_discrete_charge_to_phi:
                wf_ext_basis = self._basis_change_n_to_phi(
                    wf_ext_basis,
                    wf_dim,
                    var_index,
                    cast(discretization.Grid1d, grids_dict[var_index]),
                )
        return wf_ext_basis

    def generate_wf_plot_data(
        self,
        which: int = 0,
        mode: str = "abs-sqr",
        var_indices: tuple[int] = (1,),
        eigensys: tuple[ndarray, ndarray] | None = None,
        change_discrete_charge_to_phi: bool = True,
        grids_dict: dict[int, discretization.Grid1d] | None = None,
    ):
        """Return the processed wave function for the specified variables.

        Parameters
        ----------
        which:
            integer to choose which wave function to plot
        mode:
            "abs", "real", "imag", "abs-sqr" - decides which part of the wave
            function is plotted.
        var_indices:
            A tuple containing the indices of the variables chosen to plot the
            wave function in. Should not have more than 2 entries.
        eigensys:
            eigenvalues and eigenstates of the Circuit instance; if not provided,
            calling this method will perform a diagonalization to obtain these.
        change_discrete_charge_to_phi:
            If True, periodic (charge-basis) variables are converted to the
            discretized phi basis before processing; otherwise they remain in
            the charge basis.
        grids_dict:
            A dictionary which pairs var indices with the requested grids used to create
            the plot.

        Notes
        -----
        For ``mode="real"`` and ``mode="imag"`` the imaginary or real part of
        the wave function is dropped via :func:`numpy.real`/:func:`numpy.imag`.
        """
        # checking to see if eigensys needs to be generated
        if eigensys is None:
            _, wfs = self.eigensys(evals_count=which + 1)
        else:
            _, wfs = eigensys

        wf = wfs[:, which]
        # change the wf to the basis in which the variables were initially defined
        wf_original_basis = self._reshape_and_change_to_variable_basis(
            wf=wf, var_indices=var_indices
        )

        # making a basis change to the desired basis for every var_index
        wf_ext_basis = self._change_to_phi_basis(
            wf_original_basis,
            var_indices=var_indices,
            grids_dict=cast("dict[int, discretization.Grid1d | ndarray]", grids_dict),
            change_discrete_charge_to_phi=change_discrete_charge_to_phi,
        )

        # sum over the dimensions not relevant to the ones in var_indices
        # finding the dimensions which needs to be summed over
        dims_to_be_summed = self._dims_to_be_summed(
            var_indices, len(wf_ext_basis.shape)
        )
        # summing over the dimensions
        # summing over the dimensions
        if mode == "abs-sqr":
            wf_plot = np.sum(
                np.abs(wf_ext_basis) ** 2,
                axis=tuple(dims_to_be_summed),
            )
            return wf_plot
        if mode == "abs":
            if len(dims_to_be_summed) == 0:
                return np.abs(wf_ext_basis)
            else:
                raise AttributeError(
                    "Cannot plot the absolute value of the wave function in more than 2 dimensions."
                )
        elif mode == "real":
            if len(dims_to_be_summed) == 0:
                return np.real(wf_ext_basis)
            else:
                raise AttributeError(
                    "Cannot plot the real part of the wave function in more than 2 dimensions."
                )
        elif mode == "imag":
            if len(dims_to_be_summed) == 0:
                return np.imag(wf_ext_basis)
            else:
                raise AttributeError(
                    "Cannot plot the imaginary part of the wave function in more than 2 dimensions."
                )

    def plot_wavefunction(
        self,
        which=0,
        mode: str = "abs-sqr",
        var_indices: tuple[int] = (1,),
        esys: tuple[ndarray, ndarray] | None = None,
        change_discrete_charge_to_phi: bool = True,
        zero_calibrate: bool = True,
        grids_dict: dict[int, discretization.Grid1d] = {},
        **kwargs,
    ) -> tuple[Figure, Axes]:
        """Plot the wavefunction in the requested variables.

        At most two variables may be specified as plotting axes. If the number
        of plotting variables is smaller than the number of variables in the
        circuit, the marginal probability distribution (norm-squared wave
        function integrated over the remaining variables) is plotted instead.

        Parameters
        ----------
        which:
            integer to choose which wave function to plot
        mode:
            ``"abs"``, ``"real"``, ``"imag"``, or ``"abs-sqr"`` — selects which
            part of the wave function is plotted (default: ``"abs-sqr"``)
        var_indices:
            tuple containing the indices of the variables chosen as plot axes;
            should not have more than 2 entries
        esys:
            object returned by :meth:`eigensys`; used to avoid re-evaluating
            the eigensystem if already computed
        change_discrete_charge_to_phi:
            if ``True``, plot in the phi basis for periodic variables; if
            ``False``, plot in the charge basis for periodic variables
        zero_calibrate:
            if ``True``, colors are adjusted to use zero wavefunction amplitude
            as the neutral color in the palette
        grids_dict:
            dict pairing var indices with the grids used to create the plot.
            For extended variables, grids should be of type
            :class:`scqubits.core.discretization.Grid1d`. When the discretized
            phi basis is used for an extended variable, the diagonalization
            grid is used instead. For periodic variables, the grid is used
            only if ``change_discrete_charge_to_phi`` is ``True``, specified as
            an integer point count over ``[-pi, pi]``. Missing grids fall back
            to defaults.
        **kwargs:
            plotting parameters

        Returns
        -------
        ``(Figure, Axes)`` tuple for further editing.
        """
        if len(var_indices) > 2:
            raise AttributeError(
                "Cannot plot wave function in more than 2 dimensions. The number of "
                "dimensions should be less than 2."
            )
        var_indices = np.sort(var_indices)  # type: ignore[assignment]
        grids_per_varindex_dict = grids_dict or self.discretized_grids_dict_for_vars()

        plot_data = self.generate_wf_plot_data(
            which=which,
            mode=mode,
            var_indices=var_indices,
            eigensys=esys,
            change_discrete_charge_to_phi=change_discrete_charge_to_phi,
            grids_dict=grids_per_varindex_dict,
        )

        var_types = []

        for var_index in var_indices:
            if var_index in self.var_categories["periodic"]:
                if not change_discrete_charge_to_phi:
                    var_types.append("Charge in units of 2e, periodic variable:")
                else:
                    var_types.append("Dimensionless flux, periodic variable:")
            if var_index in self.var_categories["extended"]:
                var_types.append("Dimensionless flux, extended variable:")

        if len(var_indices) == 1:
            return self._plot_wf_pdf_1D(
                plot_data,
                mode,
                var_indices,
                grids_per_varindex_dict,
                change_discrete_charge_to_phi,
                kwargs,
            )

        elif len(var_indices) == 2:
            return self._plot_wf_pdf_2D(
                plot_data,
                var_indices,
                grids_per_varindex_dict,
                change_discrete_charge_to_phi,
                zero_calibrate=zero_calibrate,
                kwargs=kwargs,
            )

    def _plot_wf_pdf_2D(
        self,
        wf_plot: ndarray,
        var_indices,
        grids_per_varindex_dict,
        change_discrete_charge_to_phi: bool,
        zero_calibrate: bool,
        kwargs,
    ) -> tuple[Figure, Axes]:
        """Render a 2D wavefunction probability density on the supplied grids.

        Internal helper for :meth:`plot_wavefunction` when two variables are
        chosen as plot axes.

        Parameters
        ----------
        wf_plot:
            2D wavefunction values (real, ``abs``, ``imag``, or ``abs-sqr``)
        var_indices:
            tuple of two variable indices selecting the plot axes
        grids_per_varindex_dict:
            mapping from variable index to its
            :class:`scqubits.core.discretization.Grid1d`
        change_discrete_charge_to_phi:
            if ``True``, periodic variables are plotted in phi basis
        zero_calibrate:
            if ``True``, neutral color in the palette corresponds to zero
        kwargs:
            extra keyword arguments forwarded to
            :func:`scqubits.utils.plotting.wavefunction2d`
        """
        # check if each variable is periodic
        grids = []
        labels = []
        for index_order in [1, 0]:
            if not change_discrete_charge_to_phi and (
                var_indices[index_order] in self.var_categories["periodic"]
            ):
                grids.append(
                    [
                        -getattr(self, "cutoff_n_" + str(var_indices[index_order])),
                        getattr(self, "cutoff_n_" + str(var_indices[index_order])),
                        2 * getattr(self, "cutoff_n_" + str(var_indices[index_order]))
                        + 1,
                    ]
                )
                labels.append(r"$n_{{{}}}$".format(str(var_indices[index_order])))
            else:
                grids.append(
                    list(
                        grids_per_varindex_dict[var_indices[index_order]]
                        .get_initdata()
                        .values()
                    ),
                )
                labels.append(r"$\theta_{{{}}}$".format(str(var_indices[index_order])))
        wavefunc_grid = discretization.GridSpec(np.asarray(grids))

        wavefunc = storage.WaveFunctionOnGrid(wavefunc_grid, wf_plot)
        # obtain fig and axes from
        fig, axes = plot.wavefunction2d(
            wavefunc,
            zero_calibrate=zero_calibrate,
            ylabel=labels[1],
            xlabel=labels[0],
            **kwargs,
        )
        # change frequency of tick mark for variables in charge basis
        # also force the tick marks to be integers
        if not change_discrete_charge_to_phi:
            if var_indices[0] in self.var_categories["periodic"]:
                if getattr(self, "cutoff_n_" + str(var_indices[0])) >= 6:
                    axes.yaxis.set_major_locator(plt.MaxNLocator(13, integer=True))
                else:
                    axes.yaxis.set_major_locator(
                        plt.MaxNLocator(
                            1 + 2 * getattr(self, "cutoff_n_" + str(var_indices[0])),
                            integer=True,
                        )
                    )
            if var_indices[1] in self.var_categories["periodic"]:
                if getattr(self, "cutoff_n_" + str(var_indices[1])) >= 15:
                    axes.xaxis.set_major_locator(plt.MaxNLocator(31, integer=True))
                else:
                    axes.xaxis.set_major_locator(
                        plt.MaxNLocator(
                            1 + 2 * getattr(self, "cutoff_n_" + str(var_indices[1])),
                            integer=True,
                        )
                    )

        return fig, axes

    def _plot_wf_pdf_1D(
        self,
        wf_plot: ndarray,
        mode: str,
        var_indices,
        grids_per_varindex_dict,
        change_discrete_charge_to_phi: bool,
        kwargs,
    ) -> tuple[Figure, Axes]:
        """Render a 1D wavefunction (or marginal probability density).

        Internal helper for :meth:`plot_wavefunction` when a single variable
        is chosen as the plot axis.

        Parameters
        ----------
        wf_plot:
            1D wavefunction values (real, ``abs``, ``imag``, or ``abs-sqr``)
        mode:
            ``"abs"``, ``"real"``, ``"imag"``, or ``"abs-sqr"``
        var_indices:
            single-element tuple of the variable index selecting the plot axis
        grids_per_varindex_dict:
            mapping from variable index to its
            :class:`scqubits.core.discretization.Grid1d`
        change_discrete_charge_to_phi:
            if ``True``, periodic variables are plotted in phi basis
        kwargs:
            extra keyword arguments forwarded to the underlying matplotlib
            plot helper
        """
        var_index = var_indices[0]

        if not change_discrete_charge_to_phi and (
            var_indices[0] in self.var_categories["periodic"]
        ):
            ncut = self.cutoffs_dict()[var_indices[0]]
            wavefunc = storage.WaveFunction(
                basis_labels=np.linspace(-ncut, ncut, 2 * ncut + 1),
                amplitudes=wf_plot,
            )
            kwargs = {
                **defaults.wavefunction1d_discrete("abs_sqr"),
                **kwargs,
            }
            wavefunc.basis_labels = np.arange(
                -getattr(self, "cutoff_n_" + str(var_index)),
                getattr(self, "cutoff_n_" + str(var_index)) + 1,
            )
            fig, axes = plot.wavefunction1d_discrete(wavefunc, **kwargs)
            # changing the tick frequency for axes
            if getattr(self, "cutoff_n_" + str(var_index)) >= 7:
                axes.xaxis.set_major_locator(plt.MaxNLocator(15, integer=True))
            else:
                axes.xaxis.set_major_locator(
                    plt.MaxNLocator(1 + 2 * getattr(self, "cutoff_n_" + str(var_index)))
                )
        else:
            wavefunc = storage.WaveFunction(
                basis_labels=grids_per_varindex_dict[var_indices[0]].make_linspace(),
                amplitudes=wf_plot,
            )
            if mode == "abs":
                ylabel = r"$|\psi(\theta_{{{}}})|$".format(str(var_indices[0]))
            elif mode == "abs-sqr":
                ylabel = r"$|\psi(\theta_{{{}}})|^2$".format(str(var_indices[0]))
            elif mode == "real":
                ylabel = r"$\mathrm{{Re}}(\psi(\theta_{{{}}}))$".format(
                    str(var_indices[0])
                )
            elif mode == "imag":
                ylabel = r"$\mathrm{{Im}}(\psi(\theta_{{{}}}))$".format(
                    str(var_indices[0])
                )
            fig, axes = plot.wavefunction1d_nopotential(
                wavefunc,
                0,
                xlabel=r"$\theta_{{{}}}$".format(str(var_indices[0])),
                ylabel=ylabel,
                **kwargs,
            )
        return fig, axes

    # ****************************************************************
    # ************* Functions for plotting potential *****************
    # ****************************************************************
    def potential_energy(self, **kwargs) -> ndarray:
        r"""Return the circuit potential evaluated on a user-specified grid.

        If a variable's value is not supplied via ``kwargs``, the current
        instance attribute (for parameters/external fluxes) or an error (for
        ``θ<index>`` coordinates) is used.

        Parameters
        ----------
        θ<index>:
            value(s) for variable :math:`\theta_i` in the potential
        """
        periodic_indices = self.var_categories["periodic"]
        discretized_ext_indices = self.var_categories["extended"]
        var_categories = discretized_ext_indices + periodic_indices

        # substituting the parameters
        potential_sym = self.potential_symbolic.subs("I", 1)
        for ext_flux in self.external_fluxes:
            potential_sym = potential_sym.subs(ext_flux, ext_flux * 2 * np.pi)

        # constructing the grids
        parameters = dict.fromkeys(
            [f"θ{index}" for index in var_categories]
            + [var.name for var in self.external_fluxes]
            + [var.name for var in self.symbolic_params]
        )

        for var_name in kwargs:
            if isinstance(kwargs[var_name], np.ndarray):
                parameters[var_name] = kwargs[var_name]
            elif isinstance(kwargs[var_name], (int, float)):
                parameters[var_name] = kwargs[var_name]
            else:
                raise AttributeError(
                    "Only float, int or Numpy ndarray assignments are allowed."
                )

        for var_name in parameters.keys():
            if parameters[var_name] is None:
                if var_name in [
                    var.name
                    for var in list(self.symbolic_params.keys()) + self.external_fluxes
                ]:
                    parameters[var_name] = getattr(self, var_name)
                elif var_name in [f"θ{index}" for index in var_categories]:
                    raise AttributeError(var_name + " is not set.")

        # creating a meshgrid for multiple dimensions
        sweep_vars = {}
        for var_name in kwargs:
            if isinstance(kwargs[var_name], np.ndarray):
                sweep_vars[var_name] = kwargs[var_name]
        if len(sweep_vars) > 1:
            sweep_vars.update(
                zip(
                    sweep_vars,
                    np.meshgrid(*[grid for grid in sweep_vars.values()]),
                )
            )
            for var_name in sweep_vars:
                parameters[var_name] = sweep_vars[var_name]

        potential_func = sm.lambdify(
            parameters.keys(), potential_sym, [{"saw": sawtooth_potential}, "numpy"]
        )

        return potential_func(*parameters.values())

    def plot_potential(self, **kwargs) -> tuple[Figure, Axes]:
        r"""Plot the circuit potential as a function of one or two coordinates.

        At most two ``θ<index>`` variables may be supplied as numpy arrays; any
        further array argument would require more than three plot dimensions
        and is rejected.

        Parameters
        ----------
        θ<index>:
            value(s) for the variable :math:`\theta_i` occurring in the potential

        Returns
        -------
        ``(Figure, Axes)`` tuple for further editing.
        """

        periodic_indices = self.var_categories["periodic"]
        discretized_ext_indices = self.var_categories["extended"]
        var_categories = discretized_ext_indices + periodic_indices

        # constructing the grids
        parameters = dict.fromkeys(
            [f"θ{index}" for index in var_categories]
            + [var.name for var in self.external_fluxes]
            + [var.name for var in self.symbolic_params]
        )

        # filtering the plotting options
        plot_kwargs = {}
        list_of_keys = list(kwargs.keys())
        for key in list_of_keys:
            if key not in parameters:
                plot_kwargs[key] = kwargs[key]
                del kwargs[key]

        sweep_vars = {}
        for var_name in kwargs:
            if isinstance(kwargs[var_name], np.ndarray):
                sweep_vars[var_name] = kwargs[var_name]
        if len(sweep_vars) > 1:
            sweep_vars.update(zip(sweep_vars, np.meshgrid(*list(sweep_vars.values()))))
            for var_name in sweep_vars:
                parameters[var_name] = sweep_vars[var_name]

        if len(sweep_vars) > 2:
            raise AttributeError(
                "Cannot plot with a dimension greater than 3; Only give a maximum of "
                "two grid inputs"
            )

        potential_energies = self.potential_energy(**kwargs)

        fig, axes = kwargs.get("fig_ax") or plt.subplots()

        if len(sweep_vars) == 1:
            axes.plot(*(list(sweep_vars.values()) + [potential_energies]))
            axes.set_xlabel(
                r"$\theta_{{{}}}$".format(
                    get_trailing_number(list(sweep_vars.keys())[0])
                )
            )
            axes.set_ylabel("Potential energy in " + get_units())

        if len(sweep_vars) == 2:
            contourset = axes.contourf(
                *(list(sweep_vars.values()) + [potential_energies])
            )
            var_indices = [
                get_trailing_number(var_name) for var_name in list(sweep_vars.keys())
            ]
            axes.set_xlabel(r"$\theta_{{{}}}$".format(var_indices[0]))
            axes.set_ylabel(r"$\theta_{{{}}}$".format(var_indices[1]))
            cbar = plt.colorbar(contourset, ax=axes)
            cbar.set_label("Potential energy in " + get_units())
        _process_options(fig, axes, **plot_kwargs)
        return fig, axes
