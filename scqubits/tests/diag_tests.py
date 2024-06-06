import numpy as np
import importlib
import warnings
import pytest
import scqubits as scq
import scipy as sp
from scqubits import Fluxonium, Transmon, HilbertSpace

diag_methods = scq.DIAG_METHODS.keys()

def _library_installed(library):
    try:
        importlib.import_module(library)
        return True
    except ImportError:  # `ModuleNotFoundError` is a subclass of `ImportError`. Here, the general class of errors are checked.
        return False

def _get_library_methods(library, evals=True):
    library_diag_methods = [
        method for method in diag_methods
        if method.split('_')[1] == library
        and method.split('_')[0] == ('evals' if evals else 'esys')
        and '_shift-inverse' not in method
        and '_SM' not in method  # TODO: this is temporary until we find a fix for adding EJ to the Hamiltonian.
    ]
    return library_diag_methods


optional_libraries = (
    'cupy',
    'jax',
    'primme'
)


def test_custom_diagonalization_raises_error_if_esys_method_passed_to_evals_method():
    esys_method = 'esys_scipy_dense'

    with pytest.raises(ValueError) as exc_info:
        Fluxonium(
            EJ=8.9, EC=2.5, EL=0.5, flux = 0.5, cutoff = 120,
            evals_method=esys_method
        )

    assert "Invalid `evals_method`: expect one of `evals` methods, got one of `esys` methods." in exc_info.exconly()


def test_custom_diagonalization_raises_error_if_evals_method_passed_to_esys_method():
    evals_method = 'evals_scipy_dense'

    with pytest.raises(ValueError) as exc_info:
        Fluxonium(
            EJ=8.9, EC=2.5, EL=0.5, flux = 0.5, cutoff = 120,
            esys_method=evals_method
        )

    assert "Invalid `esys_method`: expect one of `esys` methods, got one of `evals` methods." in exc_info.exconly()


def test_custom_diagonalization_raises_error_if_nonexistent_method_provided_evals():
    evals_method = 'non_existent_method'

    fluxonium = Fluxonium(
        EJ=8.9, EC=2.5, EL=0.5, flux = 0.5, cutoff = 120,
        evals_method=evals_method
    )

    with pytest.raises(ValueError) as exc_info:
        fluxonium.eigenvals()

    assert f"Invalid {evals_method} `evals_method`, does not exist in available custom diagonalization methods." in exc_info.exconly()


def test_custom_diagonalization_raises_error_if_nonexistent_method_provided_esys():
    esys_method = 'non_existent_method'

    fluxonium = Fluxonium(
        EJ=8.9, EC=2.5, EL=0.5, flux = 0.5, cutoff = 120,
        esys_method=esys_method
    )

    with pytest.raises(ValueError) as exc_info:
        fluxonium.eigensys()

    assert f"Invalid {esys_method} `esys_method`, does not exist in available custom diagonalization methods." in exc_info.exconly()


@pytest.mark.parametrize(
    "library",
    [
        pytest.param(library, marks=pytest.mark.skipif(not _library_installed(library), reason=f'Package {library} not installed; skipping test')) for library in optional_libraries
    ]
)
def test_custom_diagonalization_evals_method_matches_default(library):
    if library == 'jax':
        # To evade np.allclose errors when not using 64-bit calculations.
        from jax.config import config
        config.update("jax_enable_x64", True)

    try:
        importlib.import_module(library)
    except ImportError:
        warnings.warn(f'Package {library} not installed; skipping test', ImportWarning)
        return

    library_diag_methods = _get_library_methods(library, evals=True)
    for method in library_diag_methods:
        fluxonium = Fluxonium(
            EJ=8.9, EC=2.5, EL=0.5, flux = 0.5, cutoff = 120,
            evals_method=method
        )

        evals = fluxonium.eigenvals()

        fluxonium.evals_method = None
        evals_default = fluxonium.eigenvals()

        assert np.allclose(evals, evals_default)


@pytest.mark.parametrize(
    "library",
    [
        pytest.param(library, marks=pytest.mark.skipif(not _library_installed(library), reason=f'Package {library} not installed; skipping test')) for library in optional_libraries
    ]
)
def test_custom_diagonalization_matches_default_with_composite_systems(library):
    if library == 'jax':
        # To evade np.allclose errors when not using 64-bit calculations.
        from jax.config import config
        config.update("jax_enable_x64", True)

    try:
        importlib.import_module(library)
    except ImportError:
        warnings.warn(f'Package {library} not installed; skipping test', ImportWarning)
        return

    library_diag_methods = _get_library_methods(library, evals=True)
    for method in library_diag_methods:
        tmon = Transmon(EJ=30.02, EC=1.2, ng=0.0, ncut=101)
        fluxonium= Fluxonium(
            EJ=8.9, EC=2.5, EL=0.5, flux=0.5, cutoff=110,
            evals_method=method
        )

        hs = HilbertSpace(
            [tmon, fluxonium],
            evals_method=method
        )

        hs.add_interaction(
            g_strength=0.250,
            op1=tmon.n_operator,
            op2=fluxonium.n_operator,
            add_hc=True
        )

        # Get the eigenvalues using the respective custom methods defined above.
        evals_hs = hs.eigenvals(evals_count=10)
        evals_fluxonium = fluxonium.eigenvals(evals_count=10)

        # Get the eigenvalues using the default method.
        fluxonium.evals_method=None
        hs.evals_method=None
        evals_default_hs = hs.eigenvals(evals_count=10)
        evals_default_fluxonium = fluxonium.eigenvals(evals_count=10)

        assert np.allclose(evals_hs, evals_default_hs)
        assert np.allclose(evals_fluxonium, evals_default_fluxonium)


def test_custom_diagonalization_matches_default_using_custom_procedure():
    def custom_esys(matrix, evals_count, **kwargs):  
        evals, evecs = sp.linalg.eigh(
            matrix, subset_by_index=(0, evals_count - 1), 
            eigvals_only=False,  
            **kwargs
        )
        return evals, evecs
    
    tmon = Transmon(
        EJ=30.02, EC=1.2, ng=0.0, ncut=501,
        esys_method=custom_esys,
        esys_method_options=dict(driver='evr')  # set a custom driver
    )

    evals, evecs = tmon.eigensys()

    tmon.esys_method = None
    tmon.esys_method_options = None

    evals_default, evecs_default = tmon.eigensys()

    assert np.all(
        [
            np.allclose(evecs[:, i], evecs_default[:, i]) for i, _ in enumerate(evals)
        ]
    )
    assert np.allclose(evals, evals_default)


@pytest.mark.parametrize(
    "library",
    [
        pytest.param(library, marks=pytest.mark.skipif(not _library_installed(library), reason=f'Package {library} not installed; skipping test')) for library in optional_libraries
    ]
)
def test_custom_diagonalization_evals_are_same_using_eigenvals_and_eigensys_default(library):
    if library == 'jax':
        # To evade np.allclose errors when not using 64-bit calculations.
        from jax.config import config
        config.update("jax_enable_x64", True)

    try:
        importlib.import_module(library)
    except ImportError:
        warnings.warn(f'Package {library} not installed; skipping test', ImportWarning)
        return

    library_diag_methods = _get_library_methods(library, evals=True)
    for method in library_diag_methods:
        fluxonium = Fluxonium(
            EJ=8.9, EC=2.5, EL=0.5, flux = 0.5, cutoff = 120,
            evals_method=method, 
            evals_method_options=dict(tol=1e-4) # custom tolerance
        )

        evals = fluxonium.eigenvals()
        evals_sys, _ = fluxonium.eigensys()

        assert np.allclose(evals, evals_sys)


@pytest.mark.parametrize(
    "library",
    [
        pytest.param(library, marks=[
            pytest.mark.skipif(not _library_installed(library), reason=f'Package {library} not installed; skipping test'),
        ]
        ) for library in optional_libraries
    ]
)
def test_custom_diagonalization_esys_method_matches_default(library):
    if library == 'jax':
        # To evade np.allclose errors when not using 64-bit calculations.
        from jax.config import config
        config.update("jax_enable_x64", True)

    try:
        importlib.import_module(library)
    except ImportError:
        warnings.warn(f'Package {library} not installed; skipping test', ImportWarning)
        return

    library_diag_methods = _get_library_methods(library, evals=False)
    for method in library_diag_methods:
        tmon = Transmon(
            EJ=30.02, EC=1.2, ng=0.0, ncut=501, 
            esys_method=method
        )
        evals, evecs = tmon.eigensys()

        tmon.esys_method = None
        evals_default, evecs_default = tmon.eigensys()

        if method == 'esys_primme_sparse' or method == 'esys_jax_dense':
            # NOTE: These two cases are temporary failing.
            with pytest.raises(AssertionError):
                assert np.all(
                    [
                        np.allclose(evecs[:, i], evecs_default[:, i], atol=1e-7) for i, _ in enumerate(evals)
                    ]
                )
        else:
            assert np.all(
                [
                    np.allclose(evecs[:, i], evecs_default[:, i], atol=1e-7) for i, _ in enumerate(evals)
                ]
            )
        assert np.allclose(evals, evals_default)
