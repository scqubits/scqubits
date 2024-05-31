import numpy as np
import importlib
import warnings
import pytest
import scqubits as scq
import scipy as sp
from scqubits import Fluxonium, Transmon, HilbertSpace

# diag_methods = scq.DIAG_METHODS.keys()
optional_libraries = (
    'cupy',
    'jax',
    'primme'
)
have_libraries = []

for library in optional_libraries:
    try:
        importlib.import_module(library)
        have_libraries.append(True)
    except ImportError:
        have_libraries.append(False)


def test_custom_diagonalization_raises_error_if_nonexistent_method_provided():
    evals_method = 'non_existent_method'
    library = evals_method.split('_')[1]
    try:
        importlib.import_module(library)
    except ModuleNotFoundError:
        warnings.warn(f'Package {library} not installed; skipping test', ImportWarning)
        return

    fluxonium = Fluxonium(
        EJ=8.9, EC=2.5, EL=0.5, flux = 0.5, cutoff = 120,
        evals_method=evals_method, 
        evals_method_options=dict(tol=1e-4) # custom tolerance
    )

    with pytest.raises(KeyError):
        fluxonium.eigenvals()


@pytest.mark.skipif(
        not have_libraries[2],
        reason=f'Package {optional_libraries[2]} not installed; skipping test'
)
def test_custom_diagonalization_evals_method_matches_default():
    evals_method = 'evals_primme_sparse'
    library = evals_method.split('_')[1]
    try:
        importlib.import_module(library)
    except ModuleNotFoundError:
        warnings.warn(f'Package {library} not installed; skipping test', ImportWarning)
        return

    fluxonium = Fluxonium(
        EJ=8.9, EC=2.5, EL=0.5, flux = 0.5, cutoff = 120,
        evals_method=evals_method, 
        evals_method_options=dict(tol=1e-4) # custom tolerance
    )

    evals = fluxonium.eigenvals()

    fluxonium.evals_method = None
    fluxonium.evals_method_options = None

    evals_default = fluxonium.eigenvals()

    assert np.allclose(evals, evals_default)


@pytest.mark.skipif(
        not have_libraries[2],
        reason=f'Package {optional_libraries[2]} not installed; skipping test'
)
def test_custom_diagonalization_evals_are_same_using_eigenvals_or_eigensys():
    evals_method = 'evals_primme_sparse'
    esys_method = 'esys_primme_sparse'
    library = evals_method.split('_')[1]

    try:
        importlib.import_module(library)
    except ModuleNotFoundError:
        warnings.warn(f'Package {library} not installed; skipping test', ImportWarning)
        return

    fluxonium = Fluxonium(
        EJ=8.9, EC=2.5, EL=0.5, flux = 0.5, cutoff = 120,
        evals_method=evals_method, 
        evals_method_options=dict(tol=1e-4) # custom tolerance
    )

    evals = fluxonium.eigenvals()
    evals_sys, _ = fluxonium.eigensys()

    assert np.allclose(evals, evals_sys)


def test_custom_diagonalization_esys_method_matches_default():
    # Using sparse diagonalization does not match eigenvectors obtained through the default and custom methods.
    esys_method = 'esys_scipy_dense'
    library = esys_method.split('_')[1]
    try:
        importlib.import_module(library)
    except ModuleNotFoundError:
        warnings.warn(f'Package {library} not installed; skipping test', ImportWarning)
        return

    tmon = Transmon(
        EJ=30.02, EC=1.2, ng=0.0, ncut=501, 
        esys_method=esys_method
    )
    evals, evecs = tmon.eigensys()

    tmon.esys_method = None

    evals_default, evecs_default = tmon.eigensys()

    assert np.all(
        [
            np.allclose(evecs[:, i], evecs_default[:, i], atol=1e-7) for i, _ in enumerate(evals)
        ]
    )
    assert np.allclose(evals, evals_default)


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
        esys_method_options=dict(driver='evr') # here we set a custom driver
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


@pytest.mark.skipif(
        not have_libraries[2],
        reason=f'Package {optional_libraries[2]} not installed; skipping test'
)
def test_custom_diagonalization_works_with_composite_systems():
    tmon = Transmon(EJ=30.02, EC=1.2, ng=0.0, ncut=101) # default diaongalization method
    fluxonium= Fluxonium(
        EJ=8.9, EC=2.5, EL=0.5, flux=0.5, cutoff=110,
        evals_method='evals_primme_sparse' # use the primme library
    )

    hs = HilbertSpace(
        [tmon, fluxonium],
        evals_method='evals_scipy_sparse' # use scipy
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

    hs.esys_method='esys_primme_sparse'
    evals_primme_sparse, _ = hs.eigensys()
    hs.esys_method='esys_scipy_sparse'
    evals_scipy_sparse, _ = hs.eigensys()

    # Get the eigenvalues using the default method.
    fluxonium.evals_method=None
    hs.evals_method=None
    evals_default_hs = hs.eigenvals(evals_count=10)
    evals_default_fluxonium = fluxonium.eigenvals(evals_count=10)

    assert np.allclose(evals_hs, evals_default_hs)
    assert np.allclose(evals_fluxonium, evals_default_fluxonium)
    # Ensure eigenvalues using different methods is the same.
    assert np.allclose(evals_primme_sparse, evals_scipy_sparse)


@pytest.mark.skipif(
        not have_libraries[1],
        reason=f'Package {optional_libraries[1]} not installed; skipping test'
)
def test_custom_diagonalization_with_jax():
    fluxonium= Fluxonium(
        EJ=8.9, EC=2.5, EL=0.5, flux = 0.5, cutoff = 120, 
        evals_method='evals_jax_dense'
    ) 
    evals = fluxonium.eigenvals()
    fluxonium.evals_method = None
    evals_default = fluxonium.eigenvals()

    assert np.allclose(evals, evals_default, rtol=1e-4)  # rtol needs to be set to check closeness here.


@pytest.mark.skipif(
        not have_libraries[0],
        reason=f'Package {optional_libraries[0]} not installed; skipping test'
)
def test_custom_diagonalization_with_cupy():
    fluxonium= Fluxonium(
        EJ=8.9, EC=2.5, EL=0.5, flux = 0.5, cutoff = 120, 
        evals_method='evals_cupy_dense'
    ) 
    evals = fluxonium.eigenvals()
    fluxonium.evals_method = None
    evals_default = fluxonium.eigenvals()

    assert np.allclose(evals, evals_default)
