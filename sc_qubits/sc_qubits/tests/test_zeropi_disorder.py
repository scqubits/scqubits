# test_zeropi_disorder.py
# meant to be run with 'nose'

from __future__ import division
from __future__ import print_function

import numpy as np

import sc_qubits as qubit



def ZeroPi_disordered_initialize():
    phi_grid = qubit.Grid1d(-8 * np.pi, 8 * np.pi, 200)

    # parameters
    EJ_CONST = 1 / 3.95

    zeropi_dis = qubit.ZeroPi(
        grid=phi_grid,
        ncut=30,
        EJ=EJ_CONST,
        dEJ=0.1,
        EL=10.0 ** (-3),
        ECJ=1 / (8.0 * EJ_CONST),
        dCJ=0.1,
        ECS=10.0 ** (-3),
        EC=None,
        ng=0.3,
        flux=0.2
    )
    return zeropi_dis


def test_ZeroPi_disordered_init():
    print("ZeroPi_disordered_disordered_init()")
    zero_pi = ZeroPi_disordered_initialize()


def test_ZeroPi_disordered_plot_potential():
    print("ZeroPi_disordered_plot_potential()")
    zero_pi = ZeroPi_disordered_initialize()
    zero_pi.plot_potential(contour_vals=np.linspace(0, 3, 25), aspect_ratio=0.12)


def test_ZeroPi_disordered_eigenvals():
    print("ZeroPi_disordered_eigenvals()")
    evals_reference = np.asarray([0.42778386, 0.42854964, 0.45620118, 0.45685018, 0.46941385, 0.47730651])
    zero_pi = ZeroPi_disordered_initialize()
    evals_calculated = zero_pi.eigenvals()
    assert np.allclose(evals_reference, evals_calculated)


def test_ZeroPi_disordered_eigenvecs():
    print("ZeroPi_disordered_eigenvecs()")
    evecs_reference = np.array([-2.46649418e-03+1.54003226e-03j,  1.12013900e-03-7.00743586e-04j,
       -4.54074767e-04+2.84608538e-04j,  1.65323473e-04-1.03821017e-04j,
       -5.43773115e-05+3.42130646e-05j,  1.62409902e-05-1.02377463e-05j,
       -4.42590584e-06+2.79512725e-06j,  1.10515739e-06-6.99232877e-07j,
       -2.53860865e-07+1.60908167e-07j,  5.38313315e-08-3.41813587e-08j,
       -1.05723036e-08+6.72478957e-09j,  1.92870934e-09-1.22889518e-09j,
       -3.27737355e-10+2.09167083e-10j,  5.20017612e-11-3.32419746e-11j,
       -7.72256826e-12+4.94437301e-12j,  1.07563210e-12-6.89758724e-13j,
       -1.40828411e-13+9.04500365e-14j,  1.73780517e-14-1.11555454e-14j,
       -1.97399352e-15+1.31991585e-15j,  2.05471928e-16-1.74114608e-16j,
       -3.49186695e-17-9.23775369e-18j,  1.42430555e-19+1.84866373e-17j,
       -1.82546341e-17-1.91470205e-17j, -1.92076396e-17+2.45087309e-17j,
        5.88541171e-17+2.07175022e-18j, -3.06766118e-17-5.03751981e-18j,
       -4.92533974e-17-6.40669972e-18j, -6.36242511e-18-1.96014108e-17j,
       -2.40953651e-18+7.12223149e-18j,  1.13162398e-17+9.33552804e-18j,
        2.16277814e-17-2.32574649e-17j, -9.44773538e-17+2.59215217e-17j,
        8.55777000e-16-4.53747446e-16j, -7.23436952e-15+4.23741429e-15j,
        6.04763240e-14-3.50426330e-14j, -4.74423907e-13+2.75141966e-13j,
        3.49743245e-12-2.03324077e-12j, -2.42034798e-11+1.41094689e-11j,
        1.56909562e-10-9.17272435e-11j, -9.50868533e-10+5.57458440e-10j,
        5.37299038e-09-3.15916873e-09j, -2.82371508e-08+1.66520525e-08j,
        1.37614540e-07-8.13991303e-08j, -6.20037887e-07+3.67878138e-07j,
        2.57373300e-06-1.53177134e-06j, -9.80616676e-06+5.85452611e-06j,
        3.41499227e-05-2.04528687e-05j, -1.08214640e-04+6.50182461e-05j,
        3.10422965e-04-1.87108600e-04j, -8.01704999e-04+4.84790036e-04j,
        1.85252705e-03-1.12384302e-03j, -3.80504065e-03+2.31583494e-03j,
        6.89712120e-03-4.21137791e-03j, -1.09540179e-02+6.71027943e-03j,
        1.51348027e-02-9.30153104e-03j, -1.80862413e-02+1.11515953e-02j,
        1.86197780e-02-1.15178832e-02j, -1.65046500e-02+1.02426322e-02j,
        1.26285489e-02-7.86253391e-03j, -8.38567099e-03+5.23776202e-03j,
        4.86431864e-03-3.04806659e-03j, -2.48316123e-03+1.56097085e-03j,
        1.12350469e-03-7.08510943e-04j, -4.53682595e-04+2.87009344e-04j,
        1.64531541e-04-1.04413700e-04j, -5.39042572e-05+3.43149112e-05j,
        1.60368506e-05-1.02404562e-05j, -4.35353834e-06+2.78847036e-06j,
        1.08300739e-06-6.95766533e-07j, -2.47867082e-07+1.59712246e-07j,
        5.23743503e-08-3.38459226e-08j, -1.02510699e-08+6.64355098e-09j,
        1.86393480e-09-1.21138702e-09j, -3.15726159e-10+2.05757708e-10j,
        4.99425367e-11-3.26351213e-11j, -7.39495887e-12+4.84496237e-12j,
        1.02707320e-12-6.74665086e-13j, -1.34108827e-13+8.83193831e-14j,
        1.65061075e-14-1.08723952e-14j, -1.86996116e-15+1.28353219e-15j,
        1.93990555e-16-1.65624468e-16j, -4.03122623e-17-5.71768968e-18j,
        3.98825156e-18+2.53153400e-17j, -2.07211536e-17-1.12188885e-17j,
        7.17844394e-19+3.84900590e-17j,  5.27129619e-17-1.59361834e-18j,
       -2.72264936e-17-8.86299825e-18j, -5.47871733e-17-1.10292206e-17j,
       -1.45185830e-18-2.26297650e-17j, -3.15946627e-18-8.73181304e-18j,
        1.52706769e-17+9.62243717e-19j,  1.68190944e-17-1.70201673e-17j,
       -8.15450320e-17+1.31651598e-17j,  7.73589496e-16-3.94790571e-16j,
       -6.55024763e-15+3.74350259e-15j,  5.50621348e-14-3.10941027e-14j,
       -4.33835381e-13+2.45407584e-13j,  3.21216613e-12-1.82346863e-12j,
       -2.23282805e-11+1.27254047e-11j,  1.45411551e-10-8.32063231e-11j])
    zero_pi = ZeroPi_disordered_initialize()
    _, evecs = zero_pi.eigensys(evals_count=2)
    evecs_calculated = evecs.T[1][4000:4100]
    assert np.allclose(evecs_calculated / evecs_calculated[0], evecs_reference / evecs_reference[0])


def test_ZeroPi_disordered_plot_evals_vs_paramvals():
    print("ZeroPi_disordered_plot_evals_vs_paramvals()")
    zero_pi = ZeroPi_disordered_initialize()
    ng_list = np.linspace(0, 1, 12)
    zero_pi.plot_evals_vs_paramvals('ng', ng_list, subtract_ground=True)


def test_ZeroPi_disordered_plot_wavefunction():
    print("ZeroPi_disordered_plot_wavefunction()")
    zero_pi = ZeroPi_disordered_initialize()
    zero_pi.plot_wavefunction(esys=None, which=4, mode='real')

