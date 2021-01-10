import scqubits as qubit
import numpy as np

EJ = 1.0
ECJ = EJ / 60.0
ALPHA = 0.8
FLUX = 0.5
ECg = 50.0 * ECJ
EJlist = np.array([1.01*EJ, 0.98*EJ, ALPHA * EJ])
nglist = np.array([0.2, 0.0])
num_exc = 4
maximum_periodic_vector_length = 8

FQV = qubit.FluxQubitVTB(
    EJ1=EJ,
    EJ2=EJ,
    EJ3=ALPHA * EJ,
    ECJ1=ECJ,
    ECJ2=ECJ,
    ECJ3=ECJ / ALPHA,
    ECg1=ECg,
    ECg2=ECg,
    ng1=nglist[0],
    ng2=nglist[1],
    flux=FLUX,
    num_exc=num_exc,
    maximum_periodic_vector_length=maximum_periodic_vector_length,
    truncated_dim=6)

FQVS = qubit.FluxQubitVTBSqueezing(
    EJ1=EJ,
    EJ2=EJ,
    EJ3=ALPHA * EJ,
    ECJ1=ECJ,
    ECJ2=ECJ,
    ECJ3=ECJ / ALPHA,
    ECg1=ECg,
    ECg2=ECg,
    ng1=nglist[0],
    ng2=nglist[1],
    flux=FLUX,
    num_exc=num_exc,
    maximum_periodic_vector_length=maximum_periodic_vector_length,
    truncated_dim=6)

FQVG = qubit.FluxQubitVTBGlobal(
    EJ1=EJ,
    EJ2=EJ,
    EJ3=ALPHA * EJ,
    ECJ1=ECJ,
    ECJ2=ECJ,
    ECJ3=ECJ / ALPHA,
    ECg1=ECg,
    ECg2=ECg,
    ng1=nglist[0],
    ng2=nglist[1],
    flux=FLUX,
    num_exc=num_exc,
    maximum_periodic_vector_length=maximum_periodic_vector_length, truncated_dim=6)

FQVGS = qubit.FluxQubitVTBGlobalSqueezing(
    EJ1=EJ,
    EJ2=EJ,
    EJ3=ALPHA * EJ,
    ECJ1=ECJ,
    ECJ2=ECJ,
    ECJ3=ECJ / ALPHA,
    ECg1=ECg,
    ECg2=ECg,
    ng1=nglist[0],
    ng2=nglist[1],
    flux=FLUX,
    num_exc=num_exc,
    maximum_periodic_vector_length=maximum_periodic_vector_length, truncated_dim=6)

evals_1 = FQV.eigenvals()
evals_2 = FQVG.eigenvals()
evals_3 = FQVS.eigenvals()
evals_4 = FQVGS.eigenvals()

FQ = qubit.FluxQubit(EJ1=EJ, EJ2=EJ, EJ3=ALPHA * EJ, ECJ1=ECJ, ECJ2=ECJ, ECJ3=ECJ / ALPHA, ECg1=ECg,
                     ECg2=ECg, ng1=nglist[0], ng2=nglist[1], flux=FLUX, ncut=10)


flux_list = np.linspace(0.46, 0.54, 21)
specdata_1 = FQV.eigenvals(evals_count=12, filename='../scqubits/tests/data/fluxqubitvtb_1.hdf5')
read_1 = qubit.read(filename='../scqubits/tests/data/fluxqubitvtb_1.hdf5')
read_1.add_data(transfer_matrix=FQV.transfer_matrix())
read_1.add_data(sorted_minima=FQV.sorted_minima())
read_1.add_data(gamma_matrix=FQV.gamma_matrix())
read_1.add_data(eigensystem_normal_modes=FQV.eigensystem_normal_modes())
read_1.add_data(Xi_matrix=FQV.Xi_matrix())
read_1.add_data(nearest_neighbors=FQV.nearest_neighbors)
read_1.add_data(kinetic_matrix=FQV.kinetic_matrix())
read_1.add_data(potential_matrix=FQV.potential_matrix())
read_1.add_data(inner_product_matrix=FQV.inner_product_matrix())
qubit.write(read_1, filename='../scqubits/tests/data/fluxqubitvtb_1.hdf5')
specdata_2 = FQV.eigensys(evals_count=4, filename='../scqubits/tests/data/fluxqubitvtb_2.hdf5')
specdata_4 = FQV.get_spectrum_vs_paramvals(param_name='flux', param_vals=flux_list, evals_count=4, get_eigenstates=True,
                                           filename='../scqubits/tests/data/fluxqubitvtb_4.hdf5')

specdata_5 = FQV.matrixelement_table('n_operator', operator_args={'j': 0},
                                     filename='../scqubits/tests/data/fluxqubitvtb_5.hdf5')

specdata_1_FQ = FQ.eigenvals(evals_count=12, filename='../scqubits/tests/data/fluxqubit_1.hdf5')
specdata_2_FQ = FQ.eigensys(evals_count=4, filename='../scqubits/tests/data/fluxqubit_2.hdf5')
specdata_4_FQ = FQ.get_spectrum_vs_paramvals(param_name='flux', param_vals=flux_list, evals_count=4,
                                             get_eigenstates=True,
                                             filename='../scqubits/tests/data/fluxqubit_4.hdf5')
specdata_5_FQ = FQ.matrixelement_table('n_1_operator', evals_count=10,
                                       filename='../scqubits/tests/data/fluxqubit_5.hdf5')
