import scqubits as qubit
import numpy as np

EJ = 1.0
ECJ = EJ / 60.0
ALPHA = 0.8
FLUX = 0.5
ECg = 50.0 * ECJ
EJlist = np.array([1.01*EJ, 0.98*EJ, ALPHA * EJ])
nglist = np.array([0.2, 0.0])
num_exc = 2
maximum_periodic_vector_length = 3

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

FQ = qubit.FluxQubit(EJ1=EJ, EJ2=EJ, EJ3=ALPHA * EJ, ECJ1=ECJ, ECJ2=ECJ, ECJ3=ECJ / ALPHA, ECg1=ECg,
                     ECg2=ECg, ng1=nglist[0], ng2=nglist[1], flux=FLUX, ncut=10)


def run_tests(FQ_instance, FQ_string):
    flux_list = np.linspace(0.46, 0.54, 21)
    specdata_1 = FQ_instance.eigenvals(evals_count=12, filename='../scqubits/tests/data/'+FQ_string+'_1.hdf5')
    read_1 = qubit.read(filename='../scqubits/tests/data/'+FQ_string+'_1.hdf5')
    read_1.add_data(transfer_matrix=FQ_instance.transfer_matrix())
    sorted_minima_list = []
    minima_vals = FQV.sorted_minima_dict.values()
    for elem in minima_vals:
        sorted_minima_list.append(elem)
    read_1.add_data(sorted_minima=np.array(sorted_minima_list))
    read_1.add_data(gamma_matrix=FQ_instance.gamma_matrix())
    read_1.add_data(eigensystem_normal_modes=FQ_instance.eigensystem_normal_modes())
    read_1.add_data(Xi_matrix=FQ_instance.Xi_matrix())
    read_1.add_data(kinetic_matrix=FQ_instance.kinetic_matrix())
    read_1.add_data(potential_matrix=FQ_instance.potential_matrix())
    read_1.add_data(inner_product_matrix=FQ_instance.inner_product_matrix())
    qubit.write(read_1, filename='../scqubits/tests/data/'+FQ_string+'_1.hdf5')
    specdata_2 = FQ_instance.eigensys(evals_count=4, filename='../scqubits/tests/data/'+FQ_string+'_2.hdf5')
    specdata_4 = FQ_instance.get_spectrum_vs_paramvals(param_name='flux', param_vals=flux_list, evals_count=4, get_eigenstates=True,
                                               filename='../scqubits/tests/data/'+FQ_string+'_4.hdf5')
    # specdata_5 = FQ_instance.matrixelement_table('n_operator', operator_args={'j': 0},
    #                                      filename='../scqubits/tests/data/'+FQ_string+'_5.hdf5')

def run_tests_ED(FQ_instance):
    flux_list = np.linspace(0.46, 0.54, 21)
    specdata_1_FQ = FQ_instance.eigenvals(evals_count=12, filename='../scqubits/tests/data/fluxqubit_1.hdf5')
    specdata_2_FQ = FQ_instance.eigensys(evals_count=4, filename='../scqubits/tests/data/fluxqubit_2.hdf5')
    specdata_4_FQ = FQ_instance.get_spectrum_vs_paramvals(param_name='flux', param_vals=flux_list, evals_count=4,
                                                 get_eigenstates=True,
                                                 filename='../scqubits/tests/data/fluxqubit_4.hdf5')
    specdata_5_FQ = FQ_instance.matrixelement_table('n_1_operator', evals_count=10,
                                           filename='../scqubits/tests/data/fluxqubit_5.hdf5')


run_tests(FQV, 'fluxqubitvtb')
run_tests(FQVS, 'fluxqubitvtbsqueezing')