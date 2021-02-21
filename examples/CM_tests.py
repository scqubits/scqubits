import numpy as np
import scqubits as qubit

qubit.settings.MULTIPROC = 'pathos'

N = 2
maximum_periodic_vector_length = 4
ECB = 0.2  # 100 fF
ECJ = 20.0/2.7  # 2.7 fF
ECg = 20.0  # 0.5 fF
flux = 0.0
EJ = 20.0
EJlist = np.array([EJ for j in range(2*N)])
EJlist[0: 3] = 1.01*EJlist[0: 3]
nglist = np.array([0.0 for _ in range(2*N - 1)])
num_exc = 2
flux_list = np.linspace(0.46, 0.54, 21)
CMV = qubit.CurrentMirrorVTB(N=N, ECB=ECB, ECJ=ECJ, ECg=ECg, EJlist=EJlist, nglist=nglist, flux=flux, num_exc=num_exc,
                             maximum_periodic_vector_length=maximum_periodic_vector_length, truncated_dim=6)
CMVG = qubit.CurrentMirrorVTBGlobal(N=N, ECB=ECB, ECJ=ECJ, ECg=ECg, EJlist=EJlist, nglist=nglist,
                                    flux=flux, num_exc=num_exc,
                                    maximum_periodic_vector_length=maximum_periodic_vector_length, truncated_dim=6)
CMVS = qubit.CurrentMirrorVTBSqueezing(N=N, ECB=ECB, ECJ=ECJ, ECg=ECg, EJlist=EJlist, nglist=nglist, flux=flux,
                                       num_exc=num_exc, maximum_periodic_vector_length=maximum_periodic_vector_length,
                                       truncated_dim=6)
CMVGS = qubit.CurrentMirrorVTBGlobalSqueezing(N=N, ECB=ECB, ECJ=ECJ, ECg=ECg, EJlist=EJlist, nglist=nglist,
                                              flux=flux, num_exc=num_exc,
                                              maximum_periodic_vector_length=maximum_periodic_vector_length,
                                              truncated_dim=6)

CM = qubit.CurrentMirror(N, ECB, ECJ, ECg, EJlist, nglist, flux, ncut=2,
                         truncated_dim=6)
CM_global = qubit.CurrentMirrorGlobal(N, ECB, ECJ, ECg, EJlist, nglist, flux, num_exc=7, truncated_dim=6)
evals_1 = CMV.eigenvals()
evals_2 = CMVG.eigenvals()
evals_3 = CMVS.eigenvals()
evals_4 = CMVGS.eigenvals()

def run_tests(CM_instance, CM_string):
    specdata_1 = CM_instance.eigenvals(evals_count=12, filename='../scqubits/tests/data/'+CM_string+'_1.hdf5')
    read_1 = qubit.read(filename='../scqubits/tests/data/'+CM_string+'_1.hdf5')
    read_1.add_data(transfer_matrix=CM_instance.transfer_matrix())
    sorted_minima_list = []
    minima_vals = CM_instance.sorted_minima_dict.values()
    for elem in minima_vals:
        sorted_minima_list.append(elem)
    read_1.add_data(sorted_minima=np.array(sorted_minima_list))
    read_1.add_data(gamma_matrix=CM_instance.gamma_matrix())
    read_1.add_data(eigensystem_normal_modes=CM_instance.eigensystem_normal_modes())
    read_1.add_data(Xi_matrix=CM_instance.Xi_matrix())
    read_1.add_data(kinetic_matrix=CM_instance.kinetic_matrix())
    read_1.add_data(potential_matrix=CM_instance.potential_matrix())
    read_1.add_data(inner_product_matrix=CM_instance.inner_product_matrix())
    qubit.write(read_1, filename='../scqubits/tests/data/'+CM_string+'_1.hdf5')
    specdata_2 = CM_instance.eigensys(evals_count=4, filename='../scqubits/tests/data/'+CM_string+'_2.hdf5')
    read_1 = qubit.read(filename='../scqubits/tests/data/'+CM_string+'_2.hdf5')
    specdata_4 = CM_instance.get_spectrum_vs_paramvals(param_name='flux', param_vals=flux_list, evals_count=4, get_eigenstates=True,
                                               filename='../scqubits/tests/data/'+CM_string+'_4.hdf5')
#    specdata_5 = CM_instance.matrixelement_table('n_operator', operator_args={'j': 0},
#                                         filename='../scqubits/tests/data/'+CM_string+'_5.hdf5')

def run_tests_ED(CM_instance):
    specdata_1_CM = CM_instance.eigenvals(evals_count=12, filename='../scqubits/tests/data/currentmirror_1.hdf5')
    specdata_2_CM = CM_instance.eigensys(evals_count=4, filename='../scqubits/tests/data/currentmirror_2.hdf5')
    specdata_4_CM = CM_instance.get_spectrum_vs_paramvals(param_name='flux', param_vals=flux_list, evals_count=4, get_eigenstates=True,
                                                 filename='../scqubits/tests/data/currentmirror_4.hdf5')
    matelem_CM = CM_instance.matrixelement_table('n_operator', evals_count=10,
                                                 filename='../scqubits/tests/data/currentmirror_5.hdf5')

#run_tests(CMV, 'currentmirrorvtbs')
#run_tests(CMVS, 'currentmirrorvtbsqueezing')
run_tests_ED(CM)