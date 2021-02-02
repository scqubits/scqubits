import scqubits as qubit
import numpy as np

num_exc = 3
phi_grid = qubit.Grid1d(-9., 9., 100)
maximum_periodic_vector_length = 6
EJ = 10.0
EL = 0.1
ECJ = 20.0
EC = 0.04
ng = 0.0
flux = 0.0
phi_extent = 3
zero_pi = qubit.ZeroPi(grid=phi_grid, EJ=EJ, EL=EL, ECJ=ECJ, EC=EC, ng=ng, flux=flux, ncut=10)
ZPV = qubit.ZeroPiVTB(EJ=EJ, EL=EL, ECJ=ECJ, EC=EC, ng=ng, flux=flux,  num_exc=num_exc,
                      maximum_periodic_vector_length=maximum_periodic_vector_length, phi_extent=phi_extent,
                      quiet=True)

flux_list = np.linspace(0.46, 0.54, 15)

specdata_1 = ZPV.eigenvals(evals_count=12, filename='../scqubits/tests/data/zeropivtb_1.hdf5')
read_1 = qubit.read(filename='../scqubits/tests/data/zeropivtb_1.hdf5')
read_1.add_data(transfer_matrix=ZPV.transfer_matrix())
read_1.add_data(sorted_minima=ZPV.sorted_minima_dict())
read_1.add_data(gamma_matrix=ZPV.gamma_matrix())
read_1.add_data(eigensystem_normal_modes=ZPV.eigensystem_normal_modes())
read_1.add_data(Xi_matrix=ZPV.Xi_matrix())
read_1.add_data(nearest_neighbors=ZPV.nearest_neighbors)
read_1.add_data(kinetic_matrix=ZPV.kinetic_matrix())
read_1.add_data(potential_matrix=ZPV.potential_matrix())
read_1.add_data(inner_product_matrix=ZPV.inner_product_matrix())
qubit.write(read_1, filename='../scqubits/tests/data/zeropivtb_1.hdf5')
specdata_2 = ZPV.eigensys(evals_count=4, filename='../scqubits/tests/data/zeropivtb_2.hdf5')
specdata_4 = ZPV.get_spectrum_vs_paramvals(param_name='flux', param_vals=flux_list, evals_count=4, get_eigenstates=True,
                                           filename='../scqubits/tests/data/zeropivtb_4.hdf5')

specdata_5 = ZPV.matrixelement_table('n_operator', operator_args={'j': 1},
                                     filename='../scqubits/tests/data/zeropivtb_5.hdf5')

specdata_1_zero_pi = zero_pi.eigenvals(evals_count=12, filename='../scqubits/tests/data/zeropi_vtb_compare_1.hdf5')
specdata_2_zero_pi = zero_pi.eigensys(evals_count=4, filename='../scqubits/tests/data/zeropi_vtb_compare_2.hdf5')
specdata_4_zero_pi = zero_pi.get_spectrum_vs_paramvals(param_name='flux', param_vals=flux_list, evals_count=4,
                                                       get_eigenstates=True,
                                                       filename='../scqubits/tests/data/zeropi_vtb_compare_4.hdf5')
