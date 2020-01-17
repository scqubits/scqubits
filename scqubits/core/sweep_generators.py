# sweep_generators.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import scqubits.core.sweep_observables as obs


def generate_chi_sweep(sweep):
    """Generate data for the AC Stark shift chi as a function of the sweep parameter"""
    osc_subsys_list = sweep.hilbertspace.osc_subsys_list
    qbt_subsys_list = sweep.hilbertspace.qbt_subsys_list

    for osc_index, osc_subsys in osc_subsys_list:
        for qbt_index, qubit_subsys in qbt_subsys_list:
            sweep.compute_custom_data_sweep('chi_osc{}_qbt{}'.format(osc_index, qbt_index), obs.dispersive_chi,
                                            qubit_subsys=qubit_subsys, osc_subsys=osc_subsys, chi_indices=(1, 0))


def generate_charge_matrixelem_sweep(sweep):
    """Generate data for the charge matrix elements as a function of the sweep parameter"""
    for qbt_index, subsys in sweep.hilbertspace.qbt_subsys_list:
        if type(subsys).__name__ in ['Transmon', 'Fluxonium']:
            sweep.compute_custom_data_sweep('n_op_qbt{}'.format(qbt_index), obs.qubit_matrixelement,
                                            qubit_subsys=subsys, qubit_operator=subsys.n_operator())
