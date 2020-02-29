# cpu_switch.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import scqubits.settings as settings


def get_map_method(num_cpus):
    """
    Selects the correct `.map` method depending on the specified number of desired cores. If num_cpus>1, the
    multiprocessing/pathos pool is started here.

    Parameters
    ----------
    num_cpus: int

    Returns
    -------
    function
        `.map` method to be used by caller
    """
    if num_cpus == 1:
        return map

    # num_cpus > 1 -----------------

    # windows may require special treatment
    # if sys.platform == 'win32' and settings.POOL is None:
    #     warnings.warn("Windows users may explicitly need to  provide scqubits.settings.POOL.")

    # user is asking for more than 1 cpu; start pool from here
    if settings.MULTIPROC == 'pathos':
        try:
            import pathos
            import dill
        except ImportError:
            raise ImportError("scqubits multiprocessing mode set to 'pathos'. Need but cannot find 'pathos'/'dill'!")
        else:
            dill.settings['recurse'] = True
            settings.POOL = pathos.pools.ProcessPool(nodes=num_cpus)
            return settings.POOL.map
    if settings.MULTIPROC == 'multiprocessing':
        import multiprocessing
        settings.POOL = multiprocessing.Pool(processes=num_cpus)
        return settings.POOL.map
    else:
        raise ValueError("Unknown multiprocessing type: settings.MULTIPROC = {}".format(settings.MULTIPROC))
