# fitting.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################


import scqubits.utils.file_io_serializers as serializers


class FitData(serializers.Serializable):
    def __init__(self, datanames, datalist, fit_results=None):
        self.datanames = datanames
        self.datalist = datalist
        self.fit_results = fit_results
