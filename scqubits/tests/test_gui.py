# test_gui.py
# meant to be run with 'pytest'
#
# This file is part of scqubits.
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

from scqubits import GUI


class TestGUI:
    def test_gui(self):
        GUI()
