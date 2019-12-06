# progressbar.py
#
# This file is part of sc_qubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE.md file in the root directory of this source tree.
############################################################################
"""The progressbar module helps display a simple, text-based progress bar.
The bar length is given by 'progress_in_percent'.
"""

import sys
import sc_qubits.settings as cfg


def initialize():
    """Set up use of text-based progress bar."""
    if cfg.progressbar_enabled:
        print("")
        update(0)
    return None

def update(progress_in_percent):
    """Updates display of simple, text-based progress bar. The bar length is given by `progress_in_percent`.

    Parameters
    ----------
    progress_in_percent: float
        bar length, given in percent
    """
    if cfg.progressbar_enabled:
        bar_max_length = 20  # total length of the progress bar
        status_string = ""

        progress_in_percent = float(progress_in_percent)
        if progress_in_percent < 0.0:
            progress_in_percent = 0.0
            status_string = "Halt...\r\n"
        if progress_in_percent >= 1.0:
            progress_in_percent = 1.0
            status_string = "Done.\r\n"
        bar_length = int(round(bar_max_length * progress_in_percent))
        progress_display_string = "\r[{0}] {1}% {2}".format("=" * bar_length + "." * (bar_max_length - bar_length),
                                                            round(progress_in_percent * 100), status_string)
        sys.stdout.write(progress_display_string)
        sys.stdout.flush()
    return None
