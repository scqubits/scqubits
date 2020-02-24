# file_io_base.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

from abc import ABC, abstractmethod


class IOWriter(ABC):
    """
    ABC for writing class instance data to file.

    Parameters
    ----------
    filename: str
    file_handle: h5.Group, optional
    """
    def __init__(self, filename, file_handle=None):
        self.filename = filename
        self.io_data = None
        self.file_handle = file_handle

    @abstractmethod
    def to_file(self, io_data, **kwargs):
        pass

    @abstractmethod
    def write_attributes(self, *args, **kwargs):
        pass

    @abstractmethod
    def write_ndarrays(self, *args, **kwargs):
        pass

    @abstractmethod
    def write_objects(self, *args, **kwargs):
        pass
