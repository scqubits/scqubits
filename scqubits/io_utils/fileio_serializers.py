# fileio_serializers.py
#
# This file is part of scqubits: a Python package for superconducting qubits,
# Quantum 5, 583 (2021). https://quantum-journal.org/papers/q-2021-11-17-583/
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################
"""
Helper classes for writing data to files.
"""

import inspect

from abc import ABCMeta
from collections import OrderedDict
from numbers import Number
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, Union

import numpy as np

from numpy import ndarray
from scipy.sparse import csc_matrix
from sympy import Expr
from typing_extensions import Protocol, runtime_checkable

import scqubits.utils.misc as utils

if TYPE_CHECKING:
    from scqubits.io_utils.fileio import IOData


SERIALIZABLE_REGISTRY = {}


@runtime_checkable
class Serializable(Protocol):
    """Mix-in class that makes descendant classes serializable."""

    _subclasses: List[ABCMeta] = []

    def __new__(cls: Any, *args, **kwargs) -> "Serializable":
        """Modified `__new__` to set up `cls._init_params`. The latter is used to
        record which of the `__init__` parameters are to be stored/read in file IO."""
        cls._init_params = get_init_params(cls)
        return super().__new__(cls)

    def __init_subclass__(cls) -> None:
        """Used to register all non-abstract _subclasses as a list in
        `QuantumSystem._subclasses`."""
        super().__init_subclass__()
        if not inspect.isabstract(cls):
            cls._subclasses.append(cls)
            SERIALIZABLE_REGISTRY[cls.__name__] = cls

    @classmethod
    def deserialize(cls, io_data: "IOData") -> "Serializable":
        """
        Take the given IOData and return an instance of the described class,
        initialized with the data stored in io_data.
        """
        return cls(**io_data.as_kwargs())

    def serialize(self) -> "IOData":
        """
        Convert the content of the current class instance into IOData format.
        """
        initdata = {name: getattr(self, name) for name in self._init_params}
        if hasattr(self, "_id_str"):
            initdata["id_str"] = self._id_str  # type:ignore
        iodata = dict_serialize(initdata)
        iodata.typename = type(self).__name__
        return iodata

    def filewrite(self, filename: str) -> None:
        """Convenience method bound to the class. Simply accesses the `write`
        function."""
        import scqubits.io_utils.fileio as io

        io.write(self, filename)

    @classmethod
    def create_from_file(cls, filename: str) -> object:
        """Read initdata and spectral data from file, and use those to create a new
        SpectrumData object.

        Returns
        -------
        SpectrumData
            new SpectrumData object, initialized with data read from file
        """
        import scqubits.io_utils.fileio as io

        return io.read(filename)


def _add_object(
    name: str,
    obj: object,
    attributes: Dict[str, Any],
    ndarrays: Dict[str, ndarray],
    objects: Dict[str, object],
) -> Tuple[Dict, Dict, Dict]:
    objects[name] = obj
    return attributes, ndarrays, objects


def _add_ndarray(
    name: str,
    obj: ndarray,
    attributes: Dict[str, Any],
    ndarrays: Dict[str, ndarray],
    objects: Dict[str, object],
) -> Tuple[Dict, Dict, Dict]:
    ndarrays[name] = obj
    return attributes, ndarrays, objects


def _add_attribute(
    name: str,
    obj: Any,
    attributes: Dict[str, Any],
    ndarrays: Dict[str, ndarray],
    objects: Dict[str, object],
) -> Tuple[Dict, Dict, Dict]:
    attributes[name] = obj
    return attributes, ndarrays, objects


def _add_boundmethod_attribute(
    name: str,
    obj: Any,
    attributes: Dict[str, Any],
    ndarrays: Dict[str, ndarray],
    objects: Dict[str, object],
) -> Tuple[Dict, Dict, Dict]:
    attributes[name] = obj()
    return attributes, ndarrays, objects


TO_ATTRIBUTE = (Expr, str, Number, dict, OrderedDict, list, tuple, bool, np.bool_)
TO_NDARRAY = (np.ndarray,)
TO_OBJECT = (Serializable,)


def type_dispatch(entity: Serializable) -> Callable:
    """
    Based on the type of the object ``entity``, return the appropriate function that
    converts the entity into the appropriate category of IOData
    """
    if isinstance(entity, TO_ATTRIBUTE):
        return _add_attribute
    if isinstance(entity, TO_OBJECT):
        return _add_object
    if isinstance(entity, TO_NDARRAY):
        if entity.dtype == "O":
            return _add_object
        return _add_ndarray
    if callable(entity) and "_operator" in entity.__name__:
        return _add_boundmethod_attribute
    # no match, try treating as object, though this may fail
    return _add_object


def Expr_serialize(expr_instance: Expr) -> "IOData":
    """
    Create an IODate instance for a sympy expression via string conversion
    """
    import scqubits.io_utils.fileio as io

    attributes: Dict[str, Any] = {}
    ndarrays: Dict[str, ndarray] = {}
    objects: Dict[str, object] = {}
    typename = "Expr"
    item = str(expr_instance)
    update_func = type_dispatch(item)
    attributes, ndarrays, objects = update_func(
        "Expr", item, attributes, ndarrays, objects
    )
    return io.IOData(typename, attributes, ndarrays, objects)


def dict_serialize(dict_instance: Dict[str, Any]) -> "IOData":
    """
    Create an IOData instance from dictionary data.
    """
    import scqubits.io_utils.fileio as io

    dict_instance = utils.remove_nones(dict_instance)
    attributes: Dict[str, Any] = {}
    ndarrays: Dict[str, ndarray] = {}
    objects: Dict[str, object] = {}
    typename = "dict"

    for name, content in dict_instance.items():
        update_func = type_dispatch(content)
        attributes, ndarrays, objects = update_func(
            name, content, attributes, ndarrays, objects
        )
    return io.IOData(typename, attributes, ndarrays, objects)


def OrderedDict_serialize(dict_instance: Dict[str, Any]) -> "IOData":
    """
    Create an IOData instance from dictionary data.
    """
    import scqubits.io_utils.fileio as io

    dict_instance = utils.remove_nones(dict_instance)

    attributes: Dict[str, Any] = {}
    ndarrays: Dict[str, ndarray] = {}
    objects: Dict[str, object] = {}
    typename = "OrderedDict"

    list_representation = list(dict_instance.items())
    for index, item in enumerate(list_representation):
        update_func = type_dispatch(item)
        attributes, ndarrays, objects = update_func(
            str(index), item, attributes, ndarrays, objects
        )
    return io.IOData(typename, attributes, ndarrays, objects)


def csc_matrix_serialize(csc_matrix_instance: csc_matrix) -> "IOData":
    """
    Create an IOData instance from dictionary data.
    """
    import scqubits.io_utils.fileio as io

    attributes: Dict[str, Any] = {}
    ndarrays: Dict[str, ndarray] = {}
    objects: Dict[str, object] = {}
    typename = "csc_matrix"

    csc_dict = {
        "indices": csc_matrix_instance.indices,
        "indptr": csc_matrix_instance.indptr,
        "shape": csc_matrix_instance.shape,
        "data": csc_matrix_instance.data,
    }

    for name, content in csc_dict.items():
        update_func = type_dispatch(content)
        attributes, ndarrays, objects = update_func(
            name, content, attributes, ndarrays, objects
        )
    return io.IOData(typename, attributes, ndarrays, objects)


def NoneType_serialize(none_instance: None) -> "IOData":
    """
    Create an IOData instance to write `None` to file.
    """
    import scqubits.io_utils.fileio as io

    attributes = {"None": 0}
    ndarrays: Dict[str, ndarray] = {}
    objects: Dict[str, object] = {}
    typename = "NoneType"

    return io.IOData(typename, attributes, ndarrays, objects)


def listlike_serialize(listlike_instance: Union[List, Tuple]) -> "IOData":
    """
    Create an IOData instance from list data.
    """
    import scqubits.io_utils.fileio as io

    attributes: Dict[str, Any] = {}
    ndarrays: Dict[str, ndarray] = {}
    objects: Dict[str, object] = {}
    typename = type(listlike_instance).__name__
    for index, item in enumerate(listlike_instance):
        update_func = type_dispatch(item)
        attributes, ndarrays, objects = update_func(
            str(index), item, attributes, ndarrays, objects
        )
    return io.IOData(typename, attributes, ndarrays, objects)


list_serialize = listlike_serialize


tuple_serialize = listlike_serialize


ndarray_serialize = listlike_serialize  # this is invoked for dtype=object


def range_serialize(range_instance: range) -> "IOData":
    """
    Create an IOData instance from range data.
    """
    import scqubits.io_utils.fileio as io

    attributes = {
        "start": range_instance.start,
        "stop": range_instance.stop,
        "step": range_instance.step,
    }
    ndarrays: Dict[str, ndarray] = {}
    objects: Dict[str, object] = {}
    typename = type(range_instance).__name__
    return io.IOData(typename, attributes, ndarrays, objects)


def Expr_deserialize(iodata: "IOData") -> Expr:
    """Turn IOData instance back into a dict"""
    from sympy import sympify

    return sympify(iodata["Expr"])


def dict_deserialize(iodata: "IOData") -> Dict[str, Any]:
    """Turn IOData instance back into a dict"""
    return dict(**iodata.as_kwargs())


def OrderedDict_deserialize(iodata: "IOData") -> Dict[str, Any]:
    """Turn IOData instance back into a dict"""
    dict_data = iodata.as_kwargs()
    return OrderedDict([dict_data[key] for key in sorted(dict_data, key=int)])


def csc_matrix_deserialize(iodata: "IOData") -> csc_matrix:
    """Turn IOData instance back into a csc_matrix"""
    csc_dict = dict(**iodata.as_kwargs())
    return csc_matrix(
        (csc_dict["data"], csc_dict["indices"], csc_dict["indptr"]),
        shape=csc_dict["shape"],
    )


def NoneType_deserialize(iodata: "IOData") -> None:
    """Turn IOData instance back into a csc_matrix"""
    return None


def list_deserialize(iodata: "IOData") -> List[Any]:
    """Turn IOData instance back into a list"""
    dict_data = iodata.as_kwargs()
    return [dict_data[key] for key in sorted(dict_data, key=int)]


def tuple_deserialize(iodata: "IOData") -> Tuple:
    """Turn IOData instance back into a tuple"""
    return tuple(list_deserialize(iodata))


# this is invoked for ndarrays with dtype=object
def ndarray_deserialize(iodata: "IOData") -> ndarray:
    return np.asarray(list_deserialize(iodata), dtype=object)


def range_deserialize(iodata: "IOData") -> range:
    arguments = iodata.as_kwargs()
    return range(arguments["start"], arguments["stop"], arguments["step"])


def get_init_params(obj: Serializable) -> List[str]:
    """
    Returns a list of the parameters entering the `__init__` method of the given
    object `obj`.
    """
    init_params = list(inspect.signature(obj.__init__).parameters.keys())  # type: ignore
    if "self" in init_params:
        init_params.remove("self")
    if "kwargs" in init_params:
        init_params.remove("kwargs")
    return init_params
