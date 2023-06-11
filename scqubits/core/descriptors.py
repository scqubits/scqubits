# descriptors.py
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

# Recap on descriptors: see https://realpython.com/python-descriptors/

from typing import Any, Generic, Optional, Type, TypeVar

from scqubits.core.central_dispatch import DispatchClient

TargetType = TypeVar("TargetType")


class ReadOnlyProperty(Generic[TargetType]):
    """
    Descriptor for read-only properties (stored in xxx._name)
    """

    def __init__(self, target_type: Type[TargetType]):
        super().__init__()

    def __set_name__(self, owner, name: str):
        self.name = f"_{name}"

    def __get__(self, instance: Any, *args, **kwargs) -> TargetType:
        if instance is None:  # when accessed on class level rather than instance level
            return self  # type:ignore
        return instance.__dict__[self.name]

    def __set__(self, instance: Any, value: Any):
        raise AttributeError("Property is for reading only, cannot assign to it.")


class WatchedProperty(Generic[TargetType]):
    """
    Descriptor class for properties that are to be monitored for changes. Upon change
    of the value, the instance class invokes its `broadcast()` method to send the
    appropriate event notification to CentralDispatch

    Parameters
    ----------
    target_type:
        type of watched property
    event:
        name of event to be triggered when property is changed
    inner_object_name:
        Used, e.g., in FulLZeroPi where an inner-object property is to be set.
    attr_name:
        custom attribute name to be used (default: name from defining property in
        instance class, obtained in __set_name__
    """

    def __init__(
        self,
        target_type: Type[TargetType],
        event: str,
        inner_object_name: Optional[str] = None,
        attr_name: Optional[str] = None,
    ) -> None:
        self.event = event
        self.inner = inner_object_name
        self.attr_name = attr_name

    def __set_name__(self, owner, name: str) -> None:
        self.name = name
        self.attr_name = self.attr_name or name

    def __get__(self, instance: object, owner: Any) -> TargetType:
        if instance is None:  # when accessed on class level rather than instance level
            return self  # type:ignore

        assert self.attr_name
        if self.inner:
            inner_instance = instance.__dict__[self.inner]
            return getattr(inner_instance, self.attr_name)
        return instance.__dict__[self.attr_name]

    def __set__(self, instance: DispatchClient, value: TargetType) -> None:
        if self.inner and self.attr_name:
            inner_instance = instance.__dict__[self.inner]
            setattr(inner_instance, self.attr_name, value)
            # Rely on inner_instance.attr_name to do the broadcasting.
        else:
            assert self.attr_name
            if self.attr_name not in instance.__dict__:
                instance.__dict__[self.attr_name] = value
                # Rely on inner_instance.attr_name to do the broadcasting.
            else:
                instance.__dict__[self.attr_name] = value
                instance.broadcast(self.event)
