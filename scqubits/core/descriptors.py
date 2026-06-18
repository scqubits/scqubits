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

from __future__ import annotations

from typing import Any, Generic, Type, TypeVar

import numpy as np

from scqubits.core.central_dispatch import DispatchClient

TargetType = TypeVar("TargetType")


def _values_equal(new_value: Any, old_value: Any) -> bool:
    """Return whether two watched-property values are equal.

    Handles array-like values (where ``==`` yields an element-wise array) by
    reducing with :func:`numpy.all`, and treats any comparison that cannot be
    evaluated as "not equal" so the assignment is allowed to proceed.

    Parameters
    ----------
    new_value:
        value being assigned.
    old_value:
        value currently stored.
    """
    if new_value is old_value:
        return True
    try:
        result = new_value == old_value
    except (TypeError, ValueError):
        return False
    if isinstance(result, np.ndarray) or hasattr(result, "__iter__"):
        return bool(np.all(result))
    try:
        return bool(result)
    except (TypeError, ValueError):
        return False


class ReadOnlyProperty(Generic[TargetType]):
    """Descriptor for read-only properties (stored in ``xxx._name``).

    Parameters
    ----------
    target_type:
        type of the underlying property.
    """

    def __init__(self, target_type: Type[TargetType]):
        super().__init__()

    def __set_name__(self, owner: type, name: str) -> None:
        """Record the storage attribute name as ``_<name>``.

        Parameters
        ----------
        owner:
            class on which the descriptor is being assigned.
        name:
            attribute name under which the descriptor is being assigned.
        """
        self.name = f"_{name}"

    def __get__(self, instance: Any, *args, **kwargs) -> TargetType:
        """Return the value stored under ``self.name`` on ``instance``.

        When accessed on the class itself (``instance`` is ``None``), return
        the descriptor object.

        Parameters
        ----------
        instance:
            owning instance, or ``None`` if accessed on the class.
        """
        if instance is None:  # when accessed on class level rather than instance level
            return self  # type: ignore[return-value]
        return instance.__dict__[self.name]

    def __set__(self, instance: Any, value: Any):
        """Always raise :exc:`AttributeError`; this property is read-only.

        Parameters
        ----------
        instance:
            owning instance.
        value:
            value attempted to be assigned.
        """
        raise AttributeError("Property is for reading only, cannot assign to it.")


class WatchedProperty(Generic[TargetType]):
    """Descriptor for properties that are monitored for changes.

    On a value change, the owning instance invokes its ``broadcast()`` method
    to send the corresponding event notification to
    :class:`~scqubits.core.central_dispatch.CentralDispatch`.

    Parameters
    ----------
    target_type:
        type of the watched property.
    event:
        name of the event triggered when the property changes.
    inner_object_name:
        used, e.g., in ``FullZeroPi`` where an inner-object property is to be set.
    attr_name:
        custom attribute name to be used (default: name from the defining
        property in the instance class, obtained in ``__set_name__``).
    fget:
        optional custom getter callable.
    fset:
        optional custom setter callable.
    """

    def __init__(
        self,
        target_type: Type[TargetType],
        event: str,
        inner_object_name: str | None = None,
        attr_name: str | None = None,
        fget=None,
        fset=None,
    ) -> None:
        self.event = event
        self.inner = inner_object_name
        self.attr_name = attr_name
        self.setter = fset
        self.getter = fget

    def __set_name__(self, owner: type, name: str) -> None:
        """Record the descriptor's name and (if unset) its target attribute name.

        Parameters
        ----------
        owner:
            class on which the descriptor is being assigned.
        name:
            attribute name under which the descriptor is being assigned.
        """
        self.name = name
        self.attr_name = self.attr_name or name

    def __get__(self, instance: object, owner: Any) -> TargetType:
        """Return the watched value, dispatching through ``fget``/``inner`` as needed.

        When accessed on the class itself (``instance`` is ``None``), return
        the descriptor object.

        Parameters
        ----------
        instance:
            owning instance, or ``None`` if accessed on the class.
        owner:
            class on which the descriptor is defined.
        """
        if instance is None:  # when accessed on class level rather than instance level
            return self  # type: ignore[return-value]

        assert self.attr_name
        if self.inner:
            inner_instance = instance.__dict__[self.inner]
            return getattr(inner_instance, self.attr_name)
        if self.getter is None:
            return instance.__dict__[
                self.attr_name
            ]  # cannot use getattr, otherwise recursion
        else:
            return self.getter(instance)

    def __set__(self, instance: DispatchClient, value: TargetType) -> None:
        """Store the watched value and broadcast ``self.event`` when it changes.

        Dispatches through ``fset``/``inner`` if configured. The broadcast is
        suppressed on the first assignment (when neither ``attr_name`` nor
        ``_attr_name`` is yet present on the instance) and on any later
        reassignment whose value equals the one already stored; the value is
        still written on the first assignment regardless.

        Parameters
        ----------
        instance:
            owning instance.
        value:
            new value for the watched property.
        """
        if self.inner and self.attr_name:
            inner_instance = instance.__dict__[self.inner]
            setattr(inner_instance, self.attr_name, value)
            # Rely on inner_instance.attr_name to do the broadcasting.
        else:
            assert self.attr_name
            first_assignment = (
                self.attr_name not in instance.__dict__
                and f"_{self.attr_name}" not in instance.__dict__
            )
            # Always initialize on the first assignment; afterwards, write and
            # broadcast only when the value actually changes. The old value is
            # read through __get__ so it works for both plain and fget/fset
            # (``_attr_name``-backed) properties. The ``or`` short-circuits, so
            # __get__ is never called before the value exists.
            if first_assignment or not _values_equal(
                value, self.__get__(instance, type(instance))
            ):
                if self.setter is None:
                    instance.__dict__[self.attr_name] = value
                else:
                    self.setter(instance, value, name=self.attr_name)
                if not first_assignment:
                    instance.broadcast(self.event)
