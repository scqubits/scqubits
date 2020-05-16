# descriptors.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

# Recap on descriptors: see https://realpython.com/python-descriptors/


class ReadOnlyProperty:
    """
    Descriptor for read-only properties (stored in xxx._name)
    """
    def __set_name__(self, owner, name):
        self.name = '_' + name

    def __get__(self, instance, owner):
        if instance is None:   # when accessed on class level rather than instance level
            return self
        else:
            return instance.__dict__[self.name]

    def __set__(self, instance, value):
        raise AttributeError('Property is for reading only, cannot assign to it.')


class WatchedProperty:
    """
    Descriptor class for properties that are to be monitored for changes. Upon change of the value, the instance
    class invokes its `broadcast()` method to send the appropriate event notification to CentralDispatch

    Parameters
    ----------
    event: str
        name of event to be triggered when property is changed
    inner_object_name: str, optional
        Used, e.g., in FulLZeroPi where an inner-object property is to be set.
    attr_name: str, optional
        custom attribute name to be used (default: name from defining property in instance class,
        obtained in __set_name__
    """
    def __init__(self, event, inner_object_name=None, attr_name=None):
        self.event = event
        self.inner = inner_object_name
        self.attr_name = attr_name

    def __set_name__(self, owner, name):
        self.name = name
        self.attr_name = self.attr_name or name

    def __get__(self, instance, owner):
        if instance is None:   # when accessed on class level rather than instance level
            return self
        else:
            if self.inner:
                inner_instance = instance.__dict__[self.inner]
                return getattr(inner_instance, self.attr_name)
            return instance.__dict__[self.attr_name]

    def __set__(self, instance, value):
        if self.inner:
            inner_instance = instance.__dict__[self.inner]
            setattr(inner_instance, self.attr_name, value)
            # Rely on inner_instance.attr_name to do the broadcasting.
        else:
            if self.attr_name not in instance.__dict__:
                instance.__dict__[self.attr_name] = value
                # Rely on inner_instance.attr_name to do the broadcasting.
            else:
                instance.__dict__[self.attr_name] = value
                instance.broadcast(self.event)
