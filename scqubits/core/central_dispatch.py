# central_dispatch.py
#
# This file is part of scqubits.
#
#    Copyright (c) 2019, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################


import weakref

import scqubits.settings as settings

EVENTS = [
    'QUANTUMSYSTEM_UPDATE',
    'GRID_UPDATE',
    'INTERACTIONTERM_UPDATE',
    'INTERACTIONLIST_UPDATE',
    'HILBERTSPACE_UPDATE',
    'PARAMETERSWEEP_UPDATE'
]


class CentralDispatch:
    clients_dict = {event: weakref.WeakKeyDictionary() for event in EVENTS}    # central dispatch information
    # For each event, store a dict that maps the clients registered for that event to their callback routines
    # The objects are keys in the inner dict, implemented as a WeakKeyDictionary to allow deletion/garbage collection
    # when object should expire

    @classmethod
    def get_clients_dict(cls, event):
        """For given `event`, return the dict mapping each registered client to their callback routine

        Parameters
        ----------
        event: str
            event name from EVENTS

        Returns
        -------
        dict
        """
        return cls.clients_dict[event]

    @classmethod
    def register(cls, event, who, callback=None):
        """
        Register object `who` for event `event`. (This modifies `clients_dict`.)

        Parameters
        ----------
        event: str
            event name from EVENTS
        who: DispatchClient
            object to be registered
        callback: method, optional
            custom callback method other than `.receive()`
        """
        if callback is None:
            callback = getattr(who, 'receive')
        cls.get_clients_dict(event)[who] = callback

    @classmethod
    def unregister(cls, event, who):
        """Unregister object `who` from event `event`.  (This modifies `clients_dict`.)

        Parameters
        ----------
        event: str
            event name from EVENTS
        who: DispatchClient
            object to be unregistered
        """
        del cls.get_clients_dict(event)[who]

    @classmethod
    def unregister_object(cls, who):
        """Unregister object `who` from all events.  (This modifies `clients_dict`.)

          Parameters
          ----------
          who: DispatchClient
              object to be unregistered
          """
        for event in cls.clients_dict:
            cls.get_clients_dict(event).pop(who, None)

    @classmethod
    def _dispatch(cls, event, sender, **kwargs):
        """Issue a dispatch for `event` coming from `sender.

        Parameters
        ----------
        event: str
            event name from EVENTS
        sender: DispatchClient
            object requesting the dispatch
        **kwargs
        """
        for client, callback in cls.get_clients_dict(event).items():
            callback(event, sender=sender, **kwargs)

    @classmethod
    def listen(cls, caller, event, **kwargs):
        """Receive message from client `caller` for event `event`. If dispatch is globally enabled, trigger a dispatch
        to all clients registered for event.

        Parameters
        ----------
        caller: DispatchClient
            object requesting the dispatch
        event:  str
            event name from EVENTS
        **kwargs
        """
        if settings.DISPATCH_ENABLED:
            cls._dispatch(event, sender=caller, **kwargs)


class DispatchClient:
    """Base class inherited by objects participating in central dispatch."""
    def broadcast(self, event, **kwargs):
        """Request a broadcast from CENTRAL_DISPATCH reporting `event`.

        Parameters
        ----------
        event:  str
            event name from EVENTS
        **kwargs
        """
        CENTRAL_DISPATCH.listen(self, event, **kwargs)

    def receive(self, event, sender, **kwargs):
        """Receive a message from CENTRAL_DISPATCH and initiate action on it.

        Parameters
        ----------
        event:  str
            event name from EVENTS
        sender: DispatchClient
            original sender reporting the event
        **kwargs
        """
        pass

    def __del__(self):
        CENTRAL_DISPATCH.unregister_object(self)


class ReadOnlyProperty:
    """Descriptor for read-only properties (stored in xxx._name)"""
    def __set_name__(self, owner, name):
        self.name = '_' + name

    def __get__(self, instance, owner):
        if instance is None:   # when accessed on class level rather than instance level
            return self
        else:
            return instance.__dict__[self.name]

    def __set__(self, instance, value):
        raise AttributeError


class WatchedProperty:
    """Descriptor class for properties that are to be monitored for changes. Upon change of the value, the instance
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
        else:
            if self.attr_name not in instance.__dict__:
                instance.__dict__[self.attr_name] = value
            else:
                instance.__dict__[self.attr_name] = value
                instance.broadcast(self.event)


# Start global instance of CentralDispatch()
CENTRAL_DISPATCH = CentralDispatch()
