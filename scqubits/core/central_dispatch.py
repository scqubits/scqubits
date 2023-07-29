# central_dispatch.py
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


import logging
import warnings
import weakref
from types import MethodType
from typing import Optional
from weakref import WeakKeyDictionary

import scqubits.settings as settings

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------
# To enable logging output, uncomment the following setting:
# LOGGER.setLevel(logging.DEBUG)
# ---------------------------------------------------------------


EVENTS = [
    "QUANTUMSYSTEM_UPDATE",
    "GRID_UPDATE",
    "INTERACTIONTERM_UPDATE",
    "INTERACTIONLIST_UPDATE",
    "HILBERTSPACE_UPDATE",
    "PARAMETERSWEEP_UPDATE",
]


class CentralDispatch:
    """
    Primary class managing the central dispatch system.
    """

    def __init__(self):
        self.clients_dict = {
            event: weakref.WeakKeyDictionary() for event in EVENTS
        }  # central dispatch information

    # For each event, store a dict that maps the clients registered for that event to
    # their callback routines The objects are keys in the inner dict, implemented as
    # a WeakKeyDictionary to allow deletion/garbage collection when object should
    # expire. Callback methods are stored as weakref.WeakMethod for the same reason.

    def get_clients_dict(self, event: str) -> WeakKeyDictionary:
        """For given `event`, return the dict mapping each registered client to their
        callback routine

        Parameters
        ----------
        event: str
            event name from EVENTS

        Returns
        -------
        dict
        """
        return self.clients_dict[event]

    def register(
        self, event: str, who: "DispatchClient", callback: Optional[MethodType] = None
    ) -> None:
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
        LOGGER.debug(
            "Registering {} for {}. welcome.".format(type(who).__name__, event)
        )
        if callback is None:
            callback_ref = weakref.WeakMethod(getattr(who, "receive"))
            # For purposes of garbage collection, this should preferably be:
            # callback_ref = weakref.WeakMethod(getattr(who, 'receive'))
            # Intermittently, pathos has balked on pickling this. Workaround that
            # will likely prevent proper garbage collection:
            #
            # callback_ref = getattr(who, "receive")
        else:
            callback_ref = weakref.WeakMethod(callback)
            # See comment just above. Workaround if pathos fails to pickle:
            #
            # callback_ref = callback
        self.get_clients_dict(event)[who] = callback_ref

    def unregister(self, event: str, who: "DispatchClient") -> None:
        """Unregister object `who` from event `event`.  (This modifies `clients_dict`.)

        Parameters
        ----------
        event: str
            event name from EVENTS
        who: DispatchClient
            object to be unregistered
        """
        del self.get_clients_dict(event)[who]

    def unregister_object(self, who: "DispatchClient") -> None:
        """Unregister object `who` from all events.  (This modifies `clients_dict`.)

        Parameters
        ----------
        who: DispatchClient
            object to be unregistered
        """
        for event in self.clients_dict:
            self.get_clients_dict(event).pop(who, None)

    def _dispatch(self, event: str, sender: "DispatchClient", **kwargs) -> None:
        """Issue a dispatch for `event` coming from `sender.

        Parameters
        ----------
        event: str
            event name from EVENTS
        sender: DispatchClient
            object requesting the dispatch
        **kwargs
        """
        for client, callback_ref in self.get_clients_dict(event).items():
            LOGGER.debug(
                "Central dispatch calling {} about {}.".format(
                    type(client).__name__, event
                )
            )
            # Using WeakMethod references:
            callback_ref()(event, sender=sender, **kwargs)

            # Workaround if pickling fails, in conjunction with changes eliminating
            # weakrefs:
            # callback_ref(event, sender=sender, **kwargs)

    def listen(self, caller: "DispatchClient", event: str, **kwargs) -> None:
        """Receive message from client `caller` for event `event`. If dispatch is
        globally enabled, trigger a dispatch to all clients registered for event.

        Parameters
        ----------
        caller: DispatchClient
            object requesting the dispatch
        event:  str
            event name from EVENTS
        **kwargs
        """
        if settings.DISPATCH_ENABLED:
            self._dispatch(event, sender=caller, **kwargs)


# Start global instance of CentralDispatch()
CENTRAL_DISPATCH = CentralDispatch()


class DispatchClient:
    """Base class inherited by objects participating in central dispatch."""

    def broadcast(self, event: str, **kwargs) -> None:
        """Request a broadcast from CENTRAL_DISPATCH reporting `event`.

        Parameters
        ----------
        event:
            event name from EVENTS
        **kwargs
        """
        if settings.DISPATCH_ENABLED:
            LOGGER.debug("Client {} broadcasting {}".format(type(self).__name__, event))
        CENTRAL_DISPATCH.listen(self, event, **kwargs)

    def receive(self, event: str, sender: "DispatchClient", **kwargs) -> None:
        """Receive a message from CENTRAL_DISPATCH and initiate action on it.

        Parameters
        ----------
        event:
            event name from EVENTS
        sender:
            original sender reporting the event
        **kwargs
        """
        warnings.warn("`receive() method not implemented for {}".format(self))

    def __del__(self) -> None:
        # Garbage collection will invoke this at undetermined time. `if` clauses
        # below prevent exceptions upon program exit. (`logging` and
        # `CENTRAL_DISPATCH` may have already been removed.)
        if logging:
            LOGGER.debug("Unregistering {}. au revoir.".format(type(self).__name__))
        if CENTRAL_DISPATCH:
            CENTRAL_DISPATCH.unregister_object(self)
