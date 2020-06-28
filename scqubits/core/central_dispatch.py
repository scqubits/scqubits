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


import logging
import warnings
import weakref

import scqubits.settings as settings

# To enable logging output, uncomment the following setting:
# logging.basicConfig(level=logging.DEBUG)

EVENTS = [
    'QUANTUMSYSTEM_UPDATE',
    'GRID_UPDATE',
    'INTERACTIONTERM_UPDATE',
    'INTERACTIONLIST_UPDATE',
    'HILBERTSPACE_UPDATE',
    'PARAMETERSWEEP_UPDATE'
]


class CentralDispatch:
    """
    Primary class managing the central dispatch system.
    """
    def __init__(self):
        self.clients_dict = {event: weakref.WeakKeyDictionary() for event in EVENTS}    # central dispatch information
    # For each event, store a dict that maps the clients registered for that event to their callback routines
    # The objects are keys in the inner dict, implemented as a WeakKeyDictionary to allow deletion/garbage collection
    # when object should expire. Callback methods are stored as weakref.WeakMethod for the same reason.

    def get_clients_dict(self, event):
        """For given `event`, return the dict mapping each registered client to their callback routine

        Parameters
        ----------
        event: str
            event name from EVENTS

        Returns
        -------
        dict
        """
        return self.clients_dict[event]

    def register(self, event, who, callback=None):
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
        logging.debug("Registering {} for {}. welcome.".format(type(who).__name__, event))
        if callback is None:
            callback_ref = getattr(who, 'receive')
            # For purposes of garbage collection, this should preferably be:
            # callback_ref = weakref.WeakMethod(getattr(who, 'receive'))
            # However, as of 06/12/20, pathos balks on this on Windows (while Linux is passing).
            # Note that reference to callback methods is likely to prevent proper garbage collection,
            # so may have to revisit this issue if necessary.
        else:
            callback_ref = callback
            # For purposes of garbage collection, this should preferably be:
            # callback_ref = weakref.WeakMethod(callback)
            # However, as of 06/12/20, pathos balks on this on Windows (while Linux is passing).
            # Note that the reference to callback methods is likely to prevent proper garbage collection,
            # so may have to revisit this issue if necessary.
        self.get_clients_dict(event)[who] = callback_ref

    def unregister(self, event, who):
        """Unregister object `who` from event `event`.  (This modifies `clients_dict`.)

        Parameters
        ----------
        event: str
            event name from EVENTS
        who: DispatchClient
            object to be unregistered
        """
        del self.get_clients_dict(event)[who]

    def unregister_object(self, who):
        """Unregister object `who` from all events.  (This modifies `clients_dict`.)

          Parameters
          ----------
          who: DispatchClient
              object to be unregistered
          """
        for event in self.clients_dict:
            self.get_clients_dict(event).pop(who, None)

    def _dispatch(self, event, sender, **kwargs):
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
            logging.debug("Central dispatch calling {} about {}.".format(type(client).__name__, event))
            callback_ref(event, sender=sender, **kwargs)
            # When using WeakMethod references, this should rather be:
            # callback_ref()(event, sender=sender, **kwargs)

    def listen(self, caller, event, **kwargs):
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
            self._dispatch(event, sender=caller, **kwargs)


# Start global instance of CentralDispatch()
CENTRAL_DISPATCH = CentralDispatch()


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
        logging.debug("Client {} broadcasting {}".format(type(self).__name__, event))
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
        warnings.warn("`receive() method not implemented for {}".format(self))

    def __del__(self):
        # Garbage collection will invoke this at undetermined time. `if` clauses below prevent exceptions upon program
        # exit. (`logging` and `CENTRAL_DISPATCH` may have already been removed.)
        if logging:
            logging.debug("Unregistering {}. au revoir.".format(type(self).__name__))
        if CENTRAL_DISPATCH:
            CENTRAL_DISPATCH.unregister_object(self)
