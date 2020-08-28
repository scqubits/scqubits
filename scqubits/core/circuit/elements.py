import numpy as np
import sympy
from scipy.sparse.linalg import *
from abc import ABCMeta
from abc import abstractmethod


class CircuitElement:
    """
    Abstract class for circuit elements.
    All circuit elements defined in the QCircuit library derive from this base class.
    """

    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def is_external(self):
        pass

    @abstractmethod
    def is_phase(self):
        pass

    @abstractmethod
    def is_charge(self):
        pass

    @abstractmethod
    def energy_term(self, node_phases, node_charges):
        return None

    @abstractmethod
    def symbolic_energy_term(self, node_phases, node_charges):
        return None


class ExternalFlux(CircuitElement):
    """
    Circuit element representing a capacitor.
    """

    def __init__(self, name, flux_value=0):
        self.name = name
        self.flux_value = flux_value

    def set_flux_value(self, flux_value):
        self.flux_value = flux_value

    def get_flux_value(self):
        return self.flux_value

    def is_external(self):
        return True

    def is_phase(self):
        return True

    def is_charge(self):
        return False

    def energy_term(self, node_phases, node_charges):
        return None

    def symbolic_energy_term(self, node_phases, node_charges):
        return None


class ExternalCharge(CircuitElement):
    """
    Circuit element representing a capacitor.
    """

    def __init__(self, name, charge_value=0):
        self.name = name
        self.charge_value = charge_value

    def set_charge_value(self, charge_value):
        self.charge_value = charge_value

    def get_charge_value(self):
        return self.flux_value

    def is_external(self):
        return True

    def is_phase(self):
        return False

    def is_charge(self):
        return True

    def energy_term(self, node_phases, node_charges):
        return None

    def symbolic_energy_term(self, node_phases, node_charges):
        return None


class Capacitance(CircuitElement):
    """
    Circuit element representing a capacitor.
    """

    def __init__(self, name, capacitance=0):
        super().__init__(name)
        self.capacitance = capacitance

    def set_capacitance(self, capacitance):
        self.capacitance = capacitance

    def get_capacitance(self):
        return self.capacitance

    def is_external(self):
        return False

    def is_phase(self):
        return False

    def is_charge(self):
        return True

    def energy_term(self, node_phases, node_charges):
        return None

    def symbolic_energy_term(self, node_phases, node_charges):
        return None


class JosephsonJunction(CircuitElement):
    """
    Circuit element representing a Josephson junction.
    """
    def __init__(self, name, critical_current=0, use_offset=True):
        super().__init__(name)
        self.critical_current = critical_current
        self.use_offset = use_offset

    def set_critical_current(self, critical_current):
        self.critical_current = critical_current

    def get_critical_current(self):
        return self.critical_current

    def energy_term(self, node_phases, node_charges):
        if len(node_phases) != 2:
            raise Exception('ConnectionError',
                            'Josephson junction {0} has {1} nodes connected instead of 2.'.format(self.name, len(node_phases)))
        if self.use_offset:
            return self.critical_current * (1 - np.cos(node_phases[0] - node_phases[1]))
        else:
            return self.critical_current * (-np.cos(node_phases[0] - node_phases[1]))

    def symbolic_energy_term(self, node_phases, node_charges):
        if len(node_phases) != 2:
            raise Exception('ConnectionError',
                            'Josephson junction {0} has {1} nodes connected instead of 2.'.format(self.name, len(node_phases)))
        if self.use_offset:
            return self.critical_current * (1 - sympy.cos(node_phases[0] - node_phases[1]))
        else:
            return self.critical_current * (-sympy.cos(node_phases[0] - node_phases[1]))

    def is_phase(self):
        return True

    def is_charge(self):
        return False


class Inductance(CircuitElement):
    """
    Circuit element representing a linear inductor.
    """

    def __init__(self, name, inductance=0):
        super().__init__(name)
        self.inductance = inductance

    def set_inductance(self, inductance):
        self.inductance = inductance

    def get_inductance(self):
        return self.inductance

    def energy_term(self, node_phases, node_charges):
        if len(node_phases) != 2:
            raise Exception('ConnectionError',
                            'Inductance {0} has {1} nodes connected instead of 2.'.format(self.name, len(node_phases)))
        return (node_phases[0] - node_phases[1]) ** 2 / (2 * self.inductance)

    def symbolic_energy_term(self, node_phases, node_charges):
        if len(node_phases) != 2:
            raise Exception('ConnectionError',
                            'Inductance {0} has {1} nodes connected instead of 2.'.format(self.name, len(node_phases)))
        return (node_phases[0] - node_phases[1]) ** 2 / (2 * self.inductance)

    def is_external(self):
        return False

    def is_phase(self):
        return True

    def is_charge(self):
        return False


class LagrangianCurrentSource(CircuitElement):
    """
    Circuit element representing a Josephson junction.
    """

    def __init__(self, name, current=0):
        super().__init__(name)
        self.current = current

    def set_current(self, current):
        self.current = current

    def get_current(self):
        return self.current

    def energy_term(self, node_phases, node_charges):
        if len(node_phases) != 2:
            raise Exception('ConnectionError',
                            'Lagrangian current source {0} has {1} nodes connected instead of 2.'.format(self.name, len(
                                node_phases)))
        return self.current * (node_phases[0] - node_phases[1])

    def symbolic_energy_term(self, node_phases, node_charges):
        return self.energy_term(node_phases, node_charges)

    def is_external(self):
        return False

    def is_phase(self):
        return True

    def is_charge(self):
        return False
