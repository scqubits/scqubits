import scqubits.core.discretization as discretization
import numpy as np


class Variable(discretization.Grid1d):
    """
    Represents a variable of the circuit wavefunction or an constant external bias flux or voltage.
    """

    def __init__(self, name):
        self.variable_type = 'parameter'
        super().__init__(0, 0, 1)
        self.name = name
        self.offset_charge = 0

    def set_variable(self, pt_count, periods, center=0):
        """Creates a discrete grid for phase wavefunction variables.

        Parameters
        ----------
        pt_count: int
            number of grid points
        periods: float
            number of 2pi intervals in the grid
        center: float
            phase grid centering
        """
        self.variable_type = 'variable'
        self.pt_count = pt_count

        self.min_val = -np.pi * periods + center
        self.max_val = self.min_val + 2 * np.pi * periods * (self.pt_count - 1) / self.pt_count
        self.offset_charge = 0  # set offset charge in variable to 0

    def set_parameter(self, phase_value, voltage_value):
        """Makes the parameter an external flux and/or charge bias.

        Parameters
        ----------
        phase_value: float
            external flux bias in flux quanta/(2pi)
        voltage_value: float
            external voltage bias
        """
        self.variable_type = 'parameter'
        self.pt_count = 1

        self.min_val = phase_value
        self.max_val = phase_value
        self.offset_charge = voltage_value

    def get_phase_grid(self) -> np.ndarray:
        """Returns a numpy array of the grid points in phase representation

        Returns
        -------
        ndarray
        """
        return self.make_linspace()

    def get_charge_grid(self) -> np.ndarray:
        """Returns a numpy array of the grid points in cooper pair number representation

        Returns
        -------
        ndarray
        """
        if self.pt_count > 1:
            range = (self.max_val - self.min_val) * self.pt_count / (self.pt_count - 1)
            delta_n = 2*np.pi/range
            grid = np.arange(0, delta_n*self.pt_count, delta_n)
        else:
            grid = np.asarray([0.0])
        grid -= np.mean(grid)
        grid += self.offset_charge
        if self.pt_count % 2 == 0:
            grid -= 0.5
        return grid
