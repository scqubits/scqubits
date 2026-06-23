# test_plotting.py
# meant to be run with 'pytest'

import matplotlib

matplotlib.use("Agg")

import numpy as np

from scqubits.utils.plotting import data_vs_paramvals


class TestDataVsParamvals:
    def test_broadcast_single_column_ndarray_x(self):
        x = np.arange(5).reshape(-1, 1)
        y = np.column_stack([np.arange(5), np.arange(5) + 10])

        _, ax = data_vs_paramvals(x, y, label_list=["a", "b"])

        assert len(ax.get_lines()) == 2
        for line in ax.get_lines():
            np.testing.assert_array_equal(line.get_xdata(), np.arange(5))

        np.testing.assert_array_equal(ax.get_lines()[0].get_ydata(), np.arange(5))
        np.testing.assert_array_equal(
            ax.get_lines()[1].get_ydata(), np.arange(5) + 10
        )

    def test_broadcast_single_column_list_x(self):
        x = np.linspace(0, 1, 5)
        y = np.column_stack([np.arange(5), np.arange(5) + 10])

        _, ax = data_vs_paramvals(x, y, label_list=["a", "b"])

        assert len(ax.get_lines()) == 2
        for line in ax.get_lines():
            np.testing.assert_array_equal(line.get_xdata(), x)
