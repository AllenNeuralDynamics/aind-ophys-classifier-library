"""Tests for the ROI overlay figures."""

import unittest

import matplotlib.pyplot as plt
import numpy as np

from aind_ophys_classifier_library import plots
from tests.fixtures import border_flags, predictions, probabilities, roi_masks

N_ROIS = 3


class TestOverlayFigures(unittest.TestCase):
    """Each builder returns a figure the caller owns."""

    def setUp(self) -> None:
        """Build one plane's classification outputs."""
        self.rois = roi_masks(N_ROIS)
        self.addCleanup(plt.close, "all")

    def test_probabilities_overlay(self) -> None:
        """The positive-class column drives the overlay weights."""
        figure = plots.plot_probabilities(self.rois, probabilities(N_ROIS), "soma probabilities")
        self.assertEqual(figure.axes[0].get_title(), "soma probabilities")

    def test_predictions_overlay(self) -> None:
        """Per-ROI labels broadcast across each ROI footprint."""
        figure = plots.plot_predictions(self.rois, predictions(1, 0, 1), "soma predictions")
        self.assertEqual(figure.axes[0].get_title(), "soma predictions")

    def test_border_overlay(self) -> None:
        """Boolean border flags broadcast the same way."""
        figure = plots.plot_border_rois(self.rois, border_flags(False, True, False), "border ROIs")
        self.assertEqual(figure.axes[0].get_title(), "border ROIs")

    def test_probabilities_are_two_column(self) -> None:
        """A one-dimensional probability array would not carry a class axis."""
        values = probabilities(N_ROIS)
        self.assertEqual(values.shape, (N_ROIS, 2))
        self.assertTrue(np.issubdtype(values.dtype, np.floating))


if __name__ == "__main__":
    unittest.main()
