"""ROI overlay figures for classification QC.

Each builder returns the ``Figure`` so the caller owns saving and closing it;
the v1 code left figures open, which leaks memory across a multi-plane run.
"""

from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

TRANSPARENT_TO_RED = LinearSegmentedColormap.from_list(
    "TransparentToRed", [(1, 0, 0, 0), (1, 0, 0, 1)]
)


def _overlay(
    rois: np.ndarray,
    weighted: np.ndarray,
    title: str,
    clim: Optional[List[float]] = None,
) -> Figure:
    """Draw an ROI footprint sum with a weighted red overlay.

    Parameters
    ----------
    rois : numpy.ndarray
        Dense ROI masks of shape ``(n_rois, height, width)``.
    weighted : numpy.ndarray
        Per-pixel overlay values, same shape as ``rois``.
    title : str
        Figure title.
    clim : list, optional
        Colour limits for the overlay.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure.
    """
    fig, ax = plt.subplots()
    ax.imshow(rois.sum(axis=0), cmap="gray")
    ax.imshow(
        weighted.sum(axis=0),
        cmap=TRANSPARENT_TO_RED,
        clim=clim,
        alpha=0.9,
    )
    ax.set_title(title)
    return fig


def plot_probabilities(
    rois: np.ndarray, probabilities: np.ndarray, title: str
) -> Figure:
    """Overlay per-ROI positive-class probabilities.

    Parameters
    ----------
    rois : numpy.ndarray
        Dense ROI masks of shape ``(n_rois, height, width)``.
    probabilities : numpy.ndarray
        Class probabilities of shape ``(n_rois, n_classes)``; column 1 is the
        positive class.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure.
    """
    weighted = (rois > 0) * probabilities[:, 1, np.newaxis, np.newaxis]
    return _overlay(rois, weighted, title, clim=[0, 1])


def plot_predictions(
    rois: np.ndarray, predictions: np.ndarray, title: str
) -> Figure:
    """Overlay per-ROI binary predictions.

    Parameters
    ----------
    rois : numpy.ndarray
        Dense ROI masks of shape ``(n_rois, height, width)``.
    predictions : numpy.ndarray
        Per-ROI predictions of shape ``(n_rois,)``.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure.
    """
    weighted = (rois > 0) * predictions[:, np.newaxis, np.newaxis]
    return _overlay(rois, weighted, title)


def plot_border_rois(
    rois: np.ndarray, border_rois: np.ndarray, title: str
) -> Figure:
    """Overlay the ROIs flagged as touching the field-of-view border.

    Parameters
    ----------
    rois : numpy.ndarray
        Dense ROI masks of shape ``(n_rois, height, width)``.
    border_rois : numpy.ndarray
        Boolean per-ROI border flags of shape ``(n_rois,)``.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure.
    """
    weighted = (rois > 0) * border_rois[:, np.newaxis, np.newaxis]
    return _overlay(rois, weighted, title)
