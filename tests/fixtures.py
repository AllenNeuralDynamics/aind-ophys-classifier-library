"""Metadata and compute-result fixtures shared by the tests."""

import numpy as np


def predictions(*values: int) -> np.ndarray:
    """Return per-ROI predictions as the ONNX classifier emits them.

    Matches ROICaT's ONNX logistic-regression head: int64 labels of shape
    ``(n_rois,)``.

    Parameters
    ----------
    *values : int
        One prediction per ROI.

    Returns
    -------
    numpy.ndarray
        Labels of shape ``(n_rois,)``, dtype ``int64``.
    """
    return np.asarray(values, dtype=np.int64)


def border_flags(*values: bool) -> np.ndarray:
    """Return per-ROI border flags as ``classify_border_rois`` emits them.

    Parameters
    ----------
    *values : bool
        One flag per ROI.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(n_rois,)``.
    """
    return np.asarray(values, dtype=bool)


def probabilities(n_rois: int) -> np.ndarray:
    """Return the class probabilities the ONNX head returns with its labels.

    Parameters
    ----------
    n_rois : int
        Number of ROIs.

    Returns
    -------
    numpy.ndarray
        Probabilities of shape ``(n_rois, 2)``, dtype ``float32``; column 1 is
        the positive class the overlay figures index.
    """
    return np.zeros((n_rois, 2), dtype=np.float32)


def roi_masks(n_rois: int, height: int = 8, width: int = 8) -> np.ndarray:
    """Return dense ROI masks as ``load_extraction_rois`` returns them.

    One single-pixel ROI per row.

    Parameters
    ----------
    n_rois : int
        Number of ROIs.
    height : int, optional
        Field-of-view height in pixels.
    width : int, optional
        Field-of-view width in pixels.

    Returns
    -------
    numpy.ndarray
        Masks of shape ``(n_rois, height, width)``, dtype ``float32``.
    """
    masks = np.zeros((n_rois, height, width), dtype=np.float32)
    for index in range(n_rois):
        masks[index, index % height, index % width] = 1.0
    return masks
