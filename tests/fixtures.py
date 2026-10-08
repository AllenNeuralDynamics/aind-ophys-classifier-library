"""Metadata and compute-result fixtures shared by the tests."""

from typing import Optional

import numpy as np
from aind_pophys_metadata.io import object_type_value

UM_PER_PIXEL = 1.5
SUBJECT_ID = "783992"
DATASET_NAME = "multiplane-ophys_783992_2026-07-30_12-00-00"


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


def v1_session(
    scale_factor: Optional[float] = UM_PER_PIXEL,
    unit: Optional[str] = "micrometer",
) -> dict:
    """Return a minimal v1 ``session.json`` dict.

    Parameters
    ----------
    scale_factor : float or None, optional
        ``fov_scale_factor``; pass ``None`` to omit it entirely.
    unit : str or None, optional
        ``fov_scale_factor_unit``; pass ``None`` to omit it, which the shared
        normalizer reads as micrometres.

    Returns
    -------
    dict
        The raw session dict.
    """
    fov = {"index": 0, "targeted_structure": {"acronym": "VISp"}}
    if scale_factor is not None:
        fov["fov_scale_factor"] = scale_factor
    if unit is not None:
        fov["fov_scale_factor_unit"] = unit
    return {
        "rig_id": "MESO.1",
        "subject_id": SUBJECT_ID,
        "data_streams": [{"ophys_fovs": [fov]}],
    }


def v2_acquisition(
    scale: Optional[float] = UM_PER_PIXEL,
    axis_unit: Optional[str] = "micrometer",
) -> dict:
    """Return a minimal v2 ``acquisition.json`` dict.

    Parameters
    ----------
    scale : float or None, optional
        Magnitude of the ``Scale`` entry in
        ``image_to_acquisition_transform``; ``None`` omits the entry, leaving
        only a ``Translation``.
    axis_unit : str or None, optional
        ``coordinate_system.axis_unit``.

    Returns
    -------
    dict
        The raw acquisition dict.
    """
    from aind_data_schema.components.configs import ImagingConfig, PlanarImage
    from aind_data_schema.components.coordinates import Scale, Translation

    transforms = [
        {
            "object_type": object_type_value(Translation),
            "translation": [1.5, 1.5],
        }
    ]
    if scale is not None:
        transforms.append({"object_type": object_type_value(Scale), "scale": [scale, scale]})

    config = {
        "object_type": object_type_value(ImagingConfig),
        "sampling_strategy": {"frame_rate": 9.48},
        "images": [
            {
                "object_type": object_type_value(PlanarImage),
                "image_to_acquisition_transform": transforms,
                # The pixel dimensions are ALSO a Scale. Reading this instead
                # of the transform would silently yield 512 um/pixel.
                "dimensions": {
                    "object_type": object_type_value(Scale),
                    "scale": [512, 512],
                },
                "planes": [
                    {
                        "plane_index": 0,
                        "targeted_structure": {"acronym": "VISp"},
                    }
                ],
            }
        ],
    }
    if axis_unit is not None:
        config["coordinate_system"] = {"axis_unit": axis_unit}

    return {
        "instrument_id": "MESO.1",
        "subject_id": SUBJECT_ID,
        "data_streams": [{"configurations": [config]}],
    }


def minimal_metadata(um_per_pixel: Optional[float] = UM_PER_PIXEL) -> dict:
    """Return a flat minimal ``metadata.json`` dict.

    Parameters
    ----------
    um_per_pixel : float or None, optional
        Micrometres per pixel on the single plane; ``None`` omits it.

    Returns
    -------
    dict
        The raw minimal metadata dict.
    """
    plane = {"plane_index": 0, "structure": "VISp"}
    if um_per_pixel is not None:
        plane["um_per_pixel"] = um_per_pixel
    return {
        "frame_rate": 9.48,
        "instrument_id": "MESO.1",
        "subject_id": SUBJECT_ID,
        "planes": [plane],
    }


def data_description(name: str = DATASET_NAME) -> dict:
    """Return a minimal ``data_description.json`` dict."""
    return {"name": name}
