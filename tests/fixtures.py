"""Minimal v1 and v2 metadata fixtures shared by the tests.

Deliberately minimal: only the fields the getters actually read, so the tests
prove the library tolerates messy-but-sufficient metadata rather than
requiring a fully-validating document. v2 discriminator strings are resolved
from the schema classes, matching how the read layer works.

Lives in ``tests/`` because ``[tool.coverage.run] source`` includes ``tests``;
keep it exercised by the tests that import it.
"""

from aind_pophys_metadata.io import object_type_value

UM_PER_PIXEL = 1.5
SUBJECT_ID = "783992"
DATASET_NAME = "multiplane-ophys_783992_2026-07-30_12-00-00"


def v1_session(scale_factor=UM_PER_PIXEL, unit="um/pixel") -> dict:
    """Return a minimal v1 ``session.json`` dict.

    Parameters
    ----------
    scale_factor : float or None, optional
        ``fov_scale_factor``; pass ``None`` to omit it entirely.
    unit : str or None, optional
        ``fov_scale_factor_unit``; pass ``None`` to omit it.

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


def v2_acquisition(scale=UM_PER_PIXEL, axis_unit="micrometer") -> dict:
    """Return a minimal v2 ``acquisition.json`` dict.

    Parameters
    ----------
    scale : float or None, optional
        Magnitude of the ``Scale`` entry in
        ``image_to_acquisition_transform``; ``None`` omits the entry, leaving
        only a ``Translation`` (which is what the canonical multiplane
        example emits, and the case that must fail loudly).
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


def data_description(name=DATASET_NAME) -> dict:
    """Return a minimal ``data_description.json`` dict."""
    return {"name": name}
