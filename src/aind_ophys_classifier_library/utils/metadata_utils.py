"""aind-data-schema v1/v2 metadata reads for ophys ROI classification.

A thin, capsule-specific layer over the shared ``aind_pophys_metadata``
library. The shared library owns schema-version dispatch, the identifier
getters, the ``Processing`` builders and the stage guard that writes
``processing.json``; this module adds the one field it does not provide —
micrometres per pixel — assembles the ``ClassifierMetadata`` view, and binds
the guard to this capsule's process type and code identity.

**Why the µm/px readers live here.** v1 stored it directly as
``ophys_fovs[*].fov_scale_factor`` plus a ``fov_scale_factor_unit`` string. v2
removed both: the scale is a ``Scale`` entry inside
``PlanarImage.image_to_acquisition_transform`` and the unit moved to
``ImagingConfig.coordinate_system.axis_unit``. ``aind_pophys_metadata`` has no
getter for either shape and must not be modified, so both live here.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

from aind_data_schema.components.configs import (
    ImagingConfig,
    PlanarImage,
    PlanarImageStack,
)
from aind_data_schema.components.coordinates import Scale
from aind_data_schema.components.identifiers import Code
from aind_data_schema_models.process_names import ProcessName
from aind_pophys_metadata import fields, io
from aind_pophys_metadata import processing as shared_processing
from aind_pophys_metadata.runner import StageContext, stage_guard
from pint import UnitRegistry

PROCESS_NAME = "aind-ophys-classifier"
LIBRARY_NAME = "aind-ophys-classifier-library"
PROCESS_TYPE = ProcessName.IMAGE_CELL_CLASSIFICATION

MICRONS_PER_PIXEL = "um/pixel"

# v2 discriminators read from the schema classes, never hardcoded, so they
# track the installed aind-data-schema version.
_V2_IMAGING_CONFIG_TYPE = io.object_type_value(ImagingConfig)
_V2_PLANAR_IMAGE_TYPES = {
    io.object_type_value(PlanarImage),
    io.object_type_value(PlanarImageStack),
}
_V2_SCALE_TYPE = io.object_type_value(Scale)

# Source labels recorded in processing.json alongside the resolved value.
SOURCE_V1 = io.V1_CORE_FILE
SOURCE_V2 = io.V2_CORE_FILE
SOURCE_OVERRIDE = "--um_per_pixel"


@dataclass(frozen=True)
class ClassifierMetadata:
    """Minimal metadata the ROI classifier consumes.

    ``um_per_pixel`` scales the ROI images handed to ROInet, so it changes
    the embedding and therefore every prediction; ``um_per_pixel_source``
    records where it came from for reproducibility. The identifiers are
    logging context only and stay optional.
    """

    um_per_pixel: float
    um_per_pixel_source: str
    instrument_id: Optional[str] = None
    subject_id: Optional[str] = None
    dataset_name: Optional[str] = None


def get_um_per_pixel_v1(session: dict) -> Optional[float]:
    """Micrometres per pixel from a v1 session's first ophys FOV.

    v1 carries ``fov_scale_factor`` with a companion
    ``fov_scale_factor_unit`` string, which is normalised to µm/pixel via
    ``pint``.

    Parameters
    ----------
    session : dict
        Raw ``session.json`` dict.

    Returns
    -------
    float or None
        Micrometres per pixel, or ``None`` when no FOV declares a scale.
    """
    for stream in session.get("data_streams") or []:
        for fov in stream.get("ophys_fovs") or []:
            factor = fov.get("fov_scale_factor")
            if factor is None:
                continue
            unit = fov.get("fov_scale_factor_unit") or MICRONS_PER_PIXEL
            magnitude = UnitRegistry().parse_expression(unit).to(MICRONS_PER_PIXEL).magnitude
            return float(factor) * float(magnitude)
    return None


def get_um_per_pixel_v2(acquisition: dict) -> Optional[float]:
    """Micrometres per pixel from a v2 acquisition.

    Reads the ``Scale`` entry of
    ``PlanarImage.image_to_acquisition_transform``. Deliberately ignores
    ``PlanarImage.dimensions``, which is also a ``Scale`` but holds the
    image's pixel dimensions — reading it would silently yield e.g. 512
    µm/pixel. The unit comes from ``ImagingConfig.coordinate_system``, whose
    ``axis_unit`` is expected to be micrometres.

    Parameters
    ----------
    acquisition : dict
        Raw ``acquisition.json`` dict.

    Returns
    -------
    float or None
        Micrometres per pixel, or ``None`` when no imaging config carries a
        transform scale.
    """
    for stream in acquisition.get("data_streams") or []:
        for config in stream.get("configurations") or []:
            if config.get("object_type") != _V2_IMAGING_CONFIG_TYPE:
                continue
            unit = (config.get("coordinate_system") or {}).get("axis_unit")
            for image in config.get("images") or []:
                if image.get("object_type") not in _V2_PLANAR_IMAGE_TYPES:
                    continue
                for transform in image.get("image_to_acquisition_transform") or []:
                    if transform.get("object_type") != _V2_SCALE_TYPE:
                        continue
                    scale = transform.get("scale") or []
                    if not scale:
                        continue
                    return _to_microns_per_pixel(float(scale[0]), unit)
    return None


def _to_microns_per_pixel(value: float, unit: Optional[str]) -> float:
    """Convert a v2 transform scale to µm/pixel using the coordinate unit.

    Parameters
    ----------
    value : float
        The scale magnitude from the ``Scale`` transform.
    unit : str or None
        ``CoordinateSystem.axis_unit``; ``None`` or micrometre is a no-op.

    Returns
    -------
    float
        The value expressed in micrometres per pixel.
    """
    if not unit:
        return value
    normalised = str(unit).strip().lower()
    if normalised in ("micrometer", "micrometre", "um", "µm", "microns"):
        return value
    registry = UnitRegistry()
    return float((value * registry.parse_expression(normalised)).to("micrometer").magnitude)


def load_classifier_metadata(
    input_dir: Path,
    um_per_pixel_override: Optional[float] = None,
) -> ClassifierMetadata:
    """Read v1/v2 metadata for classification from ``input_dir``.

    Dispatches on which core file is present (``acquisition.json`` -> v2,
    ``session.json`` -> v1). Fails fast when µm/pixel cannot be resolved:
    ROInet silently assumes 1.0 µm/pixel when it is missing, which changes
    every prediction without erroring, so a missing value must never be
    defaulted here.

    Parameters
    ----------
    input_dir : Path
        Directory searched recursively for the metadata files.
    um_per_pixel_override : float, optional
        Explicit value supplied on the command line; when given, the
        metadata is not consulted for it.

    Returns
    -------
    ClassifierMetadata
        The minimal fields this capsule consumes.

    Raises
    ------
    FileNotFoundError
        If neither ``acquisition.json`` nor ``session.json`` is found.
    ValueError
        If both core files are present (ambiguous input), or if µm/pixel
        cannot be resolved and no override was supplied.
    """
    common = io.load_common(Path(input_dir))
    core_path = common.core_path
    version = common.version
    core_raw = common.core_raw

    if um_per_pixel_override is not None:
        um_per_pixel = float(um_per_pixel_override)
        source = SOURCE_OVERRIDE
    else:
        if version == io.SCHEMA_V1:
            um_per_pixel = get_um_per_pixel_v1(core_raw)
        else:
            um_per_pixel = get_um_per_pixel_v2(core_raw)
        source = core_path.name

    if um_per_pixel is None:
        raise ValueError(
            f"Could not resolve micrometres per pixel from {core_path.name} "
            f"(schema {version}). v1 expects "
            f"data_streams[*].ophys_fovs[*].fov_scale_factor; v2 expects a "
            f"'{_V2_SCALE_TYPE}' entry in "
            f"PlanarImage.image_to_acquisition_transform. Pass "
            f"{SOURCE_OVERRIDE} to override. Refusing to default: ROInet "
            "assumes 1.0 um/pixel when unset, which silently changes every "
            "classification."
        )

    return ClassifierMetadata(
        um_per_pixel=um_per_pixel,
        um_per_pixel_source=source,
        instrument_id=fields.get_instrument_id(version, core_raw),
        subject_id=fields.get_subject_id(core_raw, common.subject_raw),
        dataset_name=fields.get_dataset_name(common.data_description_raw),
    )


def validate_plane_ids(
    input_dir: Path,
    scavenged_ids: List[str],
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Warn when plane ids taken from directory names are not canonical.

    This capsule scavenges its plane ids from the layout of the extraction
    output (``<plane>/extraction/<plane>_extraction.h5``) rather than from the
    acquisition metadata. Uniqueness is the hard requirement and directory
    names already guarantee it, so a divergence from the canonical ids is
    reported and never raised.

    The core file is re-read here rather than carried on
    :class:`ClassifierMetadata`, because the scavenged ids are only known once
    the input tree has been walked, and a field that exists solely to be
    passed back in is the kind that silently rots.

    Parameters
    ----------
    input_dir : Path
        Capsule input directory holding the acquisition metadata.
    scavenged_ids : list of str
        Plane ids recovered from the input directory layout.
    logger : logging.Logger, optional
        Logger the divergence warning is emitted on.

    Returns
    -------
    bool
        ``True`` when the scavenged ids match the canonical ids.
    """
    common = io.load_common(Path(input_dir))
    canonical = fields.get_fov_ids(common.version, common.core_raw)
    # SINGLE mode checks membership, not set equality. This capsule runs one
    # task per plane, so it scavenges one id while get_fov_ids returns all of
    # them; the default ALL mode compared the two as sets and warned on every
    # plane of every multiplane run (8 warnings in run bb739583, the only
    # source of them in the whole fleet). A warning that always fires is one
    # nobody reads.
    mode = (
        fields.SCAVENGE_MODE_SINGLE
        if len(list(scavenged_ids)) == 1
        else fields.SCAVENGE_MODE_ALL
    )
    return fields.validate_scavenged_ids(
        scavenged_ids, canonical, logger, mode=mode
    )


def build_classification_code(
    parameters: dict,
    input_data: Optional[List[str]] = None,
) -> Code:
    """Construct the ``Code`` block recorded on the classification process.

    ``url`` and ``version`` are both derived by the shared builder from
    ``LIBRARY_NAME``: this library's landing page and its installed released
    version, so the two always describe the same artifact.

    Parameters
    ----------
    parameters : dict
        Resolved run parameters recorded on the code block.
    input_data : list of str, optional
        Input data path(s)/asset name(s) for provenance.

    Returns
    -------
    Code
        The populated code block.
    """
    return shared_processing.build_code(
        name=PROCESS_NAME,
        library_name=LIBRARY_NAME,
        parameters=parameters,
        input_data=input_data,
    )


def process_name(plane_id: Optional[str] = None) -> str:
    """Return the ``DataProcess.name`` the shared builder will compose.

    Parameters
    ----------
    plane_id : str, optional
        Plane/FOV id for a per-plane run; omitted for a run that never
        reached a plane.

    Returns
    -------
    str
        The composed process name.
    """
    base = PROCESS_TYPE.value
    return f"{plane_id}: {base}" if plane_id else base


@contextmanager
def classification_stage(
    output_dir: Path,
    code: Code,
    *,
    plane_id: Optional[str] = None,
    output_path: Optional[str] = None,
    input_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Iterator[StageContext]:
    """Wrap one classification stage in the shared :func:`stage_guard`.

    The guard stamps the timings, captures static resources and writes
    ``processing.json`` on the way out whether or not the body raised, with
    the error recorded in ``notes``. This wrapper adds the two capsule-level
    decisions: the process type, and the sibling upstream dependencies read
    out of the input directory.

    Every ``processing.json`` staged into ``input_dir`` is treated as a flat
    sibling upstream — no ordering and no parent/grandparent distinction —
    with this process's own name excluded so a step cannot depend on itself.

    Parameters
    ----------
    output_dir : Path
        Directory the ``processing.json`` is written to.
    code : Code
        The code block (see :func:`build_classification_code`).
    plane_id : str, optional
        Plane/FOV id, which qualifies the process name so it stays unique
        once every plane's document is merged.
    output_path : str, optional
        This stage's output directory relative to the results root.
    input_dir : Path, optional
        Capsule input directory, searched for upstream ``processing.json``
        documents. Omitted when the run failed before it was known.
    logger : logging.Logger, optional
        Logger for the stage lifecycle lines.

    Yields
    ------
    StageContext
        Handle the body records output parameters and resources on.
    """
    with stage_guard(
        output_dir,
        PROCESS_TYPE,
        code,
        plane_id=plane_id,
        output_path=output_path,
        logger=logger,
    ) as context:
        if input_dir is not None:
            context.upstream_names = (
                shared_processing.collect_upstream_process_names(
                    input_dir, exclude=process_name(plane_id)
                )
            )
        yield context
