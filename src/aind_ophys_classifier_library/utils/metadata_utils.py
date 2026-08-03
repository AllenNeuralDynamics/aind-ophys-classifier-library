"""aind-data-schema v1/v2 metadata reads for ophys ROI classification.

A thin, capsule-specific layer over the shared ``aind_pophys_metadata``
library. The shared library owns schema-version dispatch, the identifier
getters and the ``Processing`` builders; this module adds the one field it
does not provide — micrometres per pixel — and assembles the
``ClassifierMetadata`` view.

**Why the µm/px readers live here.** v1 stored it directly as
``ophys_fovs[*].fov_scale_factor`` plus a ``fov_scale_factor_unit`` string. v2
removed both: the scale is a ``Scale`` entry inside
``PlanarImage.image_to_acquisition_transform`` and the unit moved to
``ImagingConfig.coordinate_system.axis_unit``. ``aind_pophys_metadata`` has no
getter for either shape and must not be modified, so both live here.
"""

import os
from dataclasses import dataclass
from datetime import datetime as dt
from pathlib import Path
from typing import List, Optional, Tuple

import pytz
from aind_data_schema.components.configs import (
    ImagingConfig,
    PlanarImage,
    PlanarImageStack,
)
from aind_data_schema.components.coordinates import Scale
from aind_data_schema.core.processing import DataProcess, Processing
from aind_data_schema_models.process_names import ProcessName
from aind_pophys_metadata import fields, io
from aind_pophys_metadata import processing as shared_processing
from pint import UnitRegistry

from aind_ophys_classifier_library import __version__

_SEATTLE_TZ = pytz.timezone("America/Los_Angeles")

CODE_URL = "https://github.com/AllenNeuralDynamics/aind-ophys-classifier"
PROCESS_NAME = "aind-ophys-classifier"

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
    fov_ids: Tuple[str, ...] = ()


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
    input_dir = Path(input_dir)
    core_path = io.find_acquisition_file(input_dir)
    version = io.detect_schema_version(core_path)
    core_raw = io.load_json(core_path)

    subject_path = io.find(input_dir, "subject.json")
    subject_raw = io.load_json(subject_path) if subject_path else None
    dd_path = io.find(input_dir, "data_description.json")
    dd_raw = io.load_json(dd_path) if dd_path else None

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
        subject_id=fields.get_subject_id(core_raw, subject_raw),
        dataset_name=fields.get_dataset_name(dd_raw),
        fov_ids=fields.get_fov_ids(version, core_raw),
    )


def code_version() -> str:
    """Return the version recorded on the ``Code`` block.

    aind-data-schema warns when a ``Code`` carries neither ``commit_hash`` nor
    ``version``, and says it will require one in future. The library version is
    the fallback rather than the empty string a bare
    ``os.getenv("VERSION", "")`` yields wherever the capsule image does not set
    ``VERSION``. Only one of the two may be set, and the shared builder takes
    ``version``.

    Returns
    -------
    str
        The ``VERSION`` environment value, or this library's version.
    """
    return os.getenv("VERSION") or __version__


def build_data_process(
    parameters: dict,
    start_time: dt,
    end_time: dt,
    input_data: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    output_parameters: Optional[dict] = None,
    plane_id: Optional[str] = None,
    resources=None,
    error: Optional[BaseException] = None,
) -> DataProcess:
    """Construct a v2 DataProcess for ROI classification.

    Parameters
    ----------
    parameters : dict
        Resolved run parameters recorded on the code block.
    start_time : dt
        Processing start time.
    end_time : dt
        Processing end time.
    input_data : list of str, optional
        Input data path(s)/asset name(s) for provenance.
    output_path : str, optional
        Path to this process's output directory, relative to the results
        root.
    output_parameters : dict, optional
        Actual parameters used and output metrics.
    plane_id : str, optional
        Plane/FOV id this run processed. Qualifies the process name so it
        stays unique once every plane's document is merged.
    resources : ResourceUsage, optional
        Host resource usage; defaults to the shared CPU-only builder. The
        classifier is the pipeline's only GPU process, so the caller passes
        a GPU-aware value.
    error : Exception, optional
        The exception that terminated the run, or ``None`` on success. Its
        message is recorded in ``notes``.

    Returns
    -------
    DataProcess
        The populated classification data process.
    """
    notes = None if error is None else f"Run failed with {type(error).__name__}: {error}"
    code = shared_processing.build_code(
        url=CODE_URL,
        name=PROCESS_NAME,
        version=code_version(),
        parameters=parameters,
        input_data=input_data,
    )
    return shared_processing.build_data_process(
        process_type=ProcessName.IMAGE_CELL_CLASSIFICATION,
        plane_id=plane_id,
        code=code,
        start_time=start_time.astimezone(_SEATTLE_TZ),
        end_time=end_time.astimezone(_SEATTLE_TZ),
        output_path=output_path,
        output_parameters=output_parameters,
        notes=notes,
        resources=resources or shared_processing.resource_usage(),
    )


def build_processing(data_processes: List[DataProcess]) -> Processing:
    """Wrap data processes into a full v2 Processing object."""
    return shared_processing.build_processing(data_processes)


def write_processing_json(processing: Processing, output_dir: Path) -> Path:
    """Serialize a Processing object to ``processing.json``.

    Parameters
    ----------
    processing : Processing
        The processing document to write.
    output_dir : Path
        Directory into which ``processing.json`` is written.

    Returns
    -------
    Path
        The path written.
    """
    return shared_processing.write_processing_json(processing, output_dir)
