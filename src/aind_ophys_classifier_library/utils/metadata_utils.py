"""Classification metadata view, ``Code`` block and stage guard over ``aind_pophys_metadata``."""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

from aind_data_schema.components.identifiers import Code
from aind_data_schema_models.process_names import ProcessName
from aind_pophys_metadata import CoreMetadata
from aind_pophys_metadata import processing as shared_processing
from aind_pophys_metadata.runner import StageContext, stage_guard

PROCESS_NAME = "aind-ophys-classifier"
LIBRARY_NAME = "aind-ophys-classifier-library"
LIBRARY_URL = (
    "https://github.com/AllenNeuralDynamics/aind-ophys-classifier-library"
)
PROCESS_TYPE = ProcessName.IMAGE_CELL_CLASSIFICATION

# Source label recorded in processing.json when the value came from the CLI
# rather than the core file.
SOURCE_OVERRIDE = "--um_per_pixel"


@dataclass(frozen=True)
class ClassifierMetadata:
    """Minimal metadata the ROI classifier consumes."""

    um_per_pixel: float
    um_per_pixel_source: str
    instrument_id: Optional[str] = None
    subject_id: Optional[str] = None
    dataset_name: Optional[str] = None


def load_classifier_metadata(
    input_dir: Path,
    um_per_pixel_override: Optional[float] = None,
) -> ClassifierMetadata:
    """Read the metadata classification consumes from ``input_dir``.

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
        The minimal fields this step consumes.

    Raises
    ------
    FileNotFoundError
        If no core metadata file is found.
    ValueError
        If the input is ambiguous or unsupported, or if µm/pixel cannot be
        resolved and no override was supplied.
    """
    core = CoreMetadata.load(Path(input_dir))
    core_path = core.core_path

    if um_per_pixel_override is not None:
        um_per_pixel = float(um_per_pixel_override)
        source = SOURCE_OVERRIDE
    else:
        um_per_pixel = core.get_um_per_pixel()
        source = core_path.name

    if um_per_pixel is None:
        raise ValueError(
            f"Could not resolve micrometres per pixel from {core_path.name} "
            f"(schema {core.version.value}): the metadata either does not declare "
            "a per-pixel scale or declares it in a unit that cannot be "
            f"converted to micrometres. Pass {SOURCE_OVERRIDE} to supply it "
            "explicitly. Refusing to default: ROInet assumes 1.0 um/pixel "
            "when unset, which silently changes every classification."
        )

    return ClassifierMetadata(
        um_per_pixel=um_per_pixel,
        um_per_pixel_source=source,
        instrument_id=core.get_instrument_id(),
        subject_id=core.get_subject_id(),
        dataset_name=core.get_dataset_name(),
    )


def build_classification_code(
    parameters: dict,
    input_data: Optional[List[str]] = None,
) -> Code:
    """Construct the ``Code`` block recorded on the classification process.

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
        url=LIBRARY_URL,
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
        Input directory, searched for upstream ``processing.json``
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
