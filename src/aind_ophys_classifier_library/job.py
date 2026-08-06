"""Top-level ROI-classification orchestration.

``run()`` is the library entry point called by the thin capsule wrapper:
read metadata (v1 or v2) -> configure logging -> for each plane, embed and
classify its ROIs -> write the classification HDF5, the overlay figures and
the combined figure -> emit a full ``processing.json`` and
``quality_control.json`` per plane.

``processing.json`` is written by the shared ``stage_guard``, so a failed run
still leaves a document describing what was attempted, with the error recorded
in ``notes``. Each plane gets its own guard, and a failure before any plane is
known writes one document at the results root.

In the pipeline this capsule is flattened (one task per plane), so the loop
normally runs once; a standalone capsule run may mount several planes.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from aind_data_schema.core.processing import ResourceUsage
from aind_pophys_metadata.paths import relative_to_root
from log_schema.main import setup_logging

from . import classification, plots, qc
from .settings import ClassifierSettings
from .utils import image_utils
from .utils import metadata_utils as mu

logger = logging.getLogger("Ophys classification")

PROCESS_NAME = mu.PROCESS_NAME
CLASSIFIER_ASSET_NAME = "2p_roi_classifier"


def _verify_outputs(output_dir: Path) -> None:
    """Round-trip the emitted metadata through the core v2 schema objects."""
    from aind_data_schema.core.processing import Processing
    from aind_data_schema.core.quality_control import QualityControl

    Processing.model_validate_json((Path(output_dir) / "processing.json").read_text())
    QualityControl.model_validate_json((Path(output_dir) / "quality_control.json").read_text())


def _with_gpu(resources: ResourceUsage) -> ResourceUsage:
    """Add the active GPU name to the guard's static resource capture."""
    gpu = classification.gpu_name()
    if gpu is None:
        return resources
    return resources.model_copy(update={"gpu": gpu})


def _save(figure, path: Path) -> None:
    """Write a matplotlib figure and close it."""
    figure.savefig(str(path), bbox_inches="tight")
    plt.close(figure)


def _write_figures(plane: classification.Plane, result: dict, border) -> dict:
    """Write the four overlay figures plus the combined figure.

    Returns
    -------
    dict
        Figure paths, keyed by role.
    """
    name = plane.name
    paths = {
        "soma_probabilities": plane.output_dir / f"{name}_soma_probabilities.png",
        "dendrite_probabilities": plane.output_dir / f"{name}_dendrite_probabilities.png",
        "soma_predictions": plane.output_dir / f"{name}_soma_predictions.png",
        "dendrite_predictions": plane.output_dir / f"{name}_dendrite_predictions.png",
        "border_rois": plane.output_dir / f"{name}_border_rois.png",
        "combined": plane.output_dir / f"{name}_combined_classifications.png",
    }
    rois = result["rois"]

    _save(
        plots.plot_probabilities(rois, result["soma_probabilities"], "soma probabilities"),
        paths["soma_probabilities"],
    )
    _save(
        plots.plot_probabilities(rois, result["dendrite_probabilities"], "dendrite probabilities"),
        paths["dendrite_probabilities"],
    )
    _save(
        plots.plot_predictions(rois, result["soma_predictions"], "soma predictions"),
        paths["soma_predictions"],
    )
    _save(
        plots.plot_predictions(rois, result["dendrite_predictions"], "dendrite predictions"),
        paths["dendrite_predictions"],
    )
    _save(
        plots.plot_border_rois(rois, border, "border ROIs"),
        paths["border_rois"],
    )

    # Net new: the retired QC aggregator used to assemble this, and the
    # classification metric has referenced it all along.
    image_utils.build_combined_classifications(
        soma_predictions_png=paths["soma_predictions"],
        soma_probabilities_png=paths["soma_probabilities"],
        dendrite_predictions_png=paths["dendrite_predictions"],
        dendrite_probabilities_png=paths["dendrite_probabilities"],
        output_path=paths["combined"],
    )
    return paths


def _collect_metrics(
    plane: classification.Plane,
    result: dict,
    border,
    figure_paths: dict,
    border_size: int,
    results_root: Path,
) -> List:
    """Build every QC metric for one plane (the collector pattern)."""
    name = plane.name
    combined = relative_to_root(results_root, figure_paths["combined"])
    border_ref = relative_to_root(results_root, figure_paths["border_rois"])
    soma_ref = relative_to_root(results_root, figure_paths["soma_predictions"])
    dendrite_ref = relative_to_root(results_root, figure_paths["dendrite_predictions"])

    soma = result["soma_predictions"]
    dendrite = result["dendrite_predictions"]
    num_rois = int(len(soma))

    labels = qc.roi_labels(soma, dendrite, border)
    return [
        qc.build_classification_metric(name, combined),
        qc.build_border_roi_metric(name, border_ref),
        qc.build_roi_label_curation_metric(name, labels, combined),
        qc.build_roi_count_metric(name, num_rois, combined),
        qc.build_positive_count_metric(name, "Soma", int(soma.sum()), num_rois, soma_ref),
        qc.build_positive_count_metric(
            name, "Dendrite", int(dendrite.sum()), num_rois, dendrite_ref
        ),
        qc.build_border_count_metric(name, int(border.sum()), border_size, border_ref),
    ]


def _run_parameters(
    settings: ClassifierSettings,
    metadata: mu.ClassifierMetadata,
    model_name: str,
) -> dict:
    """Assemble the parameter dict recorded in processing.json.

    ``um_per_pixel`` and its source are recorded because they scale the
    embedding, making them the single most important reproducibility fields
    — the v1 parameter dict omitted both.
    """
    return {
        "border_size": settings.border_size,
        "model": model_name,
        "soma_classifier_path": str(settings.soma_classifier_path),
        "soma_classifier_md5_hash": classification.file_md5(settings.soma_classifier_path),
        "dendrite_classifier_path": str(settings.dendrite_classifier_path),
        "dendrite_classifier_md5_hash": classification.file_md5(settings.dendrite_classifier_path),
        "um_per_pixel": metadata.um_per_pixel,
        "um_per_pixel_source": metadata.um_per_pixel_source,
    }


def _output_parameters(result: dict, border) -> dict:
    """Summarise this plane's classification outputs for QC triage.

    ``DataProcess.output_parameters`` carries scalar summary statistics
    describing the outputs this step produced, chosen so a reviewer can triage
    a plane without opening the classification HDF5. It is deliberately not a
    record of the run's *input* parameters — those belong on
    ``code.parameters``, and conflating the two is how the field rots.

    Parameters
    ----------
    result : dict
        Output of :func:`classification.classify_plane`.
    border : numpy.ndarray
        Boolean per-ROI border flags.

    Returns
    -------
    dict
        ROI counts: total, soma-positive, dendrite-positive and border.
    """
    return {
        "num_rois": int(len(result["soma_predictions"])),
        "num_soma": int(result["soma_predictions"].sum()),
        "num_dendrite": int(result["dendrite_predictions"].sum()),
        "num_border": int(border.sum()),
    }


def _require_classifiers(settings: ClassifierSettings) -> None:
    """Fail fast when either ONNX model is missing, naming which."""
    for label, path in (
        ("soma", settings.soma_classifier_path),
        ("dendrite", settings.dendrite_classifier_path),
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(f"{label} classifier ONNX not found: {path}")


@dataclass(frozen=True)
class _Prepared:
    """Everything resolved once, before any plane is classified.

    Attributes
    ----------
    metadata : mu.ClassifierMetadata
        The acquisition metadata view.
    planes : tuple of classification.Plane
        Every mounted plane, in deterministic order.
    parameters : dict
        Run parameters recorded on each plane's ``Code`` block.
    soma_classifier : callable
        The loaded soma ONNX classifier.
    dendrite_classifier : callable
        The loaded dendrite ONNX classifier.
    network_files_dir : Path
        Directory holding or receiving the ROInet archive.
    download_method : str
        ROInet download method.
    """

    metadata: mu.ClassifierMetadata
    planes: Tuple[classification.Plane, ...]
    parameters: dict
    soma_classifier: object
    dendrite_classifier: object
    network_files_dir: Path
    download_method: str


def _prepare(settings: ClassifierSettings) -> _Prepared:
    """Read metadata, configure logging, and resolve every shared input.

    Everything that a plane's classification depends on is resolved here, so
    a per-plane stage guard covers the whole of the work that can fail once a
    plane is known.
    """
    metadata = mu.load_classifier_metadata(settings.input_dir, settings.um_per_pixel)
    setup_logging(
        use_cloudwatch=True,
        model={
            "acquisition_name": metadata.dataset_name or metadata.subject_id,
            "process_name": PROCESS_NAME,
        },
    )
    logger.info("Starting classification")
    logger.info(
        "Resolved %s um/pixel from %s",
        metadata.um_per_pixel,
        metadata.um_per_pixel_source,
    )

    _require_classifiers(settings)
    model_name = settings.model_name or settings.soma_classifier_path.parent.name
    network_files_dir, download_method = classification.resolve_network_files_dir(
        settings.tmp_dir, settings.roinet_dir
    )

    input_dir = settings.input_dir.resolve()
    results_root = settings.output_dir.resolve()
    extraction_files = classification.find_extraction_files(input_dir)
    planes = tuple(
        classification.prepare_plane(extraction_file, results_root)
        for extraction_file in extraction_files
    )
    # Plane ids come from the input directory layout; the canonical ids are
    # only advisory here, so a divergence warns and the run continues.
    mu.validate_plane_ids(input_dir, [plane.name for plane in planes], logger)
    logger.info("Classifying %s plane(s)", len(planes))

    return _Prepared(
        metadata=metadata,
        planes=planes,
        parameters=_run_parameters(settings, metadata, model_name),
        soma_classifier=classification.load_onnx_classifier(settings.soma_classifier_path),
        dendrite_classifier=classification.load_onnx_classifier(
            settings.dendrite_classifier_path
        ),
        network_files_dir=network_files_dir,
        download_method=download_method,
    )


def _classify_plane(
    settings: ClassifierSettings,
    prepared: _Prepared,
    plane: classification.Plane,
) -> None:
    """Classify one plane inside its own stage guard.

    The guard writes the plane's ``processing.json`` on the way out whether or
    not this raised, so a failed plane still leaves a document naming what was
    attempted and why it stopped.
    """
    results_root = settings.output_dir.resolve()
    input_dir = settings.input_dir.resolve()
    code = mu.build_classification_code(
        prepared.parameters,
        input_data=[
            relative_to_root(input_dir, plane.input_extraction_file),
            CLASSIFIER_ASSET_NAME,
        ],
    )
    with mu.classification_stage(
        plane.output_dir,
        code,
        plane_id=plane.name,
        output_path=relative_to_root(results_root, plane.output_dir),
        input_dir=input_dir,
        logger=logger,
    ) as stage:
        stage.resources = _with_gpu(stage.resources)

        rois = classification.load_extraction_rois(plane.input_extraction_file)
        result = classification.classify_plane(
            rois=rois,
            um_per_pixel=prepared.metadata.um_per_pixel,
            soma_classifier=prepared.soma_classifier,
            dendrite_classifier=prepared.dendrite_classifier,
            network_files_dir=prepared.network_files_dir,
            download_method=prepared.download_method,
        )
        result["rois"] = rois
        border = classification.classify_border_rois(rois, settings.border_size)

        figure_paths = _write_figures(plane, result, border)
        classification.write_classification_h5(plane, result, border)

        stage.output_parameters = _output_parameters(result, border)
        metrics = _collect_metrics(
            plane,
            result,
            border,
            figure_paths,
            settings.border_size,
            results_root,
        )
        qc.write_quality_control_json(qc.build_quality_control(metrics), plane.output_dir)
    logger.info("Wrote classification outputs for %s", plane.name)


def run(settings: Optional[ClassifierSettings] = None) -> None:
    """Run ophys ROI classification end to end.

    Parameters
    ----------
    settings : ClassifierSettings, optional
        Parsed settings; constructed from CLI/env when omitted. When
        ``settings.verify`` is True, the emitted metadata is round-tripped
        through the core v2 schema objects after writing.
    """
    settings = settings or ClassifierSettings()
    try:
        prepared = _prepare(settings)
    except BaseException:
        logger.error("Classification failed")
        logger.exception("Unhandled exception")
        # No plane is known yet, so the guard writes one document at the
        # results root; re-raising inside it is what records the error.
        with mu.classification_stage(
            settings.output_dir.resolve(),
            mu.build_classification_code({}),
            input_dir=settings.input_dir.resolve(),
            logger=logger,
        ):
            raise

    for plane in prepared.planes:
        _classify_plane(settings, prepared, plane)
    logger.info("Classification complete")

    if settings.verify:
        _verify_outputs(prepared.planes[-1].output_dir)
