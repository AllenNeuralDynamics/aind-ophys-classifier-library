"""Top-level ROI-classification orchestration.

``run()`` is the library entry point called by the thin capsule wrapper:
read metadata (v1 or v2) -> configure logging -> for each plane, embed and
classify its ROIs -> write the classification HDF5, the overlay figures and
the combined figure -> emit a full ``processing.json`` and
``quality_control.json`` per plane.

``processing.json`` is written from a ``finally`` block so a failed run still
leaves a document describing what was attempted, with the error recorded in
``notes``.

In the pipeline this capsule is flattened (one task per plane), so the loop
normally runs once; a standalone capsule run may mount several planes.
"""

import logging
from datetime import datetime as dt
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
from aind_data_schema.core.processing import ResourceUsage
from aind_pophys_metadata import processing as shared_processing
from log_schema.main import setup_logging

from . import classification, plots, qc
from .settings import ClassifierSettings
from .utils import image_utils
from .utils import metadata_utils as mu
from .utils.paths import relative_to_root

logger = logging.getLogger("Ophys classification")

PROCESS_NAME = mu.PROCESS_NAME
CLASSIFIER_ASSET_NAME = "2p_roi_classifier"


def _verify_outputs(output_dir: Path) -> None:
    """Round-trip the emitted metadata through the core v2 schema objects."""
    from aind_data_schema.core.processing import Processing
    from aind_data_schema.core.quality_control import QualityControl

    Processing.model_validate_json((Path(output_dir) / "processing.json").read_text())
    QualityControl.model_validate_json((Path(output_dir) / "quality_control.json").read_text())


def _resource_usage() -> ResourceUsage:
    """Host resources, including the GPU the shared builder omits."""
    usage = shared_processing.resource_usage()
    gpu = classification.gpu_name()
    if gpu is None:
        return usage
    return ResourceUsage(
        os=usage.os,
        architecture=usage.architecture,
        cpu_cores=usage.cpu_cores,
        gpu=gpu,
    )


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


def _write_processing_metadata(
    settings: ClassifierSettings,
    ctx: dict,
    start_time: dt,
    error: Optional[BaseException],
) -> None:
    """Best-effort writer for processing.json. Never raises.

    Failures here must not mask a run failure already propagating out of
    ``run()``. Falls back to the results root when the run failed before any
    per-plane output directory was created.
    """
    results_root = settings.output_dir.resolve()
    output_dir = ctx.get("output_dir") or results_root
    try:
        data_process = mu.build_data_process(
            parameters=ctx.get("parameters") or {},
            start_time=start_time,
            end_time=dt.now(),
            input_data=ctx.get("input_data"),
            output_path=relative_to_root(results_root, output_dir),
            output_parameters=ctx.get("output_parameters"),
            resources=_resource_usage(),
            error=error,
        )
        mu.write_processing_json(mu.build_processing([data_process]), output_dir)
    except Exception as proc_exc:
        logger.error(f"processing.json write failed (non-fatal): {proc_exc}")


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


def _require_classifiers(settings: ClassifierSettings) -> None:
    """Fail fast when either ONNX model is missing, naming which."""
    for label, path in (
        ("soma", settings.soma_classifier_path),
        ("dendrite", settings.dendrite_classifier_path),
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(f"{label} classifier ONNX not found: {path}")


def _run_classification(
    settings: ClassifierSettings,
    metadata: mu.ClassifierMetadata,
    ctx: dict,
) -> None:
    """Classify every mounted plane and write all outputs.

    ``ctx`` is populated as the run progresses so the ``processing.json``
    writer in ``run()``'s ``finally`` can report partial provenance even when
    this raises.
    """
    results_root = settings.output_dir.resolve()
    input_dir = settings.input_dir.resolve()

    _require_classifiers(settings)
    model_name = settings.model_name or settings.soma_classifier_path.parent.name
    parameters = _run_parameters(settings, metadata, model_name)
    ctx["parameters"] = parameters

    network_files_dir, download_method = classification.resolve_network_files_dir(
        settings.tmp_dir, settings.roinet_dir
    )
    soma_classifier = classification.load_onnx_classifier(settings.soma_classifier_path)
    dendrite_classifier = classification.load_onnx_classifier(settings.dendrite_classifier_path)

    extraction_files = classification.find_extraction_files(input_dir)
    logger.info("Classifying %s plane(s)", len(extraction_files))

    for extraction_file in extraction_files:
        plane = classification.prepare_plane(extraction_file, results_root)
        ctx["output_dir"] = plane.output_dir
        ctx["input_data"] = [
            relative_to_root(input_dir, extraction_file),
            CLASSIFIER_ASSET_NAME,
        ]

        rois = classification.load_extraction_rois(extraction_file)
        result = classification.classify_plane(
            rois=rois,
            um_per_pixel=metadata.um_per_pixel,
            soma_classifier=soma_classifier,
            dendrite_classifier=dendrite_classifier,
            network_files_dir=network_files_dir,
            download_method=download_method,
        )
        result["rois"] = rois
        border = classification.classify_border_rois(rois, settings.border_size)

        figure_paths = _write_figures(plane, result, border)
        classification.write_classification_h5(plane, result, border)

        ctx["output_parameters"] = {
            "num_rois": int(len(result["soma_predictions"])),
            "num_soma": int(result["soma_predictions"].sum()),
            "num_dendrite": int(result["dendrite_predictions"].sum()),
            "num_border": int(border.sum()),
        }
        metrics = _collect_metrics(
            plane,
            result,
            border,
            figure_paths,
            settings.border_size,
            results_root,
        )
        qc.write_quality_control_json(qc.build_quality_control(metrics), plane.output_dir)
        _write_processing_metadata(settings, ctx, ctx["start_time"], None)
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
    start_time = dt.now()

    metadata = mu.load_classifier_metadata(settings.input_dir, settings.um_per_pixel)
    setup_logging(
        use_cloudwatch=True,
        model={
            "acquisition_name": metadata.dataset_name or metadata.subject_id,
            "process_name": PROCESS_NAME,
        },
    )
    logger.info("Starting classification", extra={"event_type": "stage_start"})
    logger.info(
        "Resolved %s um/pixel from %s",
        metadata.um_per_pixel,
        metadata.um_per_pixel_source,
    )

    ctx: dict = {"start_time": start_time}
    try:
        _run_classification(settings, metadata, ctx)
        logger.info(
            "Classification complete",
            extra={"event_type": "stage_complete"},
        )
    except BaseException as exc:
        logger.error("Classification failed", extra={"event_type": "stage_error"})
        logger.exception("Unhandled exception")
        # Written here rather than in a `finally`: each plane writes its own
        # processing.json as it completes, so a single trailing write would
        # only ever describe the last plane. On failure this records the plane
        # that was in flight, with the error in `notes`.
        _write_processing_metadata(settings, ctx, start_time, exc)
        raise

    if settings.verify:
        _verify_outputs(ctx["output_dir"])
