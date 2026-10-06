"""quality_control.json generation for ophys ROI classification.

Classification-specific metric content over the shared ``aind_pophys_metadata`` QC
builders. Metric-producing functions *return* metrics; ``job.run``
accumulates them and writes ONE ``quality_control.json`` per plane
(the collector pattern).

v2 has no ``QCEvaluation``, so the grouping is carried by every metric's
``evaluation`` tag, matching ``QualityControl.default_grouping =
["evaluation"]``.
"""

from datetime import datetime as dt
from pathlib import Path
from typing import Any, List, Optional, Sequence

import pytz
from aind_data_schema.core.quality_control import (
    CurationMetric,
    QCMetric,
    QCStatus,
    QualityControl,
    Stage,
    Status,
)
from aind_data_schema_models.modalities import Modality
from aind_pophys_metadata import quality_control as shared_qc

_SEATTLE_TZ = pytz.timezone("America/Los_Angeles")

EVALUATION = "Classification"
METRIC_TYPE = "Operational QC"
CURATION_TYPE = "ROI soma/dendrite/border label"

REASONABLE_OPTIONS = ["Reasonable", "Unreasonable"]
REASONABLE_STATUS = [Status.PASS, Status.FAIL]

LABEL_SOMA = "soma"
LABEL_DENDRITE = "dendrite"
LABEL_BOTH = "soma+dendrite"
LABEL_NEITHER = "unclassified"
BORDER_SUFFIX = "|border"


def _automated_pass() -> QCStatus:
    """Return an automated PASS status stamped in US/Pacific.

    A metric with an empty ``status_history`` raises ``IndexError`` when
    ``QualityControl`` derives its status, so every metric seeds one.
    """
    return QCStatus(
        evaluator="Automated",
        status=Status.PASS,
        timestamp=dt.now(_SEATTLE_TZ).isoformat(),
    )


def _numeric_metric(
    name: str,
    value: Any,
    description: str,
    reference: Optional[str],
) -> QCMetric:
    """Build an automatically-derived numeric QCMetric.

    Parameters
    ----------
    name : str
        Metric name, already scoped to the plane.
    value : Any
        The measured value (a scalar or a small dict).
    description : str
        Reviewer-facing description.
    reference : str or None
        Figure path relative to the results root.

    Returns
    -------
    QCMetric
        The populated metric.
    """
    return QCMetric(
        name=name,
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        tags={"evaluation": EVALUATION, "type": METRIC_TYPE},
        description=description,
        reference=reference,
        status_history=[_automated_pass()],
        value=value,
    )


def build_classification_metric(plane_id: str, reference: str) -> QCMetric:
    """Reviewer dropdown over the combined-classifications figure.

    Parameters
    ----------
    plane_id : str
        Plane identifier used in the metric name.
    reference : str
        Path to the combined figure, relative to the results root.

    Returns
    -------
    QCMetric
        A pending dropdown metric.
    """
    return shared_qc.dropdown_metric(
        name=f"{plane_id} Classification",
        evaluation=EVALUATION,
        options=REASONABLE_OPTIONS,
        status=REASONABLE_STATUS,
        reference=reference,
        description="Classification of ROIs as soma or dendrite",
        metric_type=METRIC_TYPE,
    )


def build_border_roi_metric(plane_id: str, reference: str) -> QCMetric:
    """Reviewer dropdown over the border-ROI figure.

    Parameters
    ----------
    plane_id : str
        Plane identifier used in the metric name.
    reference : str
        Path to the border-ROI figure, relative to the results root.

    Returns
    -------
    QCMetric
        A pending dropdown metric.
    """
    return shared_qc.dropdown_metric(
        name=f"{plane_id} Border ROIs",
        evaluation=EVALUATION,
        options=REASONABLE_OPTIONS,
        status=REASONABLE_STATUS,
        reference=reference,
        description="ROIs that are on the border of the field of view",
        metric_type=METRIC_TYPE,
    )


def roi_labels(
    soma_predictions: Sequence,
    dendrite_predictions: Sequence,
    border_labels: Sequence,
) -> List[str]:
    """Combine the three per-ROI booleans into one label per ROI.

    The soma and dendrite classifiers are independent, so both-positive and
    neither-positive are both possible. List position is the ROI id: ROIs are
    positional indices in suite2p order, the same order as ``rois/coords`` in
    the extraction HDF5, and downstream consumers pair by position.

    Parameters
    ----------
    soma_predictions : sequence
        Per-ROI soma prediction, truthy for positive.
    dendrite_predictions : sequence
        Per-ROI dendrite prediction, truthy for positive.
    border_labels : sequence
        Per-ROI border flag, truthy when the ROI touches the border.

    Returns
    -------
    list of str
        One label per ROI, in ROI order.

    Raises
    ------
    ValueError
        If the three sequences are not the same length — a mismatch would
        silently misalign labels from ROIs.
    """
    lengths = {
        len(soma_predictions),
        len(dendrite_predictions),
        len(border_labels),
    }
    if len(lengths) != 1:
        raise ValueError(
            "soma_predictions, dendrite_predictions and border_labels must "
            f"be the same length; got {sorted(lengths)}. Per-ROI labels are "
            "paired by position, so a mismatch would misalign them."
        )

    labels = []
    for soma, dendrite, border in zip(soma_predictions, dendrite_predictions, border_labels):
        if soma and dendrite:
            label = LABEL_BOTH
        elif soma:
            label = LABEL_SOMA
        elif dendrite:
            label = LABEL_DENDRITE
        else:
            label = LABEL_NEITHER
        labels.append(f"{label}{BORDER_SUFFIX}" if border else label)
    return labels


def build_roi_label_curation_metric(
    plane_id: str,
    labels: List[str],
    reference: str,
) -> CurationMetric:
    """Per-ROI labels as a core v2 ``CurationMetric``.

    These are automatically-produced labels intended to be corrected by a
    human, which is exactly what ``CurationMetric`` plus ``curation_history``
    models. ``CurationMetric`` has no ``options`` field, so the vocabulary is
    stated in the description.

    Parameters
    ----------
    plane_id : str
        Plane identifier used in the metric name.
    labels : list of str
        One label per ROI, from :func:`roi_labels`.
    reference : str
        Path to the combined figure, relative to the results root.

    Returns
    -------
    CurationMetric
        The per-ROI label curation metric.
    """
    return CurationMetric(
        name=f"{plane_id} ROI classification labels",
        modality=Modality.POPHYS,
        stage=Stage.PROCESSING,
        type=CURATION_TYPE,
        value=list(labels),
        description=(
            "Per-ROI classification label, indexed by ROI in suite2p order. "
            f"One of '{LABEL_SOMA}', '{LABEL_DENDRITE}', '{LABEL_BOTH}' or "
            f"'{LABEL_NEITHER}', suffixed '{BORDER_SUFFIX}' when the ROI "
            "touches the field-of-view border."
        ),
        reference=reference,
        tags={"evaluation": EVALUATION, "type": METRIC_TYPE},
        status_history=[_automated_pass()],
    )


def build_roi_count_metric(plane_id: str, num_rois: int, reference: str) -> QCMetric:
    """Total ROI count for the plane."""
    return _numeric_metric(
        name=f"{plane_id} ROI count",
        value=int(num_rois),
        description="Number of ROIs classified in this plane",
        reference=reference,
    )


def build_positive_count_metric(
    plane_id: str,
    label: str,
    count: int,
    num_rois: int,
    reference: str,
) -> QCMetric:
    """Count and fraction of ROIs a classifier called positive.

    Parameters
    ----------
    plane_id : str
        Plane identifier used in the metric name.
    label : str
        ``"Soma"`` or ``"Dendrite"``.
    count : int
        Number of positive ROIs.
    num_rois : int
        Total ROI count, used for the fraction.
    reference : str
        Figure path relative to the results root.

    Returns
    -------
    QCMetric
        The populated metric.
    """
    return _numeric_metric(
        name=f"{plane_id} {label}-positive ROIs",
        value={
            "count": int(count),
            "fraction": (float(count) / num_rois) if num_rois else 0.0,
        },
        description=f"ROIs classified as {label.lower()}",
        reference=reference,
    )


def build_border_count_metric(
    plane_id: str,
    count: int,
    border_size: int,
    reference: str,
) -> QCMetric:
    """Border-ROI count, recording the border width that produced it."""
    return _numeric_metric(
        name=f"{plane_id} Border ROI count",
        value={"count": int(count), "border_size_px": int(border_size)},
        description=("ROIs within border_size pixels of the field-of-view edge"),
        reference=reference,
    )


def build_quality_control(metrics: List[QCMetric]) -> QualityControl:
    """Wrap metrics into a full QualityControl with the standard grouping."""
    return shared_qc.build_quality_control(metrics)


def write_quality_control_json(
    quality_control: QualityControl,
    output_dir: Path,
) -> Path:
    """Serialize a QualityControl object to ``quality_control.json``.

    Parameters
    ----------
    quality_control : QualityControl
        The QC document to write.
    output_dir : Path
        Directory into which ``quality_control.json`` is written.

    Returns
    -------
    Path
        The path written.
    """
    return shared_qc.write_quality_control_json(quality_control, output_dir)
