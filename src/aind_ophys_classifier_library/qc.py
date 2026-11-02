"""Classification QC metrics and the ``quality_control.json`` writer."""

from datetime import datetime as dt
from pathlib import Path
from typing import List

import pytz
from aind_data_schema.core.quality_control import (
    QCMetric,
    QCStatus,
    QualityControl,
    Status,
)
from aind_pophys_metadata import quality_control as shared_qc

_SEATTLE_TZ = pytz.timezone("America/Los_Angeles")

EVALUATION = "Classification"
METRIC_TYPE = "Informational"

REASONABLE_OPTIONS = ["Reasonable", "Unreasonable"]
REASONABLE_STATUS = [Status.PASS, Status.FAIL]


def _automated_pass() -> QCStatus:
    """Return an automated PASS status stamped in US/Pacific."""
    return QCStatus(
        evaluator="Automated",
        status=Status.PASS,
        timestamp=dt.now(_SEATTLE_TZ).isoformat(),
    )


def _passed_dropdown(name: str, reference: str, description: str) -> QCMetric:
    """Build a Reasonable/Unreasonable dropdown passed by automation.

    The passing option is selected and the status history holds one
    automated PASS, so the metric needs no manual review.

    Parameters
    ----------
    name : str
        Metric name, already scoped to the plane.
    reference : str
        Figure path relative to the results root.
    description : str
        Reviewer-facing description.

    Returns
    -------
    QCMetric
        The passed dropdown metric.
    """
    pending = shared_qc.dropdown_metric(
        name=name,
        evaluation=EVALUATION,
        options=REASONABLE_OPTIONS,
        status=REASONABLE_STATUS,
        reference=reference,
        description=description,
        metric_type=METRIC_TYPE,
    )
    passing_option = REASONABLE_OPTIONS[REASONABLE_STATUS.index(Status.PASS)]
    return pending.model_copy(
        update={
            "value": pending.value.model_copy(update={"value": passing_option}),
            "status_history": [_automated_pass()],
        }
    )


def build_classification_metric(plane_id: str, reference: str) -> QCMetric:
    """Passed dropdown over the combined-classifications figure.

    Parameters
    ----------
    plane_id : str
        Plane identifier used in the metric name.
    reference : str
        Path to the combined figure, relative to the results root.

    Returns
    -------
    QCMetric
        An automatically passed dropdown metric.
    """
    return _passed_dropdown(
        name=f"{plane_id} Classification",
        reference=reference,
        description="Classification of ROIs as soma and/or dendrite.",
    )


def build_border_roi_metric(plane_id: str, reference: str) -> QCMetric:
    """Passed dropdown over the border-ROI figure.

    Parameters
    ----------
    plane_id : str
        Plane identifier used in the metric name.
    reference : str
        Path to the border-ROI figure, relative to the results root.

    Returns
    -------
    QCMetric
        An automatically passed dropdown metric.
    """
    return _passed_dropdown(
        name=f"{plane_id} Border ROIs",
        reference=reference,
        description="ROIs on or near the border of the field of view.",
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
