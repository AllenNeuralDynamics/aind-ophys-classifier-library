"""Tests for quality_control.json generation."""

import tempfile
import unittest
from pathlib import Path
from typing import List

from aind_data_schema.core.quality_control import QCMetric, QualityControl, Status

from aind_ophys_classifier_library import qc

PLANE = "VISp_0"
COMBINED = "VISp_0/classification/VISp_0_combined_classifications.png"
BORDER = "VISp_0/classification/VISp_0_border_rois.png"


class TestDropdownMetrics(unittest.TestCase):
    """The two reviewer sign-off dropdowns."""

    def test_classification_metric(self) -> None:
        """References the combined figure and is informational."""
        metric = qc.build_classification_metric(PLANE, COMBINED)
        self.assertEqual(metric.name, f"{PLANE} Classification")
        self.assertEqual(metric.reference, COMBINED)
        self.assertEqual(metric.tags["evaluation"], qc.EVALUATION)
        self.assertEqual(metric.tags["type"], "Informational")
        self.assertEqual(metric.description, "Classification of ROIs as soma and/or dendrite.")
        self.assertTrue(metric.status_history)

    def test_border_metric(self) -> None:
        """References the border figure and shares the evaluation group."""
        metric = qc.build_border_roi_metric(PLANE, BORDER)
        self.assertEqual(metric.name, f"{PLANE} Border ROIs")
        self.assertEqual(metric.tags["evaluation"], qc.EVALUATION)
        self.assertEqual(metric.tags["type"], "Informational")
        self.assertEqual(metric.description, "ROIs on or near the border of the field of view.")

    def test_dropdowns_are_passed_by_automation(self) -> None:
        """Both dropdowns select the passing option with one automated PASS."""
        for metric in (
            qc.build_classification_metric(PLANE, COMBINED),
            qc.build_border_roi_metric(PLANE, BORDER),
        ):
            with self.subTest(metric=metric.name):
                self.assertEqual(metric.value.value, "Reasonable")
                self.assertEqual(metric.value.options, qc.REASONABLE_OPTIONS)
                self.assertEqual(metric.value.status, qc.REASONABLE_STATUS)
                self.assertEqual(len(metric.status_history), 1)
                self.assertEqual(metric.status_history[0].status, Status.PASS)
                self.assertEqual(metric.status_history[0].evaluator, "Automated")


class TestQualityControlDocument(unittest.TestCase):
    """The full document, its grouping and its writer."""

    def _metrics(self) -> List[QCMetric]:
        """Build the whole per-plane metric set."""
        return [
            qc.build_classification_metric(PLANE, COMBINED),
            qc.build_border_roi_metric(PLANE, BORDER),
        ]

    def test_all_metrics_share_one_evaluation_group(self) -> None:
        """Every metric carries the same evaluation tag."""
        groups = {m.tags["evaluation"] for m in self._metrics()}
        self.assertEqual(groups, {qc.EVALUATION})

    def test_all_metrics_are_informational(self) -> None:
        """The whole group is typed Informational."""
        types = {m.tags["type"] for m in self._metrics()}
        self.assertEqual(types, {"Informational"})

    def test_every_metric_is_passed_by_automation(self) -> None:
        """Informational metrics never wait for manual review."""
        for metric in self._metrics():
            with self.subTest(metric=metric.name):
                self.assertEqual(metric.status_history[-1].status, Status.PASS)
                self.assertEqual(metric.status_history[-1].evaluator, "Automated")

    def test_build_uses_project_default_grouping(self) -> None:
        """The shared standard grouping is applied."""
        document = qc.build_quality_control(self._metrics())
        self.assertIsInstance(document, QualityControl)
        self.assertEqual(document.default_grouping, ["evaluation"])

    def test_write_and_round_trip(self) -> None:
        """The written file validates back into a core v2 QualityControl."""
        document = qc.build_quality_control(self._metrics())
        with tempfile.TemporaryDirectory() as tmp:
            written = qc.write_quality_control_json(document, Path(tmp))
            self.assertTrue(written.exists())
            reloaded = QualityControl.model_validate_json(written.read_text())
        self.assertEqual(
            [m.name for m in reloaded.metrics],
            [f"{PLANE} Classification", f"{PLANE} Border ROIs"],
        )


if __name__ == "__main__":
    unittest.main()
