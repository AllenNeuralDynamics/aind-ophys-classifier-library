"""Tests for quality_control.json generation."""

import tempfile
import unittest
from pathlib import Path

from aind_data_schema.core.quality_control import (
    CurationMetric,
    QualityControl,
)

from aind_ophys_classifier_library import qc

PLANE = "VISp_0"
COMBINED = "VISp_0/classification/VISp_0_combined_classifications.png"
BORDER = "VISp_0/classification/VISp_0_border_rois.png"


class TestRoiLabels(unittest.TestCase):
    """The per-ROI label vocabulary."""

    def test_all_four_combinations(self):
        """soma, dendrite, both and neither are all distinguishable."""
        labels = qc.roi_labels(
            soma_predictions=[1, 0, 1, 0],
            dendrite_predictions=[0, 1, 1, 0],
            border_labels=[False, False, False, False],
        )
        self.assertEqual(
            labels,
            [
                qc.LABEL_SOMA,
                qc.LABEL_DENDRITE,
                qc.LABEL_BOTH,
                qc.LABEL_NEITHER,
            ],
        )

    def test_border_suffix(self):
        """A border ROI keeps its class label and gains a suffix."""
        labels = qc.roi_labels([1], [0], [True])
        self.assertEqual(labels, [qc.LABEL_SOMA + qc.BORDER_SUFFIX])

    def test_length_mismatch_raises(self):
        """Mismatched lengths would misalign labels from ROIs."""
        with self.assertRaises(ValueError) as ctx:
            qc.roi_labels([1, 0], [1], [False, False])
        self.assertIn("same length", str(ctx.exception))

    def test_empty_is_allowed(self):
        """A plane with no ROIs yields no labels rather than raising."""
        self.assertEqual(qc.roi_labels([], [], []), [])


class TestDropdownMetrics(unittest.TestCase):
    """The two reviewer sign-off dropdowns carried over from v1."""

    def test_classification_metric(self):
        """References the net-new combined figure."""
        metric = qc.build_classification_metric(PLANE, COMBINED)
        self.assertEqual(metric.name, f"{PLANE} Classification")
        self.assertEqual(metric.reference, COMBINED)
        self.assertEqual(metric.tags["evaluation"], qc.EVALUATION)
        self.assertTrue(metric.status_history)

    def test_border_metric(self):
        """References the border figure and shares the evaluation group."""
        metric = qc.build_border_roi_metric(PLANE, BORDER)
        self.assertEqual(metric.name, f"{PLANE} Border ROIs")
        self.assertEqual(metric.tags["evaluation"], qc.EVALUATION)

    def test_dropdown_value_is_a_string(self):
        """DropdownMetric.value is a string, empty when unevaluated."""
        metric = qc.build_classification_metric(PLANE, COMBINED)
        self.assertIsInstance(metric.value.value, str)


class TestCurationMetric(unittest.TestCase):
    """The per-ROI labels as the core v2 CurationMetric."""

    def _metric(self):
        """Build a curation metric over three ROIs."""
        labels = qc.roi_labels([1, 0, 1], [0, 1, 1], [False, True, False])
        return qc.build_roi_label_curation_metric(PLANE, labels, COMBINED)

    def test_is_core_curation_metric(self):
        """The core class is used; qcportal no longer exports one."""
        self.assertIsInstance(self._metric(), CurationMetric)

    def test_required_v2_fields(self):
        """``type`` is set and ``value`` is a list, as v2 requires."""
        metric = self._metric()
        self.assertEqual(metric.type, qc.CURATION_TYPE)
        self.assertIsInstance(metric.value, list)
        self.assertEqual(len(metric.value), 3)

    def test_status_history_seeded(self):
        """An empty status_history breaks QualityControl status derivation."""
        self.assertTrue(self._metric().status_history)

    def test_description_states_the_vocabulary(self):
        """CurationMetric has no options field, so the description must."""
        description = self._metric().description
        for label in (qc.LABEL_SOMA, qc.LABEL_DENDRITE, qc.LABEL_NEITHER):
            self.assertIn(label, description)


class TestNumericMetrics(unittest.TestCase):
    """The scalar metrics the v1 capsule discarded."""

    def test_roi_count(self):
        """ROI count is an int value."""
        metric = qc.build_roi_count_metric(PLANE, 42, COMBINED)
        self.assertEqual(metric.value, 42)

    def test_positive_count_and_fraction(self):
        """Count and fraction are both recorded."""
        metric = qc.build_positive_count_metric(PLANE, "Soma", 5, 20, COMBINED)
        self.assertEqual(metric.value["count"], 5)
        self.assertAlmostEqual(metric.value["fraction"], 0.25)

    def test_positive_fraction_with_zero_rois(self):
        """A zero ROI count must not divide by zero."""
        metric = qc.build_positive_count_metric(PLANE, "Soma", 0, 0, COMBINED)
        self.assertEqual(metric.value["fraction"], 0.0)

    def test_border_count_records_the_parameter(self):
        """border_size is recorded so the count is auditable."""
        metric = qc.build_border_count_metric(PLANE, 3, 10, BORDER)
        self.assertEqual(metric.value["count"], 3)
        self.assertEqual(metric.value["border_size_px"], 10)


class TestQualityControlDocument(unittest.TestCase):
    """The full document, its grouping and its writer."""

    def _metrics(self):
        """Build the whole per-plane metric set."""
        labels = qc.roi_labels([1, 0], [0, 1], [False, True])
        return [
            qc.build_classification_metric(PLANE, COMBINED),
            qc.build_border_roi_metric(PLANE, BORDER),
            qc.build_roi_label_curation_metric(PLANE, labels, COMBINED),
            qc.build_roi_count_metric(PLANE, 2, COMBINED),
            qc.build_positive_count_metric(PLANE, "Soma", 1, 2, COMBINED),
            qc.build_positive_count_metric(PLANE, "Dendrite", 1, 2, COMBINED),
            qc.build_border_count_metric(PLANE, 1, 10, BORDER),
        ]

    def test_all_metrics_share_one_evaluation_group(self):
        """One group keeps the reviewer view equivalent to v1's evaluation."""
        groups = {m.tags["evaluation"] for m in self._metrics()}
        self.assertEqual(groups, {qc.EVALUATION})

    def test_build_uses_project_default_grouping(self):
        """The shared standard grouping is applied."""
        document = qc.build_quality_control(self._metrics())
        self.assertIsInstance(document, QualityControl)
        self.assertEqual(document.default_grouping, ["evaluation"])

    def test_write_and_round_trip(self):
        """The written file validates back into a core v2 QualityControl."""
        document = qc.build_quality_control(self._metrics())
        with tempfile.TemporaryDirectory() as tmp:
            written = qc.write_quality_control_json(document, Path(tmp))
            self.assertTrue(written.exists())
            reloaded = QualityControl.model_validate_json(written.read_text())
        self.assertEqual(len(reloaded.metrics), 7)
        curation = [m for m in reloaded.metrics if isinstance(m, CurationMetric)]
        self.assertEqual(len(curation), 1)


if __name__ == "__main__":
    unittest.main()
