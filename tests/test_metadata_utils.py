"""Tests for the v1/v2 metadata read layer and processing.json builders.

The um/pixel readers carry the most risk in this library: ROInet silently
assumes 1.0 um/pixel when the value is missing, which changes every
prediction without erroring, so absence must raise.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime as dt
from pathlib import Path
from unittest.mock import patch

from aind_data_schema.core.processing import DataProcess, Processing
from aind_data_schema_models.process_names import ProcessName

from aind_ophys_classifier_library.utils import metadata_utils as mu
from tests import fixtures


def _write(directory: Path, name: str, payload: dict) -> None:
    """Write ``payload`` as JSON into ``directory/name``."""
    (directory / name).write_text(json.dumps(payload))


class TestUmPerPixelV1(unittest.TestCase):
    """v1 reads fov_scale_factor with a pint-parsed unit."""

    def test_reads_scale_factor(self):
        """The FOV scale factor is returned in um/pixel."""
        value = mu.get_um_per_pixel_v1(fixtures.v1_session())
        self.assertAlmostEqual(value, fixtures.UM_PER_PIXEL)

    def test_missing_scale_factor_returns_none(self):
        """A FOV with no scale factor yields None, not a guess."""
        self.assertIsNone(mu.get_um_per_pixel_v1(fixtures.v1_session(scale_factor=None)))

    def test_missing_unit_assumes_microns_per_pixel(self):
        """An absent unit string is treated as um/pixel."""
        value = mu.get_um_per_pixel_v1(fixtures.v1_session(unit=None))
        self.assertAlmostEqual(value, fixtures.UM_PER_PIXEL)

    def test_no_data_streams_returns_none(self):
        """An empty session yields None."""
        self.assertIsNone(mu.get_um_per_pixel_v1({}))


class TestUmPerPixelV2(unittest.TestCase):
    """v2 reads the Scale transform, never the pixel dimensions."""

    def test_reads_transform_scale(self):
        """The Scale entry of image_to_acquisition_transform is used."""
        value = mu.get_um_per_pixel_v2(fixtures.v2_acquisition())
        self.assertAlmostEqual(value, fixtures.UM_PER_PIXEL)

    def test_ignores_pixel_dimensions(self):
        """``dimensions`` is also a Scale; reading it would give 512."""
        value = mu.get_um_per_pixel_v2(fixtures.v2_acquisition())
        self.assertNotEqual(value, 512)

    def test_translation_only_returns_none(self):
        """A transform list with no Scale yields None, not a guess.

        This is what the canonical multiplane acquisition example emits.
        """
        self.assertIsNone(mu.get_um_per_pixel_v2(fixtures.v2_acquisition(scale=None)))

    def test_absent_axis_unit_is_treated_as_microns(self):
        """No coordinate_system means the scale is already um/pixel."""
        value = mu.get_um_per_pixel_v2(fixtures.v2_acquisition(axis_unit=None))
        self.assertAlmostEqual(value, fixtures.UM_PER_PIXEL)

    def test_millimetre_axis_unit_is_converted(self):
        """A non-micrometre axis unit is converted, not assumed."""
        value = mu.get_um_per_pixel_v2(fixtures.v2_acquisition(scale=2.0, axis_unit="millimeter"))
        self.assertAlmostEqual(value, 2000.0)

    def test_no_data_streams_returns_none(self):
        """An empty acquisition yields None."""
        self.assertIsNone(mu.get_um_per_pixel_v2({}))

    def test_skips_non_imaging_configurations(self):
        """A data stream carrying only other config types is skipped."""
        acquisition = {"data_streams": [{"configurations": [{"object_type": "Patch cord config"}]}]}
        self.assertIsNone(mu.get_um_per_pixel_v2(acquisition))

    def test_skips_non_planar_images(self):
        """An imaging config whose images are not planar is skipped."""
        acquisition = fixtures.v2_acquisition()
        image = acquisition["data_streams"][0]["configurations"][0]["images"][0]
        image["object_type"] = "Some other image"
        self.assertIsNone(mu.get_um_per_pixel_v2(acquisition))

    def test_skips_scale_with_empty_list(self):
        """A Scale transform with no magnitudes is not a usable value."""
        acquisition = fixtures.v2_acquisition()
        transforms = acquisition["data_streams"][0]["configurations"][0]["images"][0][
            "image_to_acquisition_transform"
        ]
        for transform in transforms:
            if transform["object_type"] == mu._V2_SCALE_TYPE:
                transform["scale"] = []
        self.assertIsNone(mu.get_um_per_pixel_v2(acquisition))


class TestLoadClassifierMetadata(unittest.TestCase):
    """Version dispatch by file presence, and fail-fast on missing um/px."""

    def test_reads_v1_session(self):
        """A v1 session supplies um/pixel and the identifiers."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "session.json", fixtures.v1_session())
            _write(root, "data_description.json", fixtures.data_description())
            metadata = mu.load_classifier_metadata(root)
        self.assertAlmostEqual(metadata.um_per_pixel, fixtures.UM_PER_PIXEL)
        self.assertEqual(metadata.um_per_pixel_source, "session.json")
        self.assertEqual(metadata.instrument_id, "MESO.1")
        self.assertEqual(metadata.dataset_name, fixtures.DATASET_NAME)
        self.assertEqual(metadata.fov_ids, ("plane_0",))

    def test_reads_v2_acquisition(self):
        """A v2 acquisition supplies the same fields from v2 paths."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "acquisition.json", fixtures.v2_acquisition())
            metadata = mu.load_classifier_metadata(root)
        self.assertAlmostEqual(metadata.um_per_pixel, fixtures.UM_PER_PIXEL)
        self.assertEqual(metadata.um_per_pixel_source, "acquisition.json")
        self.assertEqual(metadata.subject_id, fixtures.SUBJECT_ID)

    def test_override_wins_and_is_recorded(self):
        """An explicit value bypasses the metadata and is labelled as such."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "acquisition.json", fixtures.v2_acquisition())
            metadata = mu.load_classifier_metadata(root, um_per_pixel_override=0.8)
        self.assertEqual(metadata.um_per_pixel, 0.8)
        self.assertEqual(metadata.um_per_pixel_source, mu.SOURCE_OVERRIDE)

    def test_missing_um_per_pixel_fails_loudly(self):
        """Absence raises and names both schema locations and the override."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "acquisition.json", fixtures.v2_acquisition(scale=None))
            with self.assertRaises(ValueError) as ctx:
                mu.load_classifier_metadata(root)
        message = str(ctx.exception)
        self.assertIn("fov_scale_factor", message)
        self.assertIn("image_to_acquisition_transform", message)
        self.assertIn(mu.SOURCE_OVERRIDE, message)
        self.assertIn("1.0", message)

    def test_v1_missing_um_per_pixel_fails_loudly(self):
        """The v1 path fails the same way."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "session.json", fixtures.v1_session(scale_factor=None))
            with self.assertRaises(ValueError):
                mu.load_classifier_metadata(root)

    def test_subject_json_fallback(self):
        """subject.json is consulted only when the core file omits the id."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            acquisition = fixtures.v2_acquisition()
            del acquisition["subject_id"]
            _write(root, "acquisition.json", acquisition)
            _write(root, "subject.json", {"subject_id": "NY129"})
            metadata = mu.load_classifier_metadata(root)
        self.assertEqual(metadata.subject_id, "NY129")

    def test_no_core_file_raises(self):
        """Neither core file present is a hard error."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                mu.load_classifier_metadata(Path(tmp))

    def test_both_core_files_is_ambiguous(self):
        """Both core files present must not be guessed."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "session.json", fixtures.v1_session())
            _write(root, "acquisition.json", fixtures.v2_acquisition())
            with self.assertRaises(ValueError):
                mu.load_classifier_metadata(root)


class TestBuildDataProcess(unittest.TestCase):
    """The capsule's DataProcess and the Processing document around it."""

    def _build(self, **kwargs):
        """Build a DataProcess with sensible defaults."""
        now = dt.now()
        return mu.build_data_process(
            parameters=kwargs.pop("parameters", {"border_size": 10}),
            start_time=now,
            end_time=now,
            **kwargs,
        )

    def test_process_type_is_cell_classification(self):
        """The v2 ProcessName member for this capsule is used."""
        process = self._build()
        self.assertIsInstance(process, DataProcess)
        self.assertEqual(process.process_type, ProcessName.IMAGE_CELL_CLASSIFICATION)

    def test_parameters_and_input_data_recorded(self):
        """Run parameters and input provenance land on the Code block."""
        process = self._build(input_data=["plane_0/extraction/plane_0_extraction.h5"])
        self.assertEqual(process.code.parameters.model_dump()["border_size"], 10)
        self.assertEqual(
            process.code.input_data[0].name,
            "plane_0/extraction/plane_0_extraction.h5",
        )

    def test_output_path_recorded(self):
        """output_path carries the results-relative subpath."""
        process = self._build(output_path="plane_0/classification")
        self.assertEqual(str(process.output_path), "plane_0/classification")

    def test_error_recorded_in_notes(self):
        """A failed run records its exception in notes."""
        process = self._build(error=RuntimeError("boom"))
        self.assertIn("RuntimeError", process.notes)

    def test_no_notes_on_success(self):
        """A successful run leaves notes unset."""
        self.assertIsNone(self._build().notes)

    def test_experimenters_is_never_an_institution(self):
        """experimenters is a list of people; the default is empty."""
        self.assertEqual(self._build().experimenters, [])


class TestProcessingDocument(unittest.TestCase):
    """Processing.pipelines linkage and the writer."""

    def test_no_pipeline_env_omits_pipelines(self):
        """Outside the pipeline both pipelines and pipeline_name stay unset."""
        with patch.dict(os.environ, {}, clear=False):
            for key in ("PIPELINE_NAME", "PIPELINE_URL", "PIPELINE_VERSION"):
                os.environ.pop(key, None)
            now = dt.now()
            process = mu.build_data_process(parameters={}, start_time=now, end_time=now)
            document = mu.build_processing([process])
        self.assertIsNone(document.pipelines)
        self.assertIsNone(process.pipeline_name)

    def test_pipeline_env_populates_both_together(self):
        """The schema requires pipeline_name to resolve within pipelines."""
        env = {
            "PIPELINE_NAME": "aind-pophys-pipeline",
            "PIPELINE_URL": "https://example.org/pipeline",
            "PIPELINE_VERSION": "2.0.0",
        }
        with patch.dict(os.environ, env):
            now = dt.now()
            process = mu.build_data_process(parameters={}, start_time=now, end_time=now)
            document = mu.build_processing([process])
        self.assertEqual(process.pipeline_name, "aind-pophys-pipeline")
        self.assertEqual(
            [code.name for code in document.pipelines],
            ["aind-pophys-pipeline"],
        )

    def test_write_and_round_trip(self):
        """The written file validates back into a core v2 Processing."""
        now = dt.now()
        process = mu.build_data_process(
            parameters={"border_size": 10},
            start_time=now,
            end_time=now,
            output_path="plane_0/classification",
        )
        document = mu.build_processing([process])
        with tempfile.TemporaryDirectory() as tmp:
            written = mu.write_processing_json(document, Path(tmp))
            self.assertTrue(written.exists())
            reloaded = Processing.model_validate_json(written.read_text())
        self.assertEqual(len(reloaded.data_processes), 1)


if __name__ == "__main__":
    unittest.main()
