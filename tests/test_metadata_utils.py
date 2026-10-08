"""Tests for the metadata read layer and the classification stage."""

import json
import os
import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path
from typing import Any
from unittest.mock import patch

from aind_data_schema.core.processing import Processing
from aind_data_schema_models.process_names import ProcessName

from aind_ophys_classifier_library.utils import metadata_utils as mu
from tests import fixtures

LIBRARY = "aind-ophys-classifier-library"


def _write(directory: Path, name: str, payload: dict) -> None:
    """Write ``payload`` as JSON into ``directory/name``."""
    (directory / name).write_text(json.dumps(payload))


class TestLoadClassifierMetadata(unittest.TestCase):
    """The fields are read, and missing um/px fails fast."""

    def test_reads_v1_session(self) -> None:
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

    def test_reads_v2_acquisition(self) -> None:
        """A v2 acquisition supplies the same fields from v2 paths."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "acquisition.json", fixtures.v2_acquisition())
            metadata = mu.load_classifier_metadata(root)
        self.assertAlmostEqual(metadata.um_per_pixel, fixtures.UM_PER_PIXEL)
        self.assertEqual(metadata.um_per_pixel_source, "acquisition.json")
        self.assertEqual(metadata.subject_id, fixtures.SUBJECT_ID)

    def test_override_wins_and_is_recorded(self) -> None:
        """An explicit value bypasses the metadata and is labelled as such."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "acquisition.json", fixtures.v2_acquisition())
            metadata = mu.load_classifier_metadata(root, um_per_pixel_override=0.8)
        self.assertEqual(metadata.um_per_pixel, 0.8)
        self.assertEqual(metadata.um_per_pixel_source, mu.SOURCE_OVERRIDE)

    def test_reads_minimal_metadata(self) -> None:
        """A flat minimal metadata.json supplies the same fields."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "metadata.json", fixtures.minimal_metadata())
            metadata = mu.load_classifier_metadata(root)
        self.assertAlmostEqual(metadata.um_per_pixel, fixtures.UM_PER_PIXEL)
        self.assertEqual(metadata.um_per_pixel_source, "metadata.json")
        self.assertEqual(metadata.instrument_id, "MESO.1")

    def test_reads_the_transform_scale_not_pixel_dimensions(self) -> None:
        """v2 ``dimensions`` is also a Scale; reading it would give 512."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "acquisition.json", fixtures.v2_acquisition())
            metadata = mu.load_classifier_metadata(root)
        self.assertAlmostEqual(metadata.um_per_pixel, fixtures.UM_PER_PIXEL)
        self.assertNotEqual(metadata.um_per_pixel, 512)

    def test_non_micrometre_unit_is_converted(self) -> None:
        """A millimetre axis unit is converted, not assumed."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root,
                "acquisition.json",
                fixtures.v2_acquisition(scale=2.0, axis_unit="millimeter"),
            )
            metadata = mu.load_classifier_metadata(root)
        self.assertAlmostEqual(metadata.um_per_pixel, 2000.0)

    def test_absent_unit_is_read_as_micrometres(self) -> None:
        """No coordinate_system means the scale is already um/pixel."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "acquisition.json", fixtures.v2_acquisition(axis_unit=None))
            metadata = mu.load_classifier_metadata(root)
        self.assertAlmostEqual(metadata.um_per_pixel, fixtures.UM_PER_PIXEL)

    def test_missing_um_per_pixel_fails_loudly(self) -> None:
        """Absence names the core file, the schema version and the override."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "acquisition.json", fixtures.v2_acquisition(scale=None))
            with self.assertRaises(ValueError) as ctx:
                mu.load_classifier_metadata(root)
        message = str(ctx.exception)
        self.assertIn("acquisition.json", message)
        self.assertIn("v2", message)
        self.assertIn(mu.SOURCE_OVERRIDE, message)
        self.assertIn("1.0", message)

    def test_v1_missing_um_per_pixel_fails_loudly(self) -> None:
        """The v1 path fails the same way."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "session.json", fixtures.v1_session(scale_factor=None))
            with self.assertRaises(ValueError) as ctx:
                mu.load_classifier_metadata(root)
        self.assertIn("session.json", str(ctx.exception))

    def test_minimal_missing_um_per_pixel_fails_loudly(self) -> None:
        """The minimal path fails the same way."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "metadata.json", fixtures.minimal_metadata(None))
            with self.assertRaises(ValueError):
                mu.load_classifier_metadata(root)

    def test_unconvertible_unit_fails_loudly(self) -> None:
        """An unrecognised unit reaches the raise instead of being assumed."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root,
                "acquisition.json",
                fixtures.v2_acquisition(axis_unit="furlong"),
            )
            with self.assertLogs(level="WARNING"):
                with self.assertRaises(ValueError) as ctx:
                    mu.load_classifier_metadata(root)
        self.assertIn("unit", str(ctx.exception))

    def test_override_wins_over_an_unconvertible_unit(self) -> None:
        """The override is the documented escape from an unreadable scale."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root,
                "acquisition.json",
                fixtures.v2_acquisition(axis_unit="furlong"),
            )
            metadata = mu.load_classifier_metadata(root, um_per_pixel_override=0.8)
        self.assertEqual(metadata.um_per_pixel, 0.8)
        self.assertEqual(metadata.um_per_pixel_source, mu.SOURCE_OVERRIDE)

    def test_subject_json_fallback(self) -> None:
        """subject.json is consulted only when the core file omits the id."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            acquisition = fixtures.v2_acquisition()
            del acquisition["subject_id"]
            _write(root, "acquisition.json", acquisition)
            _write(root, "subject.json", {"subject_id": "NY129"})
            metadata = mu.load_classifier_metadata(root)
        self.assertEqual(metadata.subject_id, "NY129")

    def test_no_core_file_raises(self) -> None:
        """Neither core file present is a hard error."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                mu.load_classifier_metadata(Path(tmp))

    def test_both_core_files_is_ambiguous(self) -> None:
        """Both core files present must not be guessed."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "session.json", fixtures.v1_session())
            _write(root, "acquisition.json", fixtures.v2_acquisition())
            with self.assertRaises(ValueError):
                mu.load_classifier_metadata(root)


class TestBuildClassificationCode(unittest.TestCase):
    """The v2 Code block identifying classification runs."""

    def test_identity_is_the_installed_library(self) -> None:
        """name, version and url all describe this library's package."""
        code = mu.build_classification_code({})
        self.assertEqual(code.name, LIBRARY)
        self.assertEqual(code.version, version(LIBRARY))
        self.assertTrue(code.version)
        self.assertEqual(code.url, f"https://github.com/AllenNeuralDynamics/{LIBRARY}")

    def test_parameters_and_input_data_recorded(self) -> None:
        """Run parameters and input provenance land on the Code block."""
        code = mu.build_classification_code(
            {"border_size": 10},
            input_data=["plane_0/extraction/plane_0_extraction.h5"],
        )
        self.assertEqual(code.parameters.model_dump()["border_size"], 10)
        self.assertEqual(
            code.input_data[0].name,
            "plane_0/extraction/plane_0_extraction.h5",
        )


class TestProcessName(unittest.TestCase):
    """The name the shared builder composes, reproduced for exclusion."""

    def test_plane_id_qualifies_the_name(self) -> None:
        """A per-plane run's name is unique once documents are merged."""
        self.assertEqual(
            mu.process_name("VISp_0"),
            f"VISp_0: {ProcessName.IMAGE_CELL_CLASSIFICATION.value}",
        )

    def test_without_plane_id_the_process_type_label_is_used(self) -> None:
        """A run that never reached a plane uses the schema's own label."""
        self.assertEqual(
            mu.process_name(),
            ProcessName.IMAGE_CELL_CLASSIFICATION.value,
        )


class TestClassificationStage(unittest.TestCase):
    """The stage guard writes processing.json on both paths."""

    def _run_stage(self, root: Path, raises: bool, **kwargs: Any) -> Path:
        """Run one stage, optionally raising inside it."""
        code = mu.build_classification_code({"border_size": 10})
        try:
            with mu.classification_stage(root, code, **kwargs) as stage:
                stage.output_parameters = {"num_rois": 3}
                if raises:
                    raise RuntimeError("boom")
        except RuntimeError:
            pass
        return root / "processing.json"

    def test_success_writes_a_valid_document(self) -> None:
        """The written file validates back into a core v2 Processing."""
        with tempfile.TemporaryDirectory() as tmp:
            written = self._run_stage(Path(tmp), raises=False, plane_id="VISp_0")
            document = Processing.model_validate_json(written.read_text())
        process = document.data_processes[0]
        self.assertEqual(process.process_type, ProcessName.IMAGE_CELL_CLASSIFICATION)
        self.assertEqual(process.name, mu.process_name("VISp_0"))
        self.assertEqual(process.output_parameters.model_dump()["num_rois"], 3)
        self.assertIsNone(process.notes)
        self.assertEqual(process.experimenters, [])

    def test_failure_still_writes_the_document_with_the_error(self) -> None:
        """A raising stage leaves a document naming what stopped it."""
        with tempfile.TemporaryDirectory() as tmp:
            written = self._run_stage(Path(tmp), raises=True)
            document = Processing.model_validate_json(written.read_text())
        self.assertIn("boom", document.data_processes[0].notes)

    def test_upstream_documents_become_flat_siblings(self) -> None:
        """Every upstream processing.json in the input dir is a dependency."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "data" / "VISp_0" / "extraction"
            input_dir.mkdir(parents=True)
            _write(
                input_dir,
                "processing.json",
                {"data_processes": [{"name": "VISp_0: Image segmentation"}]},
            )
            written = self._run_stage(
                root / "results",
                raises=False,
                plane_id="VISp_0",
                input_dir=root / "data",
            )
            document = Processing.model_validate_json(written.read_text())
        self.assertEqual(
            document.dependency_graph,
            {mu.process_name("VISp_0"): ["VISp_0: Image segmentation"]},
        )

    def test_own_document_is_never_its_own_dependency(self) -> None:
        """A step's own name is excluded from the collected upstreams."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "data"
            input_dir.mkdir()
            _write(
                input_dir,
                "processing.json",
                {"data_processes": [{"name": mu.process_name("VISp_0")}]},
            )
            written = self._run_stage(
                root / "results",
                raises=False,
                plane_id="VISp_0",
                input_dir=input_dir,
            )
            document = Processing.model_validate_json(written.read_text())
        self.assertEqual(document.dependency_graph, {mu.process_name("VISp_0"): []})

    def test_no_pipeline_env_omits_pipelines(self) -> None:
        """Outside the pipeline both pipelines and pipeline_name stay unset."""
        with patch.dict(os.environ, {}, clear=False):
            for key in ("PIPELINE_NAME", "PIPELINE_URL", "PIPELINE_VERSION"):
                os.environ.pop(key, None)
            with tempfile.TemporaryDirectory() as tmp:
                written = self._run_stage(Path(tmp), raises=False)
                document = Processing.model_validate_json(written.read_text())
        self.assertIsNone(document.pipelines)
        self.assertIsNone(document.data_processes[0].pipeline_name)

    def test_pipeline_env_populates_both_together(self) -> None:
        """The schema requires pipeline_name to resolve within pipelines."""
        env = {
            "PIPELINE_NAME": "aind-pophys-pipeline",
            "PIPELINE_URL": "https://example.org/pipeline",
            "PIPELINE_VERSION": "2.0.0",
        }
        with patch.dict(os.environ, env):
            with tempfile.TemporaryDirectory() as tmp:
                written = self._run_stage(Path(tmp), raises=False)
                document = Processing.model_validate_json(written.read_text())
        self.assertEqual(
            document.data_processes[0].pipeline_name,
            "aind-pophys-pipeline",
        )
        self.assertEqual(
            [code.name for code in document.pipelines],
            ["aind-pophys-pipeline"],
        )


if __name__ == "__main__":
    unittest.main()
