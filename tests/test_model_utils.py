"""Tests for model-file provenance.

These assert provenance, never verification: an unexpected digest is a
recorded fact here, not a failure, and a missing file is omitted rather than
raising.
"""

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aind_pophys_metadata import MODEL_SOURCE_ASSET, MODEL_SOURCE_NETWORK

from aind_ophys_classifier_library.utils import model_utils


def _write(path: Path, payload: bytes) -> str:
    """Write ``payload`` to ``path`` and return its SHA-256 hex digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


class TestTorchHome(unittest.TestCase):
    """Reading the Torch cache root out of the environment."""

    def test_set(self) -> None:
        """A set TORCH_HOME comes back as a Path."""
        with mock.patch.dict(
            os.environ, {model_utils.TORCH_HOME_ENV: "/some/cache"}
        ):
            self.assertEqual(model_utils.torch_home(), Path("/some/cache"))

    def test_unset(self) -> None:
        """An unset TORCH_HOME is None, not a path to the cwd."""
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(model_utils.torch_home())

    def test_empty(self) -> None:
        """An empty TORCH_HOME is treated as unset."""
        with mock.patch.dict(os.environ, {model_utils.TORCH_HOME_ENV: ""}):
            self.assertIsNone(model_utils.torch_home())


class TestModelSource(unittest.TestCase):
    """The local-versus-network distinction."""

    def test_force_local_is_asset(self) -> None:
        """force_local is only passed for pre-staged bytes."""
        self.assertEqual(
            model_utils.model_source(model_utils.DOWNLOAD_FORCE_LOCAL),
            MODEL_SOURCE_ASSET,
        )

    def test_check_local_first_is_network(self) -> None:
        """The fallback branch downloads, so it is labelled as such."""
        self.assertEqual(
            model_utils.model_source(model_utils.DOWNLOAD_CHECK_LOCAL_FIRST),
            MODEL_SOURCE_NETWORK,
        )


class TestModelProvenanceRecords(unittest.TestCase):
    """The records written to ``code.parameters``."""

    def setUp(self) -> None:
        """Stage a temporary ROInet directory and Torch cache."""
        self._temp = tempfile.TemporaryDirectory()
        root = Path(self._temp.name)
        self.network_dir = root / "roinet"
        self.torch_home = root / "torch"
        self.roinet_sha = _write(
            self.network_dir / model_utils.ROINET_ZIP, b"archive-bytes"
        )
        self.convnext_sha = _write(
            self.torch_home / model_utils.CONVNEXT_CHECKPOINT,
            b"checkpoint-bytes",
        )
        self.addCleanup(self._temp.cleanup)

    def _records(self, download_method: str) -> list:
        """Collect records with TORCH_HOME pointed at the staged cache."""
        with mock.patch.dict(
            os.environ, {model_utils.TORCH_HOME_ENV: str(self.torch_home)}
        ):
            return model_utils.model_provenance_records(
                self.network_dir, download_method
            )

    def test_both_files_staged(self) -> None:
        """Both files are described, by logical label and digest."""
        records = self._records(model_utils.DOWNLOAD_FORCE_LOCAL)
        self.assertEqual(
            records,
            [
                {
                    "filename": model_utils.ROINET_ZIP,
                    "sha256": self.roinet_sha,
                    "source_name": model_utils.ROINET_SOURCE_NAME,
                    "source": MODEL_SOURCE_ASSET,
                },
                {
                    "filename": model_utils.CONVNEXT_CHECKPOINT,
                    "sha256": self.convnext_sha,
                    "source_name": model_utils.CONVNEXT_SOURCE_NAME,
                    "source": MODEL_SOURCE_ASSET,
                },
            ],
        )

    def test_download_branch_labelled_network(self) -> None:
        """A downloaded archive is not passed off as a staged asset."""
        records = self._records(model_utils.DOWNLOAD_CHECK_LOCAL_FIRST)
        self.assertEqual(
            [record["source"] for record in records],
            [MODEL_SOURCE_NETWORK, MODEL_SOURCE_NETWORK],
        )

    def test_source_names_are_not_paths(self) -> None:
        """Labels stay logical, so a scratch copy does not leak in."""
        for record in self._records(model_utils.DOWNLOAD_FORCE_LOCAL):
            self.assertNotIn("/", record["source_name"])

    def test_checkpoint_absent(self) -> None:
        """A missing checkpoint is omitted, not an error."""
        (self.torch_home / model_utils.CONVNEXT_CHECKPOINT).unlink()
        records = self._records(model_utils.DOWNLOAD_CHECK_LOCAL_FIRST)
        self.assertEqual(
            [record["filename"] for record in records],
            [model_utils.ROINET_ZIP],
        )

    def test_torch_home_unset(self) -> None:
        """Without TORCH_HOME the checkpoint cannot be located."""
        with mock.patch.dict(os.environ, {}, clear=True):
            records = model_utils.model_provenance_records(
                self.network_dir, model_utils.DOWNLOAD_FORCE_LOCAL
            )
        self.assertEqual(
            [record["filename"] for record in records],
            [model_utils.ROINET_ZIP],
        )

    def test_archive_absent(self) -> None:
        """A run whose archive never arrived records no archive."""
        (self.network_dir / model_utils.ROINET_ZIP).unlink()
        records = self._records(model_utils.DOWNLOAD_CHECK_LOCAL_FIRST)
        self.assertEqual(
            [record["filename"] for record in records],
            [model_utils.CONVNEXT_CHECKPOINT],
        )


if __name__ == "__main__":
    unittest.main()
