"""Tests for locating the extraction step's output and the ROInet weights."""

import os
import tempfile
import unittest
from pathlib import Path
from typing import Optional, Tuple
from unittest import mock

from aind_ophys_classifier_library import classification
from aind_ophys_classifier_library.utils import model_utils


def _write_plane(root: Path, plane: str) -> Path:
    """Write ``plane``'s extraction file in a real directory under ``root``."""
    extraction = root / plane / "extraction"
    extraction.mkdir(parents=True)
    path = extraction / f"{plane}_extraction.h5"
    path.touch()
    return path


def _stage_plane(root: Path, staging: Path, plane: str) -> Path:
    """Stage ``plane`` into ``root`` the way Nextflow stages task inputs.

    The plane's directory is real and sits under ``staging``, outside
    ``root``, under a different name; ``root`` holds only a symlink to it.
    Returns the extraction file's path as seen through ``root``.
    """
    real_dir = staging / f"task_{plane}"
    (real_dir / "extraction").mkdir(parents=True)
    (real_dir / "extraction" / f"{plane}_extraction.h5").touch()
    (root / plane).symlink_to(real_dir, target_is_directory=True)
    return root / plane / "extraction" / f"{plane}_extraction.h5"


class TestFindExtractionFiles(unittest.TestCase):
    """Discovery of extraction HDF5 files under the input directory."""

    def test_follows_directory_symlink(self) -> None:
        """A plane staged as a directory symlink is found through the link."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            root = base / "data"
            root.mkdir()
            expected = _stage_plane(root, base / "work", "plane_0")

            found = classification.find_extraction_files(root)

            self.assertEqual(found, (expected,))

    def test_sorts_real_and_linked_planes_together(self) -> None:
        """Real and symlinked planes come back in one sorted tuple."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            root = base / "data"
            root.mkdir()
            first = _stage_plane(root, base / "work", "plane_0")
            second = _write_plane(root, "plane_1")
            third = _stage_plane(root, base / "work", "plane_2")

            found = classification.find_extraction_files(root)

            self.assertEqual(found, (first, second, third))

    def test_raises_when_nothing_is_found(self) -> None:
        """The error names the glob and the directory that was searched."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()

            with self.assertRaises(FileNotFoundError) as caught:
                classification.find_extraction_files(root)

            self.assertIn(classification.EXTRACTION_H5_GLOB, str(caught.exception))
            self.assertIn(str(root), str(caught.exception))


class TestResolveNetworkFilesDir(unittest.TestCase):
    """Choice of the ROInet directory and download method."""

    def setUp(self) -> None:
        """Create input and scratch directories with ``TORCH_HOME`` unset."""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.input_dir = self.root / "data"
        self.tmp_dir = self.root / "scratch"
        self.input_dir.mkdir()
        environ = mock.patch.dict(os.environ)
        environ.start()
        self.addCleanup(environ.stop)
        os.environ.pop(model_utils.TORCH_HOME_ENV, None)

    def stage_archive(self, directory: Path, payload: bytes = b"archive-bytes") -> Path:
        """Write ``ROInet.zip`` into ``directory`` and return the directory."""
        directory.mkdir(parents=True, exist_ok=True)
        (directory / model_utils.ROINET_ZIP).write_bytes(payload)
        return directory

    def stage_checkpoint(self, directory: Path) -> None:
        """Write the ConvNeXt checkpoint under ``directory/torch``."""
        checkpoint = directory / "torch" / model_utils.CONVNEXT_CHECKPOINT
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint-bytes")

    def resolve(self, roinet_dir: Optional[Path] = None) -> Tuple[Path, str]:
        """Call ``resolve_network_files_dir`` with this test's directories."""
        return classification.resolve_network_files_dir(self.input_dir, self.tmp_dir, roinet_dir)

    def test_explicit_roinet_dir_is_used_as_given(self) -> None:
        """An explicit directory is used in place; nothing is discovered or set."""
        explicit = self.stage_archive(self.root / "explicit")
        self.stage_archive(self.input_dir / "roinet")
        with self.assertLogs(classification.logger, "INFO") as logs:
            result = self.resolve(explicit)
        self.assertEqual(result, (explicit, model_utils.DOWNLOAD_FORCE_LOCAL))
        staged = explicit / model_utils.ROINET_ZIP
        self.assertIn(f"Using pre-staged ROInet weights at {staged}", logs.output[0])
        self.assertFalse(self.tmp_dir.exists())
        self.assertNotIn(model_utils.TORCH_HOME_ENV, os.environ)

    def test_explicit_roinet_dir_without_archive_raises(self) -> None:
        """An explicit directory with no archive fails instead of discovering one."""
        self.stage_archive(self.input_dir / "roinet")
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(FileNotFoundError) as caught:
            self.resolve(empty)
        self.assertIn(f"roinet_dir was set to {empty}", str(caught.exception))
        self.assertNotIn(model_utils.TORCH_HOME_ENV, os.environ)

    def assert_discovered_in(self, name: str) -> None:
        """Assert an archive under ``input_dir/name`` is copied to scratch and used offline."""
        self.stage_archive(self.input_dir / name, name.encode())
        with self.assertLogs(classification.logger, "INFO") as logs:
            result = self.resolve()
        staged = self.tmp_dir / "roinet"
        self.assertEqual(result, (staged, model_utils.DOWNLOAD_FORCE_LOCAL))
        self.assertEqual((staged / model_utils.ROINET_ZIP).read_bytes(), name.encode())
        self.assertIn("Using pre-staged ROInet weights", logs.output[-1])

    def test_archive_discovered_in_roinet_dir(self) -> None:
        """An archive in ``input_dir/roinet`` is copied to scratch and used offline."""
        self.assert_discovered_in("roinet")

    def test_archive_discovered_in_classifier_dir(self) -> None:
        """An archive in ``input_dir/2p_roi_classifier`` is handled the same way."""
        self.assert_discovered_in("2p_roi_classifier")

    def test_mounted_checkpoint_sets_torch_home_to_the_mount(self) -> None:
        """A checkpoint beside the archive is read in place."""
        self.stage_checkpoint(self.stage_archive(self.input_dir / "roinet"))
        self.resolve()
        expected = self.input_dir / "roinet" / "torch"
        self.assertEqual(os.environ[model_utils.TORCH_HOME_ENV], str(expected))

    def test_missing_checkpoint_sets_torch_home_to_scratch(self) -> None:
        """An archive without a checkpoint caches Torch downloads in scratch."""
        self.stage_archive(self.input_dir / "roinet")
        self.resolve()
        expected = self.tmp_dir / "roinet" / "torch"
        self.assertEqual(os.environ[model_utils.TORCH_HOME_ENV], str(expected))

    def test_caller_torch_home_is_respected(self) -> None:
        """A TORCH_HOME the caller set survives discovery."""
        self.stage_checkpoint(self.stage_archive(self.input_dir / "roinet"))
        os.environ[model_utils.TORCH_HOME_ENV] = "/caller/cache"
        self.resolve()
        self.assertEqual(os.environ[model_utils.TORCH_HOME_ENV], "/caller/cache")

    def test_nothing_found_falls_back_to_download(self) -> None:
        """With no archive anywhere, ROInet downloads into scratch and says so."""
        with self.assertLogs(classification.logger, "WARNING") as logs:
            result = self.resolve()
        self.assertEqual(result, (self.tmp_dir, model_utils.DOWNLOAD_CHECK_LOCAL_FIRST))
        self.assertEqual(len(logs.output), 1)
        self.assertIn("No pre-staged ROInet weights; ROInet will download ~105 MB", logs.output[0])
        self.assertIn(classification.ROINET_URL, logs.output[0])
        self.assertFalse((self.tmp_dir / "roinet" / model_utils.ROINET_ZIP).exists())
        expected = self.tmp_dir / "roinet" / "torch"
        self.assertEqual(os.environ[model_utils.TORCH_HOME_ENV], str(expected))


if __name__ == "__main__":
    unittest.main()
