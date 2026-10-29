"""Tests for locating the extraction step's output."""

import tempfile
import unittest
from pathlib import Path

from aind_ophys_classifier_library import classification


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


if __name__ == "__main__":
    unittest.main()
