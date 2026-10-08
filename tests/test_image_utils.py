"""Tests for the combined-classifications figure."""

import tempfile
import unittest
from pathlib import Path
from typing import Tuple

from PIL import Image

from aind_ophys_classifier_library.utils import image_utils

PLANE = "VISp_0"


def _panel(directory: Path, name: str, size: Tuple[int, int] = (10, 8)) -> Path:
    """Write a small solid-colour PNG to stand in for a real figure."""
    path = directory / name
    Image.new("RGB", size, (10, 20, 30)).save(path)
    return path


class TestCombineHelpers(unittest.TestCase):
    """Horizontal and vertical combination geometry."""

    def test_horizontal_geometry(self) -> None:
        """Widths sum with padding; height is the tallest panel."""
        images = [Image.new("RGB", (10, 8)), Image.new("RGB", (6, 12))]
        combined = image_utils.combine_images_horizontally(images, padding=5)
        self.assertEqual(combined.size, (10 + 5 + 6, 12))

    def test_vertical_geometry(self) -> None:
        """Heights sum with padding; width is the widest panel."""
        images = [Image.new("RGB", (10, 8)), Image.new("RGB", (6, 12))]
        combined = image_utils.combine_images_vertically(images, padding=3)
        self.assertEqual(combined.size, (10, 8 + 3 + 12))

    def test_horizontal_requires_an_image(self) -> None:
        """An empty list is a caller error, not a blank canvas."""
        with self.assertRaises(ValueError):
            image_utils.combine_images_horizontally([])

    def test_vertical_requires_an_image(self) -> None:
        """An empty list is a caller error, not a blank canvas."""
        with self.assertRaises(ValueError):
            image_utils.combine_images_vertically([])


class TestSaveImage(unittest.TestCase):
    """The writer creates its parent directory."""

    def test_creates_parent_directory(self) -> None:
        """Writing into a not-yet-existing plane directory works."""
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / PLANE / "classification" / "x.png"
            written = image_utils.save_image(Image.new("RGB", (4, 4)), target)
            self.assertTrue(written.is_file())


class TestBuildCombinedClassifications(unittest.TestCase):
    """The 2x2 composite the classification QC metric references."""

    def test_builds_two_by_two_layout(self) -> None:
        """Two rows of two panels, sized from the inputs plus padding."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panels = {
                role: _panel(root, f"{PLANE}_{role}.png")
                for role in (
                    "soma_predictions",
                    "soma_probabilities",
                    "dendrite_predictions",
                    "dendrite_probabilities",
                )
            }
            out = root / f"{PLANE}_combined_classifications.png"
            written = image_utils.build_combined_classifications(
                soma_predictions_png=panels["soma_predictions"],
                soma_probabilities_png=panels["soma_probabilities"],
                dendrite_predictions_png=panels["dendrite_predictions"],
                dendrite_probabilities_png=panels["dendrite_probabilities"],
                output_path=out,
            )
            with Image.open(written) as combined:
                # two 10px panels + 20px default horizontal padding
                self.assertEqual(combined.width, 10 + 20 + 10)
                # two 8px rows + 10px default vertical padding
                self.assertEqual(combined.height, 8 + 10 + 8)

    def test_missing_panel_names_the_file(self) -> None:
        """A missing input figure raises, naming which one is absent."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            present = _panel(root, "present.png")
            missing = root / "absent.png"
            with self.assertRaises(FileNotFoundError) as ctx:
                image_utils.build_combined_classifications(
                    soma_predictions_png=present,
                    soma_probabilities_png=missing,
                    dendrite_predictions_png=present,
                    dendrite_probabilities_png=present,
                    output_path=root / "out.png",
                )
        self.assertIn("absent.png", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
