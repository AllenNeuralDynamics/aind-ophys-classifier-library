"""Tests for ClassifierSettings, including the app-panel drivability contract.

A Code Ocean app panel can only emit ``--name=value``. Two properties must
therefore hold for every settings field, and are asserted here rather than
trusted to the generator: the field parses in that form, and its name is
longer than one character (pydantic-settings renders a one-character field
as a short option, which the panel can never produce).
"""

import sys
import typing
import unittest
from pathlib import Path
from unittest.mock import patch

from aind_ophys_classifier_library.settings import ClassifierSettings


def _make_settings(argv=None, **kwargs):
    """Construct ClassifierSettings with a controlled sys.argv."""
    with patch.object(sys, "argv", ["run_capsule.py"] + list(argv or [])):
        return ClassifierSettings(**kwargs)


def _sample_cli_value(annotation) -> str:
    """Return a valid ``--name=value`` payload for a field annotation."""
    origin = typing.get_origin(annotation)
    if origin is typing.Union:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        annotation = args[0]
    if annotation is bool:
        return "true"
    if annotation is int:
        return "2"
    if annotation is float:
        return "1.5"
    if annotation is Path:
        return "/tmp/classifier-test"
    return "sample"


class TestClassifierSettingsDefaults(unittest.TestCase):
    """Defaults match the capsule's historical argparse defaults."""

    def test_defaults(self):
        """Defaults are unchanged from the v1 argparse parser."""
        settings = _make_settings()
        self.assertEqual(settings.input_dir, Path("/data"))
        self.assertEqual(settings.output_dir, Path("/results"))
        self.assertEqual(settings.tmp_dir, Path("/scratch"))
        self.assertTrue(
            str(settings.soma_classifier_path).endswith(
                "soma.classification_training.autoclassifier.onnx"
            )
        )
        self.assertTrue(
            str(settings.dendrite_classifier_path).endswith(
                "dendrite.classification_training.autoclassifier.onnx"
            )
        )
        self.assertIsNone(settings.model_name)
        self.assertEqual(settings.border_size, 10)
        self.assertIsNone(settings.um_per_pixel)
        self.assertIsNone(settings.roinet_dir)
        self.assertFalse(settings.verify)

    def test_one_character_argparse_flags_are_renamed(self):
        """The v1 -i/-o flags become full names, never aliases."""
        for name in ("i", "o"):
            self.assertNotIn(name, ClassifierSettings.model_fields)
        self.assertIn("input_dir", ClassifierSettings.model_fields)
        self.assertIn("output_dir", ClassifierSettings.model_fields)

    def test_hyphenated_arguments_are_rejected(self):
        """pydantic-settings takes underscores only.

        The v1 argparse parser and the committed app panel both used
        hyphens, so main.nf's run line must move in lockstep.
        """
        with self.assertRaises(SystemExit):
            _make_settings(["--input-dir=/data"])

    def test_bool_string_parsing(self):
        """Bools parse from strings, so --verify 1 works."""
        self.assertTrue(_make_settings(["--verify=1"]).verify)


class TestSampleCliValue(unittest.TestCase):
    """The helper the app-panel contract test drives every field with."""

    def test_covers_every_annotation_this_settings_class_uses(self):
        """Each supported annotation yields a parseable payload."""
        self.assertEqual(_sample_cli_value(bool), "true")
        self.assertEqual(_sample_cli_value(int), "2")
        self.assertEqual(_sample_cli_value(float), "1.5")
        self.assertTrue(_sample_cli_value(Path).startswith("/"))
        self.assertEqual(_sample_cli_value(str), "sample")

    def test_optional_unwraps_to_the_inner_type(self):
        """Optional[float] is driven as a float, not as a string."""
        self.assertEqual(_sample_cli_value(typing.Optional[float]), "1.5")


class TestAppPanelContract(unittest.TestCase):
    """Every field must be drivable by a Code Ocean app panel."""

    def test_no_single_character_field_names(self):
        """No field name is one character long."""
        offenders = [name for name in ClassifierSettings.model_fields if len(name) == 1]
        self.assertEqual(offenders, [])

    def test_every_field_parses_as_named_argument(self):
        """Each field accepts the ``--name=value`` form the panel emits."""
        for name, field in ClassifierSettings.model_fields.items():
            with self.subTest(field=name):
                value = _sample_cli_value(field.annotation)
                settings = _make_settings([f"--{name}={value}"])
                self.assertIsNotNone(getattr(settings, name))

    def test_field_names_use_underscores(self):
        """pydantic-settings style: no hyphens in field names."""
        offenders = [name for name in ClassifierSettings.model_fields if "-" in name]
        self.assertEqual(offenders, [])

    def test_every_field_has_a_description(self):
        """A panel with no descriptions is unusable by an operator."""
        missing = [
            name for name, field in ClassifierSettings.model_fields.items() if not field.description
        ]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
