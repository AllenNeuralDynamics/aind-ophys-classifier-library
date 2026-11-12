"""Tests for model-file discovery and provenance."""

import contextlib
import hashlib
import os
import stat
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


class _TempRootTestCase(unittest.TestCase):
    """A resolved temporary root, with ``TORCH_HOME`` unset and restored."""

    def setUp(self) -> None:
        """Create the root and unset ``TORCH_HOME`` for the test."""
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

    def stage_archive(self, name: str, payload: bytes = b"archive-bytes") -> str:
        """Write ``ROInet.zip`` under ``input_dir/name``; return its SHA-256."""
        return _write(self.input_dir / name / model_utils.ROINET_ZIP, payload)

    def stage_checkpoint(self, name: str, payload: bytes = b"checkpoint-bytes") -> str:
        """Write the ConvNeXt checkpoint under ``input_dir/name/torch``."""
        return _write(self.input_dir / name / "torch" / model_utils.CONVNEXT_CHECKPOINT, payload)


class TestFindRoinetDir(_TempRootTestCase):
    """Search order over the candidate directories under ``input_dir``."""

    def test_finds_roinet_directory(self) -> None:
        """An archive in ``input_dir/roinet`` is found."""
        self.stage_archive("roinet")
        self.assertEqual(model_utils.find_roinet_dir(self.input_dir), self.input_dir / "roinet")

    def test_finds_classifier_directory(self) -> None:
        """An archive in ``input_dir/2p_roi_classifier`` is found."""
        self.stage_archive("2p_roi_classifier")
        self.assertEqual(
            model_utils.find_roinet_dir(self.input_dir),
            self.input_dir / "2p_roi_classifier",
        )

    def test_roinet_directory_comes_first(self) -> None:
        """With an archive in both, ``input_dir/roinet`` wins."""
        self.stage_archive("2p_roi_classifier")
        self.stage_archive("roinet")
        self.assertEqual(model_utils.find_roinet_dir(self.input_dir), self.input_dir / "roinet")

    def test_skips_a_candidate_without_an_archive(self) -> None:
        """A candidate holding no archive falls through to the next."""
        (self.input_dir / "roinet").mkdir()
        self.stage_archive("2p_roi_classifier")
        self.assertEqual(
            model_utils.find_roinet_dir(self.input_dir),
            self.input_dir / "2p_roi_classifier",
        )

    def test_a_directory_is_not_an_archive(self) -> None:
        """Only a file named like the archive counts."""
        (self.input_dir / "roinet" / model_utils.ROINET_ZIP).mkdir(parents=True)
        self.assertIsNone(model_utils.find_roinet_dir(self.input_dir))

    def test_none_when_nothing_is_found(self) -> None:
        """No candidate holding an archive gives None."""
        self.assertIsNone(model_utils.find_roinet_dir(self.input_dir))

    def test_follows_a_symlinked_candidate(self) -> None:
        """A candidate staged as a directory symlink is found through the link."""
        real_dir = self.root / "work" / "task_roinet"
        _write(real_dir / model_utils.ROINET_ZIP, b"archive-bytes")
        (self.input_dir / "roinet").symlink_to(real_dir, target_is_directory=True)
        self.assertEqual(model_utils.find_roinet_dir(self.input_dir), self.input_dir / "roinet")


class TestStageRoinetArchive(_TempRootTestCase):
    """Copying the archive into writable scratch."""

    def setUp(self) -> None:
        """Stage a source archive under ``input_dir/roinet``."""
        super().setUp()
        self.source_dir = self.input_dir / "roinet"
        self.sha = self.stage_archive("roinet")

    def test_copy_lands_in_tmp_dir(self) -> None:
        """The archive is copied byte for byte into ``tmp_dir/roinet``."""
        staged = model_utils.stage_roinet_archive(self.source_dir, self.tmp_dir)
        self.assertEqual(staged, self.tmp_dir / "roinet")
        copied = (staged / model_utils.ROINET_ZIP).read_bytes()
        self.assertEqual(hashlib.sha256(copied).hexdigest(), self.sha)

    def test_source_is_left_in_place(self) -> None:
        """Staging copies; the mounted archive is not moved."""
        model_utils.stage_roinet_archive(self.source_dir, self.tmp_dir)
        self.assertTrue((self.source_dir / model_utils.ROINET_ZIP).is_file())

    def test_copy_is_writable_when_the_source_is_read_only(self) -> None:
        """A read-only mount yields a copy ROICaT can unpack beside."""
        source = self.source_dir / model_utils.ROINET_ZIP
        source.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        staged = model_utils.stage_roinet_archive(self.source_dir, self.tmp_dir)
        mode = (staged / model_utils.ROINET_ZIP).stat().st_mode
        self.assertTrue(mode & stat.S_IWUSR)

    def test_replaces_an_existing_copy(self) -> None:
        """A copy left in ``tmp_dir`` by an earlier run is overwritten."""
        _write(self.tmp_dir / "roinet" / model_utils.ROINET_ZIP, b"stale")
        staged = model_utils.stage_roinet_archive(self.source_dir, self.tmp_dir)
        self.assertEqual((staged / model_utils.ROINET_ZIP).read_bytes(), b"archive-bytes")

    def test_archive_already_in_the_staging_directory(self) -> None:
        """Staging from the staging directory itself is not an error."""
        in_place = self.tmp_dir / "roinet"
        _write(in_place / model_utils.ROINET_ZIP, b"in-place")
        staged = model_utils.stage_roinet_archive(in_place, self.tmp_dir)
        self.assertEqual(staged, in_place)
        self.assertEqual((staged / model_utils.ROINET_ZIP).read_bytes(), b"in-place")


class TestConfigureTorchHome(_TempRootTestCase):
    """Choosing ``TORCH_HOME`` for the ConvNeXt checkpoint."""

    def setUp(self) -> None:
        """Stage an archive under ``input_dir/roinet``."""
        super().setUp()
        self.source_dir = self.input_dir / "roinet"
        self.stage_archive("roinet")
        self.scratch_cache = self.tmp_dir / "roinet" / "torch"

    def test_checkpoint_present_uses_the_mounted_cache(self) -> None:
        """A mounted checkpoint makes the mount's ``torch`` directory the cache."""
        self.stage_checkpoint("roinet")
        result = model_utils.configure_torch_home(self.source_dir, self.tmp_dir)
        self.assertEqual(result, self.source_dir / "torch")
        self.assertEqual(os.environ[model_utils.TORCH_HOME_ENV], str(self.source_dir / "torch"))

    def test_checkpoint_absent_uses_scratch(self) -> None:
        """Without a checkpoint the cache is scratch, never the mount."""
        (self.source_dir / "torch" / "hub" / "checkpoints").mkdir(parents=True)
        result = model_utils.configure_torch_home(self.source_dir, self.tmp_dir)
        self.assertEqual(result, self.scratch_cache)
        self.assertEqual(os.environ[model_utils.TORCH_HOME_ENV], str(self.scratch_cache))

    def test_no_archive_found_uses_scratch(self) -> None:
        """With no discovered directory the cache is scratch."""
        result = model_utils.configure_torch_home(None, self.tmp_dir)
        self.assertEqual(result, self.scratch_cache)
        self.assertEqual(os.environ[model_utils.TORCH_HOME_ENV], str(self.scratch_cache))

    def test_caller_value_is_kept(self) -> None:
        """A TORCH_HOME the caller set is not replaced, even beside a checkpoint."""
        self.stage_checkpoint("roinet")
        os.environ[model_utils.TORCH_HOME_ENV] = "/caller/cache"
        result = model_utils.configure_torch_home(self.source_dir, self.tmp_dir)
        self.assertEqual(result, Path("/caller/cache"))
        self.assertEqual(os.environ[model_utils.TORCH_HOME_ENV], "/caller/cache")

    def test_empty_value_counts_as_unset(self) -> None:
        """An empty TORCH_HOME is replaced rather than kept."""
        os.environ[model_utils.TORCH_HOME_ENV] = ""
        model_utils.configure_torch_home(self.source_dir, self.tmp_dir)
        self.assertEqual(os.environ[model_utils.TORCH_HOME_ENV], str(self.scratch_cache))

    def test_exported_path_is_absolute(self) -> None:
        """Relative inputs are exported as absolute paths."""
        self.stage_checkpoint("roinet")
        with contextlib.chdir(self.root):
            model_utils.configure_torch_home(Path("data/roinet"), Path("scratch"))
        self.assertEqual(os.environ[model_utils.TORCH_HOME_ENV], str(self.source_dir / "torch"))


class TestDiscoverRoinetDir(_TempRootTestCase):
    """Discovery end to end: find, stage and configure."""

    def assert_staged_from(self, name: str) -> None:
        """Assert an archive under ``input_dir/name`` is staged into ``tmp_dir/roinet``."""
        payload = f"archive-in-{name}".encode()
        self.stage_archive(name, payload)
        staged = model_utils.discover_roinet_dir(self.input_dir, self.tmp_dir)
        self.assertEqual(staged, self.tmp_dir / "roinet")
        self.assertEqual((staged / model_utils.ROINET_ZIP).read_bytes(), payload)

    def test_archive_in_roinet_dir(self) -> None:
        """An archive in ``input_dir/roinet`` is staged into ``tmp_dir/roinet``."""
        self.assert_staged_from("roinet")

    def test_archive_in_classifier_dir(self) -> None:
        """An archive in ``input_dir/2p_roi_classifier`` is staged the same way."""
        self.assert_staged_from("2p_roi_classifier")

    def test_nothing_found(self) -> None:
        """No archive gives None, copies nothing, and still sets TORCH_HOME."""
        self.assertIsNone(model_utils.discover_roinet_dir(self.input_dir, self.tmp_dir))
        self.assertFalse((self.tmp_dir / "roinet" / model_utils.ROINET_ZIP).exists())
        expected = self.tmp_dir / "roinet" / "torch"
        self.assertEqual(os.environ[model_utils.TORCH_HOME_ENV], str(expected))

    def test_discovered_files_are_recorded_as_provenance(self) -> None:
        """The staged archive and mounted checkpoint get the usual records."""
        roinet_sha = self.stage_archive("roinet")
        convnext_sha = self.stage_checkpoint("roinet")
        staged = model_utils.discover_roinet_dir(self.input_dir, self.tmp_dir)
        records = model_utils.model_provenance_records(staged, model_utils.DOWNLOAD_FORCE_LOCAL)
        self.assertEqual(
            records,
            [
                {
                    "filename": model_utils.ROINET_ZIP,
                    "sha256": roinet_sha,
                    "source_name": model_utils.ROINET_SOURCE_NAME,
                    "source": MODEL_SOURCE_ASSET,
                },
                {
                    "filename": model_utils.CONVNEXT_CHECKPOINT,
                    "sha256": convnext_sha,
                    "source_name": model_utils.CONVNEXT_SOURCE_NAME,
                    "source": MODEL_SOURCE_ASSET,
                },
            ],
        )


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
