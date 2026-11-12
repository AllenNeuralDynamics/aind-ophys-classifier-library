"""Discovery and provenance of the ROInet archive and ConvNeXt-Tiny checkpoint."""

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from aind_pophys_metadata import (
    MODEL_SOURCE_ASSET,
    MODEL_SOURCE_NETWORK,
    model_provenance,
)

from ..settings import CLASSIFIER_DIR_NAME, ROINET_DIR_NAME

logger = logging.getLogger(__name__)

ROINET_ZIP = "ROInet.zip"

# ROICaT's ``download_method`` values: ``force_local`` is only ever passed
# when the archive is already staged, so it is also the signal that no
# download happened.
DOWNLOAD_FORCE_LOCAL = "force_local"
DOWNLOAD_CHECK_LOCAL_FIRST = "check_local_first"

# Logical labels, kept stable across runs. The archive's directory is a
# per-task scratch copy, so its basename would name nothing durable.
ROINET_SOURCE_NAME = "roinet"
CONVNEXT_SOURCE_NAME = "torchvision_convnext_tiny"

# Torch resolves hub checkpoints under ``$TORCH_HOME/hub/checkpoints``.
TORCH_HOME_ENV = "TORCH_HOME"
CONVNEXT_CHECKPOINT = "hub/checkpoints/convnext_tiny-983f1562.pth"

# Directories under ``input_dir`` searched, in order, for ``ROInet.zip``.
ROINET_SEARCH_DIRS = (ROINET_DIR_NAME, CLASSIFIER_DIR_NAME)

# Under ``tmp_dir``: ROICaT unpacks beside the archive, so it is staged here
# rather than used from a read-only mount; the Torch cache sits beside it.
STAGING_DIR_NAME = "roinet"
TORCH_DIR_NAME = "torch"


def torch_home() -> Optional[Path]:
    """Return the Torch cache root from the environment, if it is set.

    Returns
    -------
    Path or None
        The configured Torch cache root, or None when unset or empty.
    """
    value = os.environ.get(TORCH_HOME_ENV)
    if not value:
        return None
    return Path(value)


def find_roinet_dir(input_dir: Path) -> Optional[Path]:
    """Find the first directory under ``input_dir`` holding ``ROInet.zip``.

    Searches ``input_dir/roinet``, then ``input_dir/2p_roi_classifier``.

    Parameters
    ----------
    input_dir : Path
        Input directory.

    Returns
    -------
    Path or None
        The first directory holding the archive, or None when none does.
    """
    for name in ROINET_SEARCH_DIRS:
        candidate = Path(input_dir) / name
        if (candidate / ROINET_ZIP).is_file():
            return candidate
    return None


def stage_roinet_archive(source_dir: Path, tmp_dir: Path) -> Path:
    """Copy ``ROInet.zip`` from ``source_dir`` into ``tmp_dir/roinet``.

    The copy is writable even when the source is read-only, and the source is
    left in place.

    Parameters
    ----------
    source_dir : Path
        Directory holding the archive, typically a read-only mount.
    tmp_dir : Path
        Writable scratch directory.

    Returns
    -------
    Path
        The ``tmp_dir/roinet`` directory holding the copy.
    """
    staging_dir = Path(tmp_dir) / STAGING_DIR_NAME
    staging_dir.mkdir(parents=True, exist_ok=True)
    source = Path(source_dir) / ROINET_ZIP
    destination = staging_dir / ROINET_ZIP
    logger.info("Copying %s to %s.", source, destination)
    try:
        shutil.copyfile(source, destination)
    except shutil.SameFileError:
        pass  # the archive is already in the staging directory
    return staging_dir


def configure_torch_home(source_dir: Optional[Path], tmp_dir: Path) -> Path:
    """Point ``TORCH_HOME`` at a cache for the ConvNeXt-Tiny checkpoint.

    A ``TORCH_HOME`` already set in the environment is kept. Otherwise it is
    set to ``source_dir/torch`` when that holds the checkpoint, and to the
    writable ``tmp_dir/roinet/torch`` when it does not. Torch reads the
    variable when the weights load, so call this before the ROInet embedder
    is built.

    Parameters
    ----------
    source_dir : Path or None
        Directory holding the discovered ROInet archive, or None when none
        was found.
    tmp_dir : Path
        Writable scratch directory.

    Returns
    -------
    Path
        The Torch cache root in effect.
    """
    current = torch_home()
    if current is not None:
        logger.info("%s=%s is already set; leaving it unchanged.", TORCH_HOME_ENV, current)
        return current

    mounted = None if source_dir is None else Path(source_dir) / TORCH_DIR_NAME
    if mounted is not None and (mounted / CONVNEXT_CHECKPOINT).is_file():
        cache_root = mounted.resolve()
        logger.info("%s=%s: using the mounted ConvNeXt checkpoint.", TORCH_HOME_ENV, cache_root)
    else:
        cache_root = (Path(tmp_dir) / STAGING_DIR_NAME / TORCH_DIR_NAME).resolve()
        logger.info(
            "%s=%s: no mounted ConvNeXt checkpoint; using a writable cache.",
            TORCH_HOME_ENV,
            cache_root,
        )
    os.environ[TORCH_HOME_ENV] = str(cache_root)
    return cache_root


def discover_roinet_dir(input_dir: Path, tmp_dir: Path) -> Optional[Path]:
    """Locate a mounted ``ROInet.zip``, stage a copy and configure ``TORCH_HOME``.

    Parameters
    ----------
    input_dir : Path
        Input directory, searched as in :func:`find_roinet_dir`.
    tmp_dir : Path
        Writable scratch directory.

    Returns
    -------
    Path or None
        The directory holding the staged copy, or None when no archive is
        found.
    """
    source_dir = find_roinet_dir(input_dir)
    configure_torch_home(source_dir, tmp_dir)
    if source_dir is None:
        return None
    return stage_roinet_archive(source_dir, tmp_dir)


def model_source(download_method: str) -> str:
    """Map a ROICaT download method onto a provenance source label.

    Parameters
    ----------
    download_method : str
        ``download_method`` handed to the embedder, from
        :func:`~aind_ophys_classifier_library.classification.resolve_network_files_dir`.

    Returns
    -------
    str
        ``MODEL_SOURCE_ASSET`` when the bytes were already local, otherwise
        ``MODEL_SOURCE_NETWORK``.
    """
    if download_method == DOWNLOAD_FORCE_LOCAL:
        return MODEL_SOURCE_ASSET
    return MODEL_SOURCE_NETWORK


def model_provenance_records(
    network_files_dir: Path,
    download_method: str,
) -> List[Dict[str, str]]:
    """Describe the model files this run loaded, for ``code.parameters``.

    Call after the embedding has run.

    Parameters
    ----------
    network_files_dir : Path
        Directory ROInet was given, holding or having received the archive.
    download_method : str
        ``download_method`` the embedder was constructed with.

    Returns
    -------
    list of dict
        One record per model file found, each with ``filename``, ``sha256``,
        ``source_name`` and ``source``. Files that are absent are omitted.
    """
    source = model_source(download_method)
    records = model_provenance(
        Path(network_files_dir),
        [ROINET_ZIP],
        source_name=ROINET_SOURCE_NAME,
        source=source,
    )
    cache_root = torch_home()
    if cache_root is None:
        logger.warning(
            "%s is unset; the ConvNeXt-Tiny checkpoint behind the ROInet "
            "backbone cannot be located and is omitted from provenance",
            TORCH_HOME_ENV,
        )
        return records
    records.extend(
        model_provenance(
            cache_root,
            [CONVNEXT_CHECKPOINT],
            source_name=CONVNEXT_SOURCE_NAME,
            source=source,
        )
    )
    return records
