"""Identity and provenance of the model files a classification run loads.

Two files decide what the ROInet embedding produces, and neither is pinned by
a package version: the ``ROInet.zip`` archive, and the Torchvision
ConvNeXt-Tiny checkpoint the archive's ``model.py`` builds a pretrained
backbone from before ROInet's state dictionary is loaded over it. Recording
their digests is what ties a run's predictions to the bytes behind them.

This is provenance, not verification: nothing here compares a digest against
an expectation, and a missing file is omitted rather than failing the run.

The download-method constants live here rather than in ``classification``
because they are the local-versus-network distinction this module reports on,
and ``classification`` cannot be imported without ROICaT.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from aind_pophys_metadata import (
    MODEL_SOURCE_ASSET,
    MODEL_SOURCE_NETWORK,
    model_provenance,
)

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


def torch_home() -> Optional[Path]:
    """Return the Torch cache root from the environment, if it is set.

    Which directory Torch caches hub checkpoints in is the capsule wrapper's
    decision, expressed as ``TORCH_HOME``; this library only reads it.

    Returns
    -------
    Path or None
        The configured Torch cache root, or None when unset or empty.
    """
    value = os.environ.get(TORCH_HOME_ENV)
    if not value:
        return None
    return Path(value)


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
    """  # noqa: E501
    if download_method == DOWNLOAD_FORCE_LOCAL:
        return MODEL_SOURCE_ASSET
    return MODEL_SOURCE_NETWORK


def model_provenance_records(
    network_files_dir: Path,
    download_method: str,
) -> List[Dict[str, str]]:
    """Describe the model files this run loaded, for ``code.parameters``.

    Call this after the embedding has run: in the download branch the archive
    does not exist until ROICaT has fetched it.

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
