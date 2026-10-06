"""ROICaT embedding and ONNX soma/dendrite classification.

The heavy compute for this step: ROInet embedding on the GPU, the two
ONNX logistic-regression classifiers, and the border-ROI geometry.

``network_files_dir`` is an explicit parameter rather than being read off a
global settings object, because where the ROInet weights come from is the
one decision that determines whether a pipeline task reaches out to the
network. See :func:`resolve_network_files_dir`.
"""

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import h5py
import numpy as np
import roicat
import sparse

from .utils.model_utils import (
    DOWNLOAD_CHECK_LOCAL_FIRST,
    DOWNLOAD_FORCE_LOCAL,
    ROINET_ZIP,
)

logger = logging.getLogger(__name__)

EXTRACTION_H5_GLOB = "*extraction.h5"
CLASSIFICATION_SUBDIR = "classification"

ROINET_URL = "https://osf.io/c8m3b/download"
ROINET_MD5 = "357a8d9b630ec79f3e015d0056a4c2d5"
ROI_IMAGE_SIZE = (36, 36)


@dataclass(frozen=True)
class Plane:
    """One plane's classification inputs and output locations."""

    name: str
    input_extraction_file: Path
    output_dir: Path
    output_classification_file: Path


def file_md5(path: Path) -> str:
    """Compute the MD5 hex digest of a file.

    Recorded in ``processing.json`` so a run can be tied to the exact
    classifier weights that produced it.

    Parameters
    ----------
    path : Path
        Path to the file.

    Returns
    -------
    str
        The MD5 hex digest.
    """
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest()


def find_extraction_files(input_dir: Path) -> Tuple[Path, ...]:
    """Locate every extraction HDF5 under ``input_dir``.

    In the pipeline this step is flattened (one task per plane), so this
    normally returns exactly one file; a standalone run may mount
    several. An empty result raises rather than exiting 0 having produced
    nothing, which surfaces in Nextflow only as the far less informative
    "Missing output file(s)".

    Parameters
    ----------
    input_dir : Path
        Input directory.

    Returns
    -------
    tuple of Path
        The extraction HDF5 files, sorted for deterministic ordering.

    Raises
    ------
    FileNotFoundError
        If no extraction HDF5 is found.
    """
    matches = tuple(sorted(Path(input_dir).rglob(EXTRACTION_H5_GLOB)))
    if not matches:
        raise FileNotFoundError(
            f"No '{EXTRACTION_H5_GLOB}' found under {input_dir}; "
            "classification needs the extraction step's output."
        )
    return matches


def prepare_plane(extraction_file: Path, output_dir: Path) -> Plane:
    """Derive a plane's identity and output paths, creating the directory.

    Parameters
    ----------
    extraction_file : Path
        The plane's extraction HDF5, at
        ``<plane>/extraction/<plane>_extraction.h5``.
    output_dir : Path
        Results root.

    Returns
    -------
    Plane
        The populated plane.
    """
    plane_name = extraction_file.parts[-3]
    plane_dir = Path(output_dir) / plane_name / CLASSIFICATION_SUBDIR
    plane_dir.mkdir(parents=True, exist_ok=True)
    return Plane(
        name=plane_name,
        input_extraction_file=extraction_file,
        output_dir=plane_dir,
        output_classification_file=(plane_dir / f"{plane_name}_classification.h5"),
    )


def load_extraction_rois(extraction_file: Path) -> np.ndarray:
    """Load dense ROI masks from an extraction HDF5.

    Reads the sparse ``rois`` group written by ``aind-ophys-extraction``.
    That group's layout is unchanged by the v2 upgrade, so this read needs no
    version dispatch.

    Parameters
    ----------
    extraction_file : Path
        Path to the extraction HDF5.

    Returns
    -------
    numpy.ndarray
        Dense ROI masks of shape ``(n_rois, height, width)``.
    """
    with h5py.File(extraction_file) as f:
        return sparse.COO(f["rois/coords"], f["rois/data"], f["rois/shape"]).todense()


def classify_border_rois(rois: np.ndarray, border_size: int) -> np.ndarray:
    """Flag ROIs within ``border_size`` pixels of the FOV edge.

    Parameters
    ----------
    rois : numpy.ndarray
        Dense ROI masks of shape ``(n_rois, height, width)``.
    border_size : int
        Distance from the border, in pixels.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(n_rois,)``.
    """
    masks = np.where(rois > 0)
    border_rois = set()
    border_rois.update(masks[0][np.where(masks[1] < border_size)[0]].tolist())
    border_rois.update(masks[0][np.where(masks[2] < border_size)[0]].tolist())
    border_rois.update(masks[0][np.where(masks[1] >= rois.shape[1] - border_size)[0]].tolist())
    border_rois.update(masks[0][np.where(masks[2] >= rois.shape[2] - border_size)[0]].tolist())

    flags = np.zeros(shape=rois.shape[0], dtype=bool)
    flags[list(border_rois)] = True
    return flags


def resolve_network_files_dir(
    tmp_dir: Path,
    roinet_dir: Optional[Path],
) -> Tuple[Path, str]:
    """Decide where the ROInet weights come from, and say so loudly.

    ROInet needs a ~105 MB ``ROInet.zip`` and unpacks ``params.json`` /
    ``model.py`` / ``*.pth`` beside it, so the directory must be writable.

    Inside the pipeline ``tmp_dir`` is created fresh per task, so the
    "check_local_first" cache always misses and **every task re-downloads the
    archive from osf.io**. When ``roinet_dir`` points at a directory that
    already holds the archive it is used with ``force_local``, which never
    touches the network. Otherwise a warning names the download, because a
    silent 105 MB fetch per task is the failure mode to avoid, not a detail.

    Parameters
    ----------
    tmp_dir : Path
        Scratch directory, used when no pre-staged archive is available.
    roinet_dir : Path or None
        Directory holding a pre-staged ``ROInet.zip``.

    Returns
    -------
    tuple of (Path, str)
        The directory to hand ROInet, and the ``download_method`` to use.

    Raises
    ------
    FileNotFoundError
        If ``roinet_dir`` is set but holds no ``ROInet.zip``.
    """
    if roinet_dir is not None:
        staged = Path(roinet_dir) / ROINET_ZIP
        if staged.is_file():
            logger.info("Using pre-staged ROInet weights at %s; no download.", staged)
            return Path(roinet_dir), DOWNLOAD_FORCE_LOCAL
        raise FileNotFoundError(
            f"roinet_dir was set to {roinet_dir} but {staged} does not "
            f"exist. Either stage {ROINET_ZIP} there or leave roinet_dir "
            "unset to allow a download."
        )

    logger.warning(
        "No pre-staged ROInet weights; ROInet will download ~105 MB from %s "
        "into %s. Inside the pipeline this happens once per task because the "
        "scratch directory is created fresh. Stage %s and pass --roinet_dir "
        "to avoid it.",
        ROINET_URL,
        tmp_dir,
        ROINET_ZIP,
    )
    return Path(tmp_dir), DOWNLOAD_CHECK_LOCAL_FIRST


def build_roinet_embedder(
    network_files_dir: Path, download_method: str
) -> "roicat.ROInet.ROInet_embedder":
    """Construct the ROInet embedder.

    Parameters
    ----------
    network_files_dir : Path
        Writable directory holding or receiving the ROInet archive.
    download_method : str
        ``"force_local"`` or ``"check_local_first"``, from
        :func:`resolve_network_files_dir`.

    Returns
    -------
    roicat.ROInet.ROInet_embedder
        The constructed embedder.
    """
    Path(network_files_dir).mkdir(parents=True, exist_ok=True)
    return roicat.ROInet.ROInet_embedder(
        device=roicat.helpers.set_device(),
        dir_networkFiles=str(network_files_dir),
        download_method=download_method,
        download_url=ROINET_URL,
        download_hash=ROINET_MD5,
        forward_pass_version="head",
        verbose=False,
    )


def load_onnx_classifier(
    path: Path,
) -> "roicat.classification.classifier.ONNX_model_sklearnLogisticRegression":
    """Load an ONNX logistic-regression classifier.

    Parameters
    ----------
    path : Path
        Path to the ONNX model.

    Returns
    -------
    roicat.classification.classifier.ONNX_model_sklearnLogisticRegression
        The loaded classifier.
    """
    return roicat.classification.classifier.ONNX_model_sklearnLogisticRegression(
        str(path)
    )


def classify_plane(
    rois: np.ndarray,
    um_per_pixel: float,
    soma_classifier: Callable,
    dendrite_classifier: Callable,
    network_files_dir: Path,
    download_method: str,
) -> dict:
    """Embed a plane's ROIs and classify them as soma and/or dendrite.

    ``um_per_pixel`` scales the ROI images fed to the network, so it changes
    the embedding and therefore the predictions. It is required here — the
    caller has already failed the run if it could not be resolved.

    Parameters
    ----------
    rois : numpy.ndarray
        Dense ROI masks of shape ``(n_rois, height, width)``.
    um_per_pixel : float
        Micrometres per pixel of the field of view.
    soma_classifier : callable
        The soma ONNX classifier.
    dendrite_classifier : callable
        The dendrite ONNX classifier.
    network_files_dir : Path
        Directory holding or receiving the ROInet archive.
    download_method : str
        ROInet download method.

    Returns
    -------
    dict
        ``soma_predictions``, ``soma_probabilities``,
        ``dendrite_predictions``, ``dendrite_probabilities``.
    """
    roinet = build_roinet_embedder(network_files_dir, download_method)

    data = roicat.data_importing.Data_roicat()
    data.set_spatialFootprints(rois, um_per_pixel=um_per_pixel)
    data.transform_spatialFootprints_to_ROIImages(out_height_width=ROI_IMAGE_SIZE)

    roinet.generate_dataloader(
        ROI_images=data.ROI_images,
        um_per_pixel=data.um_per_pixel,
        pref_plot=False,
        numWorkers_dataloader=int(os.getenv("CO_CPUS", -1)),
    )
    roinet.generate_latents()

    soma_predictions, soma_probabilities = soma_classifier(roinet.latents)
    dendrite_predictions, dendrite_probabilities = dendrite_classifier(roinet.latents)
    return {
        "soma_predictions": soma_predictions,
        "soma_probabilities": soma_probabilities,
        "dendrite_predictions": dendrite_predictions,
        "dendrite_probabilities": dendrite_probabilities,
    }


def write_classification_h5(
    plane: Plane,
    result: dict,
    border_rois: np.ndarray,
) -> Path:
    """Write the classification HDF5.

    The group and dataset names are the cross-step contract with
    ``aind-ophys-nwb``, which reads ``soma``/``dendrites`` and pairs by
    positional ROI index — so they must not be renamed here.

    Parameters
    ----------
    plane : Plane
        The plane being written.
    result : dict
        Output of :func:`classify_plane`.
    border_rois : numpy.ndarray
        Boolean per-ROI border flags.

    Returns
    -------
    Path
        The path written.
    """
    with h5py.File(plane.output_classification_file, "w") as f:
        group = f.create_group("soma")
        group.create_dataset("predictions", data=result["soma_predictions"])
        group.create_dataset("probabilities", data=result["soma_probabilities"])

        group = f.create_group("dendrites")
        group.create_dataset("predictions", data=result["dendrite_predictions"])
        group.create_dataset("probabilities", data=result["dendrite_probabilities"])

        group = f.create_group("border")
        group.create_dataset("labels", data=border_rois)
    return plane.output_classification_file


def gpu_name() -> Optional[str]:
    """Return the active CUDA device name, or ``None`` when unavailable.

    The classifier is the pipeline's only GPU process, and the shared
    ``collect_static_resources()`` records no GPU, so the name is added here.

    Returns
    -------
    str or None
        The CUDA device name, or ``None`` if no GPU is in use.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.debug("Could not query GPU name: %s", exc)
    return None
