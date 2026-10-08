import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime as dt
from datetime import timezone as tz
from pathlib import Path
from typing import List

import h5py
import matplotlib.pyplot as plt
import numpy as np
import roicat
import sparse
from aind_data_schema.core.processing import DataProcess, ProcessName
from aind_data_schema.core.quality_control import QCMetric, QCStatus, Status
from aind_qcportal_schema.metric_value import DropdownMetric
from pint import UnitRegistry
from plots import plot_border_rois, plot_predictions, plot_probabilities


@dataclass
class Plane:
    name: str
    input_extraction_file: Path
    output_dir: Path
    output_classification_file: Path
    rois: np.array


def classify_plane(
    rois: np.array, um_per_pixel: float, soma_classifier, dendrite_classifier
):
    """Classify ROIs in a plane as soma or dendrite.

    Parameters
    ----------
    rois : np.array
        The ROIs.
    um_per_pixel : float
        The um_per_pixel value.
    soma_classifier : object
        The soma classifier.
    dendrite_classifier : object
        The dendrite classifier.

    Returns
    -------
    Tuple[np.array, np.array, np.array, np.array]
        The predictions and probabilities for soma and dendrite classifications.
    """
    # ROInet embedding
    roinet = roicat.ROInet.ROInet_embedder(
        device=roicat.helpers.set_device(),  # Which torch device to use ('cpu', 'cuda', etc.)
        dir_networkFiles=args.tmp_dir,  # Directory to download the pretrained network to
        download_method="check_local_first",  # Check to see if a model has already been downloaded to the location (will skip if hash matches) # noqa E501
        download_url="https://osf.io/c8m3b/download",  # URL of the model
        download_hash="357a8d9b630ec79f3e015d0056a4c2d5",  # Hash of the model file
        forward_pass_version="head",  # How the data is passed through the network
        verbose=False,  # Whether to print updates
    )

    data = roicat.data_importing.Data_roicat()
    data.set_spatialFootprints(rois, um_per_pixel=um_per_pixel)
    data.transform_spatialFootprints_to_ROIImages(out_height_width=(36, 36))

    roinet.generate_dataloader(
        ROI_images=data.ROI_images,  # Input images of ROIs
        um_per_pixel=data.um_per_pixel,  # Resolution of FOV
        pref_plot=False,  # Whether or not to plot the ROI sizes
        numWorkers_dataloader=int(os.getenv("CO_CPUS", -1)),
    )
    roinet.generate_latents()

    soma_predictions, soma_probabilities = soma_classifier(roinet.latents)
    dendrite_predictions, dendrite_probabilities = dendrite_classifier(roinet.latents)

    return (
        soma_predictions,
        soma_probabilities,
        dendrite_predictions,
        dendrite_probabilities,
    )


def file_md5(path: Path) -> str:
    """Compute the MD5 hex digest of a file.

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


def find_um_per_pixel(input_dir: Path) -> float:
    """Find the um_per_pixel value from the session.json file.

    Parameters
    ----------
    input_dir : Path
        Path to the input directory.

    Returns
    -------
    float
        The um_per_pixel value
    """
    try:
        session_json_fp = next(input_dir.glob("session.json"))
    except StopIteration:
        session_json_fp = next(input_dir.glob("*/session.json"))
    with open(session_json_fp, "r") as j:
        session_data = json.load(j)
    for data_stream in session_data["data_streams"]:
        if data_stream.get("ophys_fovs"):
            fov = data_stream["ophys_fovs"][0]
            um_per_pixel = (
                float(fov["fov_scale_factor"])
                * (UnitRegistry().parse_expression(fov["fov_scale_factor_unit"]))
                .to("um/pixel")
                .magnitude
            )
    return um_per_pixel


def prepare_plane(extraction_file: Path, output_dir: Path) -> List[Plane]:
    """Prepare a plane for classification by loading the ROIs and creating the output directory.

    Parameters
    ----------
    extraction_file : Path
        Path to the extraction file for the plane.
    output_dir : Path
        Path to the output directory.

    Returns
    -------
    Plane
        The plane object.
    str
        The name of the plane.
    """
    plane_name = extraction_file.parts[-3]
    plane_dir = output_dir / plane_name / "classification"
    plane = Plane(
        name=plane_name,
        input_extraction_file=extraction_file,
        output_dir=plane_dir,
        output_classification_file=plane_dir / f"{plane_name}_classification.h5",
        rois=load_extraction_rois(extraction_file),
    )

    plane.output_dir.mkdir(parents=True, exist_ok=True)

    return plane, plane_name


def load_extraction_rois(extraction_file: Path) -> np.array:
    """Load the ROIs from an extraction file.

    Parameters
    ----------
    extraction_file : Path
        Path to the extraction file.

    Returns
    -------
    np.array
        The ROIs.
    """
    with h5py.File(extraction_file) as f:
        return sparse.COO(f["rois/coords"], f["rois/data"], f["rois/shape"]).todense()


def classify_border_rois(rois: np.array, border_size: int) -> np.array:
    """Classify ROIs that are on the border of the FOV.

    Parameters
    ----------
    rois : np.array
        The ROIs.
    border_size : int
        The distance from the border for an ROI to be considered on the border.

    Returns
    -------
    np.array
        A boolean array indicating which ROIs are on the border
    """
    masks = np.where(rois > 0)
    border_rois = set()
    border_rois.update(masks[0][np.where(masks[1] < border_size)[0]].tolist())
    border_rois.update(masks[0][np.where(masks[2] < border_size)[0]].tolist())
    border_rois.update(
        masks[0][np.where(masks[1] >= rois.shape[1] - border_size)[0]].tolist()
    )
    border_rois.update(
        masks[0][np.where(masks[2] >= rois.shape[2] - border_size)[0]].tolist()
    )

    border_rois_bool = np.zeros(shape=rois.shape[0], dtype=bool)
    border_rois_bool[list(border_rois)] = True

    return border_rois_bool


def write_qc_metrics(output_dir: Path, unique_id: str) -> None:
    """Write QC metrics to output directory

    Parameters
    ----------
    output_dir : Path
        Path to the output directory.
    unique_id : str
        Unique identifier for the plane.
    """

    metric = QCMetric(
        name=f"{unique_id} Classification",
        description="Classification of ROIs as soma or dendrite",
        reference=f"{unique_id}/classification/{unique_id}_combined_classifications.png",
        status_history=[
            QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)
        ],
        value=DropdownMetric(
            value="Reasonable",
            options=[
                "Reasonable",
                "Unreasonable",
            ],
            status=[
                Status.PASS,
                Status.FAIL,
            ],
        ),
    )

    with open(output_dir / f"{unique_id}_classification_metric.json", "w") as f:
        json.dump(json.loads(metric.model_dump_json()), f, indent=4)

    metric = QCMetric(
        name=f"{unique_id} Border ROIs",
        description="ROIs that are on the border of the FOV",
        reference=f"{unique_id}/classification/{unique_id}_border_rois.png",
        status_history=[
            QCStatus(evaluator="Automated", timestamp=dt.now(), status=Status.PASS)
        ],
        value=DropdownMetric(
            value="Reasonable",
            options=[
                "Reasonable",
                "Unreasonable",
            ],
            status=[
                Status.PASS,
                Status.FAIL,
            ],
        ),
    )

    with open(output_dir / f"{unique_id}_border_roi_metric.json", "w") as f:
        json.dump(json.loads(metric.model_dump_json()), f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-dir", type=str, help="Input directory", default="/data"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, help="Output directory", default="/results"
    )
    parser.add_argument(
        "--tmp-dir",
        type=str,
        default="/scratch",
        help="Directory into which to write temporary files",
    )
    parser.add_argument(
        "--soma-classifier-path",
        type=str,
        default="/data/2p_roi_classifier/soma.classification_training.autoclassifier.onnx",
        help="Path of the classifier model, comma-delimited",
    )
    parser.add_argument(
        "--dendrite-classifier-path",
        type=str,
        default="/data/2p_roi_classifier/dendrite.classification_training.autoclassifier.onnx",
        help="Path of the classifier model, comma-delimited",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "Identifier recorded in data_process.json. "
            "Defaults to the directory name of --soma-classifier-path."
        ),
    )
    parser.add_argument(
        "--border-size",
        type=int,
        default=10,
        help="Distance from border for ROI to be classified on the border",
    )
    start_time = dt.now(tz.utc)
    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    model_name = args.model_name or Path(args.soma_classifier_path).parent.name

    for label, p in (
        ("soma", args.soma_classifier_path),
        ("dendrite", args.dendrite_classifier_path),
    ):
        if not Path(p).is_file():
            raise FileNotFoundError(f"{label} classifier ONNX not found: {p}")

    soma_classifier_md5_hash = file_md5(Path(args.soma_classifier_path))
    dendrite_classifier_md5_hash = file_md5(Path(args.dendrite_classifier_path))

    um_per_pixel = find_um_per_pixel(input_dir)

    soma_classifier = (
        roicat.classification.classifier.ONNX_model_sklearnLogisticRegression(
            args.soma_classifier_path
        )
    )

    dendrite_classifier = (
        roicat.classification.classifier.ONNX_model_sklearnLogisticRegression(
            args.dendrite_classifier_path
        )
    )

    for path in Path(input_dir).rglob("*extraction.h5"):
        plane, plane_name = prepare_plane(path, output_dir)

        (
            soma_predictions,
            soma_probabilities,
            dendrite_predictions,
            dendrite_probabilities,
        ) = classify_plane(
            rois=plane.rois,
            um_per_pixel=um_per_pixel,
            soma_classifier=soma_classifier,
            dendrite_classifier=dendrite_classifier,
        )

        border_rois = classify_border_rois(
            rois=plane.rois, border_size=args.border_size
        )

        ax = plot_probabilities(plane.rois, soma_probabilities, "soma probabilities")
        plt.savefig(
            str(plane.output_dir / f"{plane_name}_soma_probabilities.png"),
            bbox_inches="tight",
        )

        ax = plot_probabilities(
            plane.rois, dendrite_probabilities, "dendrite probabilities"
        )
        plt.savefig(
            str(plane.output_dir / f"{plane_name}_dendrite_probabilities.png"),
            bbox_inches="tight",
        )

        ax = plot_predictions(plane.rois, soma_predictions, "soma predictions")
        plt.savefig(
            str(plane.output_dir / f"{plane_name}_soma_predictions.png"),
            bbox_inches="tight",
        )

        ax = plot_predictions(plane.rois, dendrite_predictions, "dendrite predictions")
        plt.savefig(
            str(plane.output_dir / f"{plane_name}_dendrite_predictions.png"),
            bbox_inches="tight",
        )

        ax = plot_border_rois(plane.rois, border_rois, "border ROIs")
        plt.savefig(
            str(plane.output_dir / f"{plane_name}_border_rois.png"), bbox_inches="tight"
        )

        # save results
        with h5py.File(plane.output_classification_file, "w") as f:
            g = f.create_group("soma")
            g.create_dataset("predictions", data=soma_predictions)
            g.create_dataset("probabilities", data=soma_probabilities)

            g = f.create_group("dendrites")
            g.create_dataset("predictions", data=dendrite_predictions)
            g.create_dataset("probabilities", data=dendrite_probabilities)

            g = f.create_group("border")
            g.create_dataset("labels", data=border_rois)

        with open(plane.output_dir / f"{plane_name}_data_process.json", "w") as f:
            dp = DataProcess(
                name=ProcessName.IMAGE_CELL_CLASSIFICATION,
                software_version=os.getenv("VERSION", ""),
                start_date_time=start_time,
                end_date_time=dt.now(),
                input_location=str(path),
                output_location=str(
                    plane.output_dir / f"{plane_name}_classification.h5"
                ),
                code_url=(os.getenv("CODE_URL")),
                parameters={
                    "border_size": args.border_size,
                    "model": model_name,
                    "soma_classifier_path": args.soma_classifier_path,
                    "soma_classifier_md5_hash": soma_classifier_md5_hash,
                    "dendrite_classifier_path": args.dendrite_classifier_path,
                    "dendrite_classifier_md5_hash": dendrite_classifier_md5_hash,
                },
            )
            f.write(dp.model_dump_json(indent=3))
        write_qc_metrics(plane.output_dir, plane_name)
