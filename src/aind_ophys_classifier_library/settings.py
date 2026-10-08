"""CLI/environment settings for ophys ROI classification."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CLASSIFIER_DIR = "/data/2p_roi_classifier"


class ClassifierSettings(BaseSettings, cli_parse_args=True):
    """Settings for ROI classification using pydantic-settings."""

    model_config = SettingsConfigDict(
        env_prefix="CLASSIFIER_",
        case_sensitive=False,
        extra="ignore",
    )

    input_dir: Path = Field(
        default=Path("/data"),
        description="Directory containing the extraction results to classify.",
    )
    output_dir: Path = Field(
        default=Path("/results"),
        description="Directory where output files will be saved.",
    )
    tmp_dir: Path = Field(
        default=Path("/scratch"),
        description=(
            "Directory for temporary files, and where the ROInet network "
            "files are unpacked. Must be writable."
        ),
    )
    soma_classifier_path: Path = Field(
        default=Path(DEFAULT_CLASSIFIER_DIR) / "soma.classification_training.autoclassifier.onnx",
        description="Path to the soma classifier ONNX model.",
    )
    dendrite_classifier_path: Path = Field(
        default=Path(DEFAULT_CLASSIFIER_DIR)
        / "dendrite.classification_training.autoclassifier.onnx",
        description="Path to the dendrite classifier ONNX model.",
    )
    model_name: Optional[str] = Field(
        default=None,
        description=(
            "Identifier recorded in processing.json. Defaults to the "
            "directory name of soma_classifier_path."
        ),
    )
    border_size: int = Field(
        default=10,
        description=(
            "Distance in pixels from the field-of-view border for an ROI to "
            "be classified as a border ROI."
        ),
    )
    um_per_pixel: Optional[float] = Field(
        default=None,
        description=(
            "Micrometres per pixel used to scale the ROI images fed to "
            "ROInet. Read from the acquisition metadata when omitted."
        ),
    )
    roinet_dir: Optional[Path] = Field(
        default=None,
        description=(
            "Directory holding a pre-staged ROInet.zip. When set, the network "
            "files are loaded from there instead of being downloaded."
        ),
    )
