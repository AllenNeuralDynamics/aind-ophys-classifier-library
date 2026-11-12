"""CLI/environment settings for ophys ROI classification."""

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

CLASSIFIER_DIR_NAME = "2p_roi_classifier"
ROINET_DIR_NAME = "roinet"
SOMA_CLASSIFIER_FILE = "soma.classification_training.autoclassifier.onnx"
DENDRITE_CLASSIFIER_FILE = "dendrite.classification_training.autoclassifier.onnx"


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
    soma_classifier_path: Optional[Path] = Field(
        default=None,
        description=(
            "Path to the soma classifier ONNX model. Defaults to "
            f"input_dir/{CLASSIFIER_DIR_NAME}/{SOMA_CLASSIFIER_FILE}."
        ),
    )
    dendrite_classifier_path: Optional[Path] = Field(
        default=None,
        description=(
            "Path to the dendrite classifier ONNX model. Defaults to "
            f"input_dir/{CLASSIFIER_DIR_NAME}/{DENDRITE_CLASSIFIER_FILE}."
        ),
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
            "Directory holding a pre-staged ROInet.zip, loaded instead of "
            f"downloaded. When unset, input_dir/{ROINET_DIR_NAME} and "
            f"input_dir/{CLASSIFIER_DIR_NAME} are searched for one."
        ),
    )

    @field_validator("soma_classifier_path", "dendrite_classifier_path", mode="before")
    @classmethod
    def _blank_path_is_unset(cls, value: object) -> object:
        """Treat an empty app-panel value as unset."""
        return None if value == "" else value

    @model_validator(mode="after")
    def _default_classifier_paths(self) -> "ClassifierSettings":
        """Resolve unset classifier paths under ``input_dir``."""
        classifier_dir = self.input_dir / CLASSIFIER_DIR_NAME
        if self.soma_classifier_path is None:
            self.soma_classifier_path = classifier_dir / SOMA_CLASSIFIER_FILE
        if self.dendrite_classifier_path is None:
            self.dendrite_classifier_path = classifier_dir / DENDRITE_CLASSIFIER_FILE
        return self
