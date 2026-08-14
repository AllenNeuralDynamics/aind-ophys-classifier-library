"""PIL helpers for the combined-classifications QC figure.

The composite was assembled by ``aind-ophys-quality-control-aggregator``,
which is unbuildable under aind-data-schema v2 (v2 deletes ``QCEvaluation``),
so it is now built by the capsule that produces its panels.
"""

from pathlib import Path
from typing import List

from PIL import Image

WHITE = (255, 255, 255)


def combine_images_horizontally(
    images: List[Image.Image],
    padding: int = 20,
) -> Image.Image:
    """Paste images side by side on a white background.

    Parameters
    ----------
    images : list of PIL.Image.Image
        Images to combine; must be non-empty.
    padding : int, optional
        Pixels of white space between adjacent images.

    Returns
    -------
    PIL.Image.Image
        The combined image.

    Raises
    ------
    ValueError
        If ``images`` is empty.
    """
    if not images:
        raise ValueError("combine_images_horizontally needs at least one image")
    total_width = sum(image.width for image in images) + (len(images) - 1) * padding
    max_height = max(image.height for image in images)

    combined = Image.new("RGB", (total_width, max_height), WHITE)
    offset = 0
    for image in images:
        combined.paste(image, (offset, 0))
        offset += image.width + padding
    return combined


def combine_images_vertically(
    images: List[Image.Image],
    padding: int = 10,
) -> Image.Image:
    """Stack images vertically on a white background.

    Parameters
    ----------
    images : list of PIL.Image.Image
        Images to combine; must be non-empty.
    padding : int, optional
        Pixels of white space between adjacent images.

    Returns
    -------
    PIL.Image.Image
        The combined image.

    Raises
    ------
    ValueError
        If ``images`` is empty.
    """
    if not images:
        raise ValueError("combine_images_vertically needs at least one image")
    total_height = sum(image.height for image in images) + (len(images) - 1) * padding
    max_width = max(image.width for image in images)

    combined = Image.new("RGB", (max_width, total_height), WHITE)
    offset = 0
    for image in images:
        combined.paste(image, (0, offset))
        offset += image.height + padding
    return combined


def save_image(image: Image.Image, output_path: Path) -> Path:
    """Write an image, creating its parent directory.

    Parameters
    ----------
    image : PIL.Image.Image
        Image to save.
    output_path : Path
        Destination path.

    Returns
    -------
    Path
        The path written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def build_combined_classifications(
    soma_predictions_png: Path,
    soma_probabilities_png: Path,
    dendrite_predictions_png: Path,
    dendrite_probabilities_png: Path,
    output_path: Path,
) -> Path:
    """Assemble the 2x2 combined-classifications figure.

    Soma predictions and probabilities on the top row, dendrite predictions
    and probabilities on the bottom row. This is the figure the classification
    QC metric references.

    Parameters
    ----------
    soma_predictions_png : Path
        Soma prediction overlay figure.
    soma_probabilities_png : Path
        Soma probability overlay figure.
    dendrite_predictions_png : Path
        Dendrite prediction overlay figure.
    dendrite_probabilities_png : Path
        Dendrite probability overlay figure.
    output_path : Path
        Destination for the combined figure.

    Returns
    -------
    Path
        The path written.

    Raises
    ------
    FileNotFoundError
        If any input figure is missing, naming the one that is absent.
    """
    panels = (
        soma_predictions_png,
        soma_probabilities_png,
        dendrite_predictions_png,
        dendrite_probabilities_png,
    )
    for panel in panels:
        if not Path(panel).is_file():
            raise FileNotFoundError(
                f"Cannot build the combined classifications figure: " f"{panel} does not exist."
            )

    with (
        Image.open(panels[0]) as soma_pred,
        Image.open(panels[1]) as soma_prob,
        Image.open(panels[2]) as dend_pred,
        Image.open(panels[3]) as dend_prob,
    ):
        combined = combine_images_vertically(
            [
                combine_images_horizontally([soma_pred, soma_prob]),
                combine_images_horizontally([dend_pred, dend_prob]),
            ]
        )
    return save_image(combined, output_path)
