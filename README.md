# aind-ophys-classifier-library

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![Python](https://img.shields.io/badge/python->=3.11-blue?logo=python)

ROI soma/dendrite classification for AIND ophys pipelines.

This is the backing library for the `aind-ophys-classifier` Code Ocean capsule:
all the logic lives here with its own tests, and the capsule is a thin wrapper
that calls `aind_ophys_classifier_library.job.run`. Metadata reads, the
`Processing` builders and the QC document builders come from the shared
[`aind-pophys-metadata`](https://github.com/AllenNeuralDynamics/aind-pophys-metadata)
package, so they stay consistent across the pipeline's processing steps.

## What it does

For each plane mounted under the input directory:

1. Read micrometres per pixel through the shared `CoreMetadata`, which owns
   every metadata format (v1 `session.json`, v2 `acquisition.json`, or a
   minimal `metadata.json`).
2. Load the plane's dense ROI masks from the extraction step's
   `<plane>/extraction/<plane>_extraction.h5`.
3. Embed the ROI images with ROICaT's ROInet, then classify them with two ONNX
   logistic-regression heads (soma and dendrite).
4. Flag ROIs within `border_size` pixels of the field-of-view edge.
5. Write the classification HDF5, the overlay figures, `processing.json` and
   `quality_control.json`.

`um_per_pixel` scales the ROI images fed to ROInet, so it changes the embedding
and therefore every prediction. It is never defaulted: when it cannot be read
from the metadata the run fails and asks for `--um_per_pixel`, because ROInet
silently assumes 1.0 um/pixel when it is unset.

## Modules

### `job.py`
Orchestration. `run()` is the entry point the thin runtime wrapper calls. Everything shared
across planes is resolved once, then each plane is classified inside its own
`stage_guard`, so a failing plane still leaves a `processing.json` naming what
was attempted and why it stopped.

### `classification.py`
The heavy compute: `find_extraction_files`, `prepare_plane`,
`load_extraction_rois`, `resolve_network_files_dir`, `build_roinet_embedder`,
`load_onnx_classifier`, `classify_plane`, `classify_border_rois`,
`write_classification_h5`, plus `file_md5` and `gpu_name`.

### `plots.py`
The three overlay figure builders — `plot_probabilities`, `plot_predictions`
and `plot_border_rois`. Each returns the `Figure`; the caller saves and closes
it.

### `qc.py`
Metric content for `quality_control.json`. Metric-producing functions
(`build_classification_metric`, `build_border_roi_metric`) return metrics;
`job.run` accumulates them and `write_quality_control_json` writes one document
per plane.

### `settings.py`
`ClassifierSettings`, a `pydantic-settings` class parsed from the command line
or `CLASSIFIER_`-prefixed environment variables. The capsule's Code Ocean app
panel is generated from this file with `auto-app-panel`.

### `utils/metadata_utils.py`
Metadata reads. `load_classifier_metadata` returns the frozen
`ClassifierMetadata` view over the shared package's `CoreMetadata.load`, and
raises rather than defaulting when micrometres per pixel is unresolvable.
Also `build_classification_code`, `process_name`, and the
`classification_stage` context manager that binds the shared `stage_guard` to
this step's process type.

### `utils/model_utils.py`
Model provenance. `model_provenance_records` records the SHA-256 of the two
files that decide what the embedding produces and that no package version pins
— `ROInet.zip` and the Torchvision ConvNeXt-Tiny checkpoint resolved under
`$TORCH_HOME` — onto `code.parameters`. This is provenance, not verification:
nothing is compared against an expected digest, and a missing file is omitted
rather than failing the run.

### `utils/image_utils.py`
PIL helpers that assemble the 2x2 combined-classifications figure the
classification QC metric references.

## Settings

| Field | Default | Purpose |
|---|---|---|
| `input_dir` | `/data` | Extraction results to classify. |
| `output_dir` | `/results` | Results root. |
| `tmp_dir` | `/scratch` | Scratch, and where ROInet unpacks. Must be writable. |
| `soma_classifier_path` | `/data/2p_roi_classifier/soma.classification_training.autoclassifier.onnx` | Soma ONNX model. |
| `dendrite_classifier_path` | `/data/2p_roi_classifier/dendrite.classification_training.autoclassifier.onnx` | Dendrite ONNX model. |
| `model_name` | derived | Identifier recorded in `processing.json`; defaults to the directory name of `soma_classifier_path`. |
| `border_size` | `10` | Pixels from the FOV edge for an ROI to count as a border ROI. |
| `um_per_pixel` | from metadata | Override, for metadata that does not carry the scale. |
| `roinet_dir` | unset | Directory holding a pre-staged `ROInet.zip`. |

Set `roinet_dir` inside the pipeline. `tmp_dir` is created fresh per task, so
without a pre-staged archive ROInet's cache always misses and every task
re-downloads ~105 MB from osf.io.

## Outputs

Per plane, under `<output_dir>/<plane>/classification/`:

- `<plane>_classification.h5` — groups `soma` and `dendrites`, each with
  `predictions` and `probabilities`, plus `border/labels`. These names are the
  cross-step contract with `aind-ophys-nwb`, which pairs by positional ROI
  index.
- `<plane>_soma_predictions.png`, `<plane>_soma_probabilities.png`,
  `<plane>_dendrite_predictions.png`, `<plane>_dendrite_probabilities.png`,
  `<plane>_border_rois.png`, `<plane>_combined_classifications.png`
- `processing.json` — a full `Processing` document.
- `quality_control.json` — a full `QualityControl` document, all metrics tagged
  into one `Classification` evaluation group.

## Level of Support
Please indicate a level of support:
 - [ ] Supported: We are releasing this code to the public as a tool we expect others to use. Issues are welcomed, and we expect to address them promptly; pull requests will be vetted by our staff before inclusion.
 - [ ] Occasional updates: We are planning on occasional updating this tool with no fixed schedule. Community involvement is encouraged through both issues and pull requests.
 - [ ] Unsupported: We are not currently supporting this code, but simply releasing it to the community AS IS but are not able to provide any guarantees of support. The community is welcome to submit issues, but you should not expect an active response.

## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

To develop the code, run
```bash
pip install -e . --group dev
```
Note: --group flag is available only in pip versions >=25.1

Alternatively, if using [uv](https://docs.astral.sh/uv/), run
```bash
uv sync
```

Requires Python >=3.11,<3.13: `roicat` 1.7.3 declares that bound itself.

## Tests

Tests are `unittest`. The gate CI runs, replicated locally:

```bash
flake8 .
interrogate --verbose .
coverage run -m unittest discover
coverage report
```

`classification.py`, `plots.py` and `job.py` are omitted from coverage —
ROICaT/GPU compute, matplotlib rendering, and orchestration only a real Code
Ocean data run exercises end to end. The remaining modules are held at 100%
rather than diluting the gate to an average.
