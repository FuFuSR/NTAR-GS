# NTAR-GS

**NTAR-GS: Nonstationary Topological Prior-Conditioned Autoregressive Gaussian Splatting for Continuous-Scale Remote Sensing Image Super-Resolution**

> Research implementation of NTAR-GS for continuous-scale remote sensing image super-resolution.  
> **Release status:** This README is a pre-release draft. Items marked **[TO UPDATE]** must be completed and tested before the repository is made public.

<!-- [TO UPDATE] Add the final paper link and bibliographic information when available. -->

## Overview

Remote sensing imagery exhibits spatially heterogeneous structures, ranging from relatively homogeneous regions to directional linear structures and complex object boundaries. NTAR-GS models these differences using an explicit, continuous 2D Gaussian representation and progressive reconstruction.

The method consists of three components:

- **Nonstationary Topological Prior (NTP):** constructs spatially varying structure-aware guidance for Gaussian organization.
- **Parent-Conditioned Autoregressive Gaussian Generation (PCAG):** progressively refines child Gaussians conditioned on their parent states and local topology.
- **Hierarchical Residual Gaussian Splatting (HRGS):** combines coarse content reconstruction with finer residual contributions for image synthesis.

![NTAR-GS architecture](assets/NTAR-GS.png)

## Repository Structure

```text
NTAR-GS/
├── assets/                 # Architecture / visualizations
├── configs/
│   ├── train/              # Training configuration(s)
│   └── test/               # Evaluation configuration(s)
├── datasets/               # Dataset loaders, wrappers, and AID split
├── examples/               # [TO ADD] Sample input images
│   └── input/
├── metrics/
│   ├── __init__.py
│   └── sr_metrics.py        # PSNR and SSIM
├── models/                 # NTAR-GS architecture and encoders
├── scripts/                # Utility scripts
├── train.py                # Training
├── test.py                 # Dataset evaluation
├── inference.py            # Single-image inference
├── requirements.txt        # Python dependencies
└── README.md
```

The Gaussian rasterizer (`gsplat`) is installed separately and is **not** included in this repository.

## Installation

### Environment

The development environment used for the provided code was:

- Python **3.9.25**
- PyTorch **1.13.1+cu117**
- CUDA runtime used by PyTorch **11.7**

A CUDA-capable GPU and a compatible CUDA compilation toolchain are required for the third-party `gsplat` extension. Other versions have not been validated with this release.

```bash
conda create -n ntargs python=3.9 -y
conda activate ntargs

# Install the matching PyTorch/CUDA build.
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117

# Install the remaining requirements after requirements.txt is finalized.
pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu117
```

> **[TO UPDATE]** The currently collected `requirements.txt` is an environment snapshot, not a clean-environment-tested dependency list. Prune and validate it before publishing this installation section.

### Install the 2D Gaussian rasterizer

NTAR-GS calls the `project_gaussians_2d` and `rasterize_gaussians_sum` interfaces from the GaussianImage `gsplat` implementation. Use the pinned upstream revision rather than assuming that the latest package called `gsplat` has the same API:

```bash
# Run from the parent directory of the NTAR-GS repository.
git clone https://github.com/XingtongGe/gsplat.git
cd gsplat
git checkout bf697c8cb0f98627e4a1abed8144cae953a667f7
git submodule update --init --recursive
pip install -e ".[dev]"
cd ../NTAR-GS

python -c "from gsplat.project_gaussians_2d import project_gaussians_2d; from gsplat.rasterize_sum import rasterize_gaussians_sum; print('gsplat import OK')"
```

> Successfully importing the functions verifies the Python API only. A complete end-to-end GPU inference run is still needed to verify the CUDA build.

## Datasets

The experiments use **AID**, **DIOR**, **NWPU-RESISC45**, and **UC Merced**. Please obtain the datasets from their official providers and respect their respective terms of use. Dataset images are not distributed in this repository.

### AID

The included `datasets/AID_split.json` defines **8,000 training / 1,000 validation / 1,000 test** images. The `image-folder` loader supports the `split_file` and `split_key` options and expects the original relative paths found in the split file.

Example directory layout:

```text
data/
└── AID/
    ├── Park/
    │   ├── park_1.jpg
    │   └── ...
    └── ...
```

Update both training and test YAML files to point to your local data directory:

```yaml
root_path: ./data/AID
split_file: ./datasets/AID_split.json
split_key: train  # use val or test for the corresponding split
```

For the simulated continuous-scale setting, LR images are constructed from HR references using bicubic downsampling by the dataset wrapper.

> **[TO UPDATE]** The uploaded `configs/train/train-base-4.yaml` and `configs/test/test-base-4.yaml` still reference `./AID` and `./AID_split.json`. The latter does not match the split-file location in the supplied ZIP. Correct the YAML paths before running the commands below.

> **[TO UPDATE]** Add tested configuration files for DIOR, NWPU-RESISC45 and UC Merced if those evaluations will be supported by the public release. The current uploaded ZIP only includes an AID-oriented test configuration.

## Pretrained Models

The final NTAR-GS checkpoint and its associated training configuration will be made available here:

| Model | Training data | Checkpoint | Notes |
| --- | --- | --- | --- |
| NTAR-GS | AID | **[TO ADD DOWNLOAD LINK]** | Final checkpoint corresponding to the paper |

Place the downloaded checkpoint at `checkpoints/epoch-best-test.pth`, or pass its path using `--model`.

**EDSR encoder initialization:** The supplied training configuration uses the EDSR baseline with 16 residual blocks, 64 features, `scale: 4`, and `no_upsampling: true`. For fresh training, `models/edsr.py` attempts to load `models/weights/edsr_baseline_x4-6b446fab.pt`. A separate link and proper attribution should be provided if this pretrained initialization is required by the released training recipe. The presence of `scale: 4` for the encoder initialization does **not** restrict the NTAR-GS reconstruction scale to ×4.

## Evaluation

`test.py` runs full-image evaluation and reports **PSNR-Y** and **SSIM-Y**. By default, the metric functions accept RGB tensors in `[0,1]` and support 8-bit quantization, MATLAB-style Y conversion, mod-cropping, and optional scale-aware border cropping as configured in the test YAML.

After the dataset paths and checkpoint have been set correctly, an AID ×4 evaluation can be launched using:

```bash
python test.py \
  --config configs/test/test-base-4.yaml \
  --model checkpoints/epoch-best-test.pth \
  --scales 4 \
  --output results/aid_x4 \
  --max_save 0 \
  --gpu 0
```

To evaluate several scales with the same checkpoint:

```bash
python test.py \
  --config configs/test/test-base-4.yaml \
  --model checkpoints/epoch-best-test.pth \
  --scales 2 2.5 3 4 \
  --output results/aid_multiscale \
  --max_save 0 \
  --gpu 0
```

The script writes per-image and scale-level metrics as CSV files, including `multiscale_summary.csv`. Set `--max_save` to a positive number to save a limited number of LR/SR/GT images; omitting it saves all images.

## Single-Image Inference

Sample low-resolution images can be placed in `examples/input/`. A final inference command will use the following argument names, as supported by `inference.py`:

```bash
# [TO ENABLE AFTER FIXING inference.py]
python inference.py \
  --input examples/input/example_01.png \
  --model checkpoints/epoch-best-test.pth \
  --scale 4,4 \
  --output examples/output/example_01_x4.png \
  --gpu 0
```

**Important — [TO FIX BEFORE RELEASE]:** The current uploaded `inference.py` passes the input image to the model without the test-time `[-1,1]` normalization and saves the output without the matching denormalization; it also parses the two scale values as integers. These must be aligned with `test.py` before the example is advertised as working and before non-integer scales can be used from the command line. The output directory must also exist or be created by the script.

## Training

The updated `train.py` supplied separately supports cosine annealing via `CosineAnnealingLR`. Once the **final, actually used** training YAML has been restored and the dataset paths are correct, launch training using:

```bash
python train.py \
  --config configs/train/train-base-4.yaml \
  --name ntargs_aid \
  --gpu 0
```

Training outputs and checkpoints are written under `save/`. The training routine can produce `epoch-best.pth` and `epoch-last.pth`.

> **[TO UPDATE]** The training YAML in the uploaded ZIP is an earlier experimental config (`epoch_max: 300`, `cosine_lr.T_max: 400`), and the `train.py` inside that ZIP does not implement the configured cosine scheduler. Replace the ZIP version with your separately supplied updated `train.py` and restore the YAML from the final training run. Do not label the included legacy YAML as the published training recipe.

## Results

Please refer to the paper for quantitative comparisons and visual results. **[TO UPDATE]** Add the verified final table entries, the exact checkpoint used for each evaluation, and the paper link when available.

## Citation

If you use NTAR-GS in your research, please cite the paper once its bibliographic information is available.

```bibtex
% [TO UPDATE AFTER PUBLICATION]
% Add the verified BibTeX record for the NTAR-GS paper here.
```

## Acknowledgements

This code builds on publicly available research implementations and tools, including:

- [ContinuousSR](https://github.com/peylnog/ContinuousSR) — upstream project context; verify and preserve the applicable license and original attribution for any reused source code.
- [GaussianImage `gsplat`](https://github.com/XingtongGe/gsplat) — third-party 2D Gaussian projection and rasterization implementation (Apache-2.0).
- EDSR and other incorporated model components — add the specific upstream links and licenses where corresponding source code or weights have been reused.

## License

**[TO UPDATE]** The final repository license has not yet been selected. Check all reused code and weight licenses before adding a repository-wide `LICENSE` file or claiming that the entire repository is released under a particular license.
