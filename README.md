
<div align="center">

<h1>NTAR-GS: Nonstationary Topological Prior-Conditioned Autoregressive Gaussian Splatting for Continuous-Scale Remote Sensing Image Super-Resolution</h1>

<p>
Jiao Fu &nbsp;&nbsp;
Yuanping Xia &nbsp;&nbsp;
Jing Chen
</p>

</div>



## Overview

We propose NTAR-GS, a novel model for continuous-scale remote sensing image super-resolution (SR) based on 2D Gaussian Splatting (2D GS). It unifies adaptation to spatial structures and cross-level generative dependencies within an explicit continuous Gaussian representation. The model uses a nonstationary topological prior (NTP) to constrain the organization of Gaussian primitives, recursively generates child Gaussians through parent-conditioned autoregressive Gaussian generation (PCAG), and incorporates hierarchical residual Gaussian Splatting (HRGS) to progressively recover local details by building on inherited structures. Experimental results show that NTAR-GS exhibits advantages over state-of-the-art SR methods in both quantitative evaluation and visual reconstruction quality, while demonstrating good generalization in scale extrapolation and real-world image tests. Compared with other SR methods based on Gaussian representations, the proposed model achieves an average PSNR gain exceeding 0.15 dB and an inference speedup of up to 23.78×.

<p align="center">
  <img src="assets/NTAR-GS.png" alt="Overall architecture of NTAR-GS" width="100%">
</p>

## Dependencies and Installation

### Environment

- Linux
- Python 3.9
- PyTorch 1.13.1
- torchvision 0.14.1
- CUDA 11.7

编译 gsplat 需要本机安装 CUDA 工具包（包含 nvcc）及兼容的 C++ 编译器。

### Installation

创建并激活 Conda 环境：

```bash
conda create -n ntargs python=3.9 -y
conda activate ntargs
python -m pip install "setuptools<70" wheel
```

下载 NTAR-GS：

```bash
git clone https://github.com/FuFuSR/NTAR-GS.git
cd NTAR-GS
```

安装 PyTorch 和项目依赖：

```bash
python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

python -m pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu117 \
    --no-build-isolation
```

安装第三方 2D 高斯光栅化库 gsplat：

```bash
cd ..
git clone https://github.com/XingtongGe/gsplat.git
cd gsplat
git checkout bf697c8cb0f98627e4a1abed8144cae953a667f7
git submodule update --init --recursive
python -m pip install -e ".[dev]" --no-build-isolation
cd ../NTAR-GS
```

本项目使用 XingtongGe/gsplat 的上述固定版本。
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
