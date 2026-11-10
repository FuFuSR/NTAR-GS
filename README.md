# NTAR-GS: Nonstationary Topological Prior-Conditioned Autoregressive Gaussian Splatting for Continuous-Scale Remote Sensing Image Super-Resolution

- Authors: Jiao Fu, Yuanping Xia<sup>&#42;</sup>, and Jing Chen<sup>&#42;</sup>
- East China University of Technology and Wuhan University

## Overview

We propose NTAR-GS, a novel model for continuous-scale remote sensing image super-resolution (SR) based on 2D Gaussian Splatting (2D GS). It unifies adaptation to spatial structures and cross-level generative dependencies within an explicit continuous Gaussian representation. The model uses a nonstationary topological prior (NTP) to constrain the organization of Gaussian primitives, recursively generates child Gaussians through parent-conditioned autoregressive Gaussian generation (PCAG), and incorporates hierarchical residual Gaussian Splatting (HRGS) to progressively recover local details by building on inherited structures. Experimental results show that NTAR-GS exhibits advantages over state-of-the-art SR methods in both quantitative evaluation and visual reconstruction quality, while demonstrating good generalization in scale extrapolation and real-world image tests. Compared with other SR methods based on Gaussian representations, the proposed model achieves an average PSNR gain exceeding 0.15 dB and an inference speedup of up to 23.78×.

<p align="center">
  <img src="assets/NTAR-GS.png" alt="Overall architecture of NTAR-GS" width="100%">
</p>

## Dependencies and Installation

- Python 3.9
- PyTorch 1.13.1 / torchvision 0.14.1
- CUDA 11.7
- BasicSR 1.3.4.9

Other dependencies:

```bash
pip install -r requirements.txt

git clone https://github.com/XingtongGe/gsplat.git
cd gsplat
git checkout bf697c8
pip install -e ".[dev]"
```

## Dataset Preparation

We use AID, DIOR, and NWPU-RESISC45 for simulated SR experiments, treating the original images as HR references and generating corresponding LR images through bicubic downsampling. UC Merced is used for real-world experiments, with the original images directly used as inputs without any simulated degradation. Download the original images from the official links below, or use your own images for inference.

| Dataset | Usage | Number of images |
| --- | --- | --- |
| [AID](https://captain-whu.github.io/AID/) | train / val / test | 8,000 / 1,000 / 1,000 |
| [DIOR](https://gcheng-nwpu.github.io/#Datasets) | cross-dataset test | 1,000 |
| [NWPU-RESISC45](https://gcheng-nwpu.github.io/#Datasets) | cross-dataset test | 900 |
| [UC Merced](https://vision.ucmerced.edu/datasets/) | Real-world no-reference evaluation | 420 |

## Usage
### Pretrained Model
Download the pretrained NTAR-GS model (trained on AID) from [Baidu Netdisk](https://pan.baidu.com/s/1gdT128vMcv8xtSJSA16pXA?pwd=68sc) and place `NTAR-GS.pth` in the project root directory.

### Inference

Run the following command from the project root directory to perform super-resolution on a single image:

```bash
python inference.py --input examples/input/playground_279.png --model NTAR-GS.pth --scale 4 --output examples/output/playground_279_x4.png
```
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
