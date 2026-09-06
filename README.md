<div align="center">

# Divide-and-Conquer Inference

### Large-Scale Image Classification with Multimodal Large Language Models

[![Paper status](https://img.shields.io/badge/Paper-Under%20Review-b31b1b?style=for-the-badge)](#paper)
[![Project Page](https://img.shields.io/badge/Project-Page-52e7d1?style=for-the-badge)](https://fourierai.github.io/DCI/)
[![Training Free](https://img.shields.io/badge/Training-Free-2ea44f?style=for-the-badge)](#method)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Zhipeng Ye<sup>1</sup> · Jiaqi Huang<sup>1,2</sup> · Feng Jiang<sup>1,*</sup> · Qiufeng Wang<sup>2</sup> · Yikang Duan<sup>2</sup> · Dawei Wang<sup>1</sup> · Kaixin Liu<sup>1</sup> · Hao Li<sup>3</sup>**

<sup>1</sup> Taizhou Institute of Science and Technology, Nanjing University of Science and Technology, China<br>
<sup>2</sup> Department of Intelligent Science, Xi’an Jiaotong-Liverpool University, China<br>
<sup>3</sup> Department of Computer Science, University of Arizona, USA<br>
<sup>*</sup> Corresponding author: Feng Jiang · jf@nustti.edu.cn<br>
Jiaqi Huang: Jiaqi.Huang26@student.xjtlu.edu.cn

[Project Page](https://fourierai.github.io/DCI/) ·
[Method](#method) ·
[Quick Start](#quick-start) ·
[Final Results](#results) ·
[Citation](#citation)

</div>

---

## Overview

**Divide-and-Conquer Inference (DCI)** is a training-free strategy for
large-scale image classification with multimodal large language models.
It replaces a single large candidate-list decision with bounded local
decisions and recursively reduces the candidate space, reusing the same
frozen model without additional training.

<p align="center">
  <img src="assets/figures/method.png" width="100%" alt="Divide-and-Conquer Inference method">
</p>

## Method

1. **Divide:** partition the active labels into groups of at most $B$ candidates.
2. **Conquer:** query the same MLLM independently for each group.
3. **Combine:** retain valid local predictions and discard `None` or invalid outputs.
4. **Recurse:** continue with the surviving candidates until a final decision.

An empty set returns `None`; a singleton returns its label; a set of two
through $B$ candidates receives one final local query. Same-level queries
can run in parallel.

The core implementation is in [`dci/runner.py`](dci/runner.py), with prompts
in [`dci/prompts.py`](dci/prompts.py) and validation in
[`dci/validation.py`](dci/validation.py). Validity requires an exact match
between the untouched response and a candidate supplied to that call.
Flat inference asks for one label; DCI additionally permits `None`.

## Installation

Python 3.9 or later is required.

```bash
git clone https://github.com/FourierAI/DCI.git
cd DCI
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Use an instruction-tuned vision-language model served through an
OpenAI-compatible endpoint. For example:

```bash
pip install vllm
vllm serve Qwen/Qwen3-VL-2B-Instruct --port 8000
```

## Quick start

After preparing the dataset images, a small local run is:

```bash
dci-eval \
  --dataset cifar100 \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --image-root /path/to/cifar100_test_images \
  --b-values 10 \
  --runs 1 \
  --max-samples 20
```

Use `--api-base` and `--api-key` for the serving endpoint, and
`--baseline` for Flat inference. See `dci-eval --help` for other options.
Outputs are saved locally under `outputs/`, which is excluded from Git.

### Data inputs

Obtain dataset images from their official sources under the applicable
licenses. Only the label resources needed by the loaders are bundled:

| Dataset | Input |
|:--|:--|
| CIFAR-100 | Images matching `data/metadata/cifar100/labels.json`, or a mapping supplied with `--metadata` |
| ImageNet-1K | Validation images matching the bundled split index, or an index supplied with `--metadata` |
| CUB-200-2011 | Official `images.txt`, `image_class_labels.txt`, `classes.txt`, and `train_test_split.txt` beside `images/` |
| Food-101 | Official `meta/test.txt` beside `images/` |
| ImageNet-21K | Images in WNID directories; the loader uses the bundled WNID label catalog |

For ImageNet-21K, an optional local index can be created with:

```bash
dci-index-imagenet21k \
  --image-root /path/to/imagenet21k \
  --output /path/to/imagenet21k-index.json
```

Core-code tests use offline inputs:

```bash
pip install -e ".[dev]"
pytest -q
```

## Results

The [final aggregate results](data/results/README.md) contain the numerical
tables reported in the manuscript. Means, standard deviations, gains, and
units are available in [`reported_results.json`](data/results/reported_results.json).

- **Main comparison:** higher observed mean accuracy in all 24 model-dataset
  pairs, with a **4.67 percentage-point** macro-average gain.
- **Large candidate spaces:** at 1,000 ImageNet-1K candidates, the six-model
  mean gain is **9.09 percentage points**.
- **Selected large-vocabulary configurations:** accuracy and measured latency
  both improve over Flat inference; these gains are configuration dependent,
  not a universal speed guarantee.

<p align="center">
  <img src="assets/figures/dci_main.png" width="100%" alt="DCI main results on four benchmarks and six MLLMs">
</p>

<p align="center">
  <img src="assets/figures/dci_suppression.png" width="100%" alt="Candidate-set scaling results for DCI and Flat inference">
</p>

<p align="center">
  <img src="assets/figures/imagenet21k_dci_advantage.png" width="100%" alt="Full-vocabulary ImageNet-21K stress-test results">
</p>

The ImageNet-21K stress test uses **20,101 distinct candidate names**.
Candidate-list coverage is distinct from the number of image classes
evaluated. At small label spaces, DCI can add latency and does not improve
every setting.

All 12 final manuscript figures are available as PDF and PNG in
[`assets/figures/`](assets/figures/). The [project page](https://fourierai.github.io/DCI/)
presents the method and final results, including the qualitative visualization.

## Repository scope

This repository contains core inference code and its required label resources,
final aggregate results, and project-page assets. Dataset images, intermediate
outputs, individual-run traces, working spreadsheets, and experiment-process
documents are not distributed here.

## Paper

**Divide-and-Conquer Inference for Large-Scale Image Classification with
Multimodal Large Language Models**

The manuscript is under review.

## Citation

```bibtex
@misc{ye2026dci,
  title  = {Divide-and-Conquer Inference for Large-Scale Image Classification with Multimodal Large Language Models},
  author = {Zhipeng Ye and Jiaqi Huang and Feng Jiang and Qiufeng Wang and Yikang Duan and Dawei Wang and Kaixin Liu and Hao Li},
  year   = {2026},
  note   = {Manuscript under review; source code available at \url{https://github.com/FourierAI/DCI}}
}
```

The research code is released under the [MIT License](LICENSE). Dataset
images remain subject to their respective licenses and are not redistributed.
