<div align="center">

# Divide-and-Conquer Inference

### Large-Scale Image Classification with Multimodal Large Language Models

[![Paper status](https://img.shields.io/badge/Paper-Under%20Review-b31b1b?style=for-the-badge)](#paper)
[![Project Page](https://img.shields.io/badge/Project-Page-52e7d1?style=for-the-badge)](https://fourierai.github.io/DCI/)
[![Training Free](https://img.shields.io/badge/Training-Free-2ea44f?style=for-the-badge)](#method)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Zhipeng Ye · Jiaqi Huang · Feng Jiang · Qiufeng Wang · Yikang Duan · Dawei Wang · Xihang Zhou · Qian Qiao**

[Project Page](https://fourierai.github.io/DCI/) ·
[Method](#method) ·
[Reproduction](#paper-protocol) ·
[Results](#results) ·
[Paper alignment](docs/PAPER_ALIGNMENT.md) ·
[Citation](#citation)

</div>

---

## Overview

When a multimodal large language model (MLLM) is asked to classify an image
from a growing list of labels, its mean accuracy generally decreases. The paper
calls this empirical pattern **candidate-space performance degradation**. All
evaluated prompts remain within their models' context windows, so truncation
does not explain the reported trend. Candidate interference and limited
long-context utilization are plausible contributors, not established causes.

**Divide-and-Conquer Inference (DCI)** is a training-free test-time strategy for
this setting. It replaces one monolithic decision with recursively smaller local
decisions, using the same frozen MLLM and no auxiliary classifier.

<p align="center">
  <img src="assets/figures/method.png" width="100%" alt="Divide-and-Conquer Inference method">
</p>

## Method

For an image and a candidate set of size $N$, DCI uses a maximum local group
size $B$:

1. **Divide:** randomly partition the active candidates into disjoint groups of
   at most $B$ labels.
2. **Conquer:** query the same MLLM independently for every group. The shared
   prompt permits one listed label or the literal response `None`.
3. **Combine:** retain at most one non-null prediction from each group and prune
   all null branches.
4. **Recurse:** repeat while more than $B$ candidates remain.

The terminal rules match Algorithm 1 in the manuscript:

- an empty candidate set returns `None` without another model call;
- a singleton candidate set returns that label without another model call;
- a set containing 2 through $B$ labels receives one final local query.

At every non-terminal level,
$n^{(t+1)} \leq \lceil n^{(t)}/B \rceil < n^{(t)}$, so the procedure
terminates for $B \geq 2$. Group calls at the same level are independent and
can be executed in parallel.

<p align="center">
  <img src="assets/figures/prompt.png" width="94%" alt="Prompt used in the DCI conquer phase">
</p>

The implementation uses the paper's fixed prompt and output rule. A response is
accepted only when it contains exactly one valid candidate label. `None`, no
valid label, or multiple valid labels become a null output; invalid successful
responses are not retried.

## Installation

```bash
git clone https://github.com/FourierAI/DCI.git
cd DCI

python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Python 3.9 or later is required. Install development dependencies with
`pip install -e ".[dev]"`.

Start an instruction-tuned vision-language model behind an OpenAI-compatible
endpoint. For example:

```bash
pip install vllm
vllm serve Qwen/Qwen3-VL-2B-Instruct --port 8000
```

Use `--api-base` and `--api-key` for another local or hosted endpoint.

## Dataset setup

Images are not redistributed. The runner validates the paper's evaluation
counts before starting inference.

| Dataset | CLI name | Paper evaluation split | Expected images | Default $B$ |
|:--|:--|:--|--:|--:|
| CIFAR-100 | `cifar100` | complete official test split | 10,000 | 10 |
| CUB-200-2011 | `cub200` | complete official test split | 5,794 | 10 |
| Food-101 | `food101` | complete official test split | 25,250 | 10 |
| ImageNet-1K | `imagenet1k` | official validation split | 50,000 | 10 |
| ImageNet-21K | `imagenet21k` | full-vocabulary stress-test pool | 1,000 sampled/run | 100 |

For CUB-200-2011, keep `images.txt`, `image_class_labels.txt`, `classes.txt`,
and `train_test_split.txt` beside the `images/` directory. For Food-101, keep
`meta/test.txt` beside `images/`. The runner reads these official files directly
instead of using unrelated train or repository-specific splits. See
[`data/README.md`](data/README.md) for exact layouts.

The ImageNet-21K experiment uses the complete 21,843-synset vocabulary and
samples 1,000 distinct images uniformly without replacement from the available
ImageNet-21K pool in each run. It does **not** substitute ImageNet-1K validation
images. If the pool is arranged under WNID directories, build a reusable index:

```bash
dci-index-imagenet21k \
  --image-root /path/to/imagenet21k \
  --output /path/to/imagenet21k-index.json
```

## Quick start

A short smoke test (one run, 20 images) is:

```bash
dci-eval \
  --dataset cifar100 \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --image-root /path/to/cifar100_test_images \
  --b-values 10 \
  --runs 1 \
  --max-samples 20
```

`--k-values` remains a compatibility alias for `--b-values`, but the repository
uses $B$ throughout to match the paper.

## Paper protocol

The defaults use five independent runs, random grouping, and the paper's
dataset-specific $B$. Use the same initial seed, candidate sizes, images,
model endpoint, and decoding configuration for paired baseline/DCI commands.

### Full ImageNet-1K comparison

```bash
dci-eval \
  --dataset imagenet1k \
  --model MODEL \
  --image-root /path/to/imagenet/val \
  --b-values 10 \
  --runs 5 \
  --seed 0

dci-eval \
  --dataset imagenet1k \
  --model MODEL \
  --image-root /path/to/imagenet/val \
  --baseline \
  --runs 5 \
  --seed 0
```

### Candidate-set scaling on ImageNet-1K

For every run and every $N<1000$, the runner independently samples $N$
classes and evaluates all 50 official validation images per selected class. At
$N=1000$, it uses all classes and all 50,000 images.

```bash
dci-eval \
  --dataset imagenet1k \
  --model MODEL \
  --image-root /path/to/imagenet/val \
  --candidate-counts 10 20 100 200 500 1000 \
  --b-values 10 \
  --runs 5 \
  --seed 0
```

Run the same command with `--baseline` for paired monolithic inference.

### ImageNet-21K full-vocabulary stress test

```bash
dci-eval \
  --dataset imagenet21k \
  --model MODEL \
  --image-root /path/to/imagenet21k \
  --metadata /path/to/imagenet21k-index.json \
  --b-values 100 \
  --runs 5 \
  --seed 0
```

The 1,000-image per-run sample is automatic for `imagenet21k`. The default
`--max-retries 0` reproduces the paper's retry-until-response policy; set a
positive value to impose a finite retry limit. Invalid model outputs are never
retried.

### Group-size sweeps

The standard-label-space sweep in the paper uses
`--b-values 2 5 10 20 50`. The ImageNet-21K trade-off figure uses
`--b-values 50 100 500 1000 5000`.

## Outputs and reproducibility

```text
outputs/<dataset>/<model>/
├── n-<N>/run-01/b-<B>.jsonl
├── n-<N>/run-01/b-<B>.txt
├── n-<N>/run-01/b-<B>.manifest.json
├── n-<N>/run-01/baseline.jsonl
└── summary.json
```

- JSONL files resume without recomputing completed images.
- Per-image groupings are deterministic from the run seed and image path, so a
  resumed run uses the same partitions.
- Manifests record the exact candidate list, run seed, prompt hash, full catalog
  hash, environment, command arguments, and Git revision. API keys are omitted.
- `summary.json` reports run-level results and mean ± population SD across runs.
- ImageNet labels that share the same display name are deterministically
  disambiguated, preserving all 1,000 class IDs and all 21,843 synsets.

## Results

Across the 24 model-dataset combinations in the main comparison, DCI has higher
observed mean accuracy in every case, with an overall macro-average gain of
4.67 percentage points. These are empirical results for the evaluated settings,
not a guarantee for every model or deployment.

<p align="center">
  <img src="assets/figures/dci_main.png" width="100%" alt="DCI results on four benchmarks and six MLLMs">
</p>

At $N=1000$ on the ImageNet-1K candidate-scaling experiment, the reported
absolute gains are +9.77 pp (Qwen2.5-VL-7B-Instruct), +10.82 pp
(Qwen3-VL-2B-Instruct), +3.60 pp (Qwen3-VL-8B-Instruct), +18.54 pp
(DeepSeek-VL-7B-chat), +6.58 pp (Kimi-VL-A3B-Instruct), and +5.21 pp
(Gemma-4-E4B).

<p align="center">
  <img src="assets/figures/dci_suppression.png" width="100%" alt="Candidate-set scaling results for DCI and monolithic inference">
</p>

In the full-vocabulary ImageNet-21K stress test, DCI improves all evaluated
local and API-served models. Selected configurations also reduce measured
latency relative to monolithic inference; this is a measured hardware-dependent
result rather than a universal latency guarantee.

<p align="center">
  <img src="assets/figures/imagenet21k_dci_advantage.png" width="100%" alt="Full-vocabulary ImageNet-21K stress-test results">
</p>

The paper's complete figure set is available as source PDF plus web PNG under
[`assets/figures/`](assets/figures/). The project page mirrors the PNGs under
`docs/assets/`.

## Complexity and limitations

For a balanced hierarchy, the paper bounds the worst-case local-call count by
approximately $N/(B-1)+\lceil\log_B N\rceil$. Its squared-attention proxy is

$$
\widetilde W_{\mathrm{attn}}(N,B)
= \frac{N-1}{B-1}(\ell_0+\alpha B)^2.
$$

This proxy exposes a trade-off: increasing $B$ reduces the number of calls but
lengthens each local prompt. DCI can take longer than monolithic inference on
smaller label spaces, and the best $B$ depends on the model and the target
accuracy-latency or accuracy-cost budget.

<p align="center">
  <img src="assets/figures/dci_complexity_analysis.png" width="100%" alt="Squared-attention work proxy for DCI">
</p>

## Paper

**Divide-and-Conquer Inference for Large-Scale Image Classification with
Multimodal Large Language Models**

The manuscript is under review. The repository implementation, protocol, claims,
notation, and figures are tracked against the manuscript in
[`docs/PAPER_ALIGNMENT.md`](docs/PAPER_ALIGNMENT.md).

## Citation

```bibtex
@misc{ye2026dci,
  title  = {Divide-and-Conquer Inference for Large-Scale Image Classification with Multimodal Large Language Models},
  author = {Zhipeng Ye and Jiaqi Huang and Feng Jiang and Qiufeng Wang and Yikang Duan and Dawei Wang and Xihang Zhou and Qian Qiao},
  year   = {2026},
  note   = {Manuscript under review; source code available at \url{https://github.com/FourierAI/DCI}}
}
```

The code is released under the [MIT License](LICENSE). Dataset images remain
subject to their respective licenses and are not redistributed.
