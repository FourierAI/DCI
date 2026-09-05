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
[Run protocol](#paper-protocol) ·
[Results](#results) ·
[Reported data](data/results/README.md) ·
[Experiment protocol](docs/EXPERIMENT_PROTOCOL.md) ·
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
long-context utilization are plausible contributors to the observed difficulty.

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
3. **Combine:** validate each output against its local group, retain at most one
   valid category, and discard `None` and invalid responses.
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

The exact author-supplied templates are in [`dci/prompts.py`](dci/prompts.py).
**Flat and DCI use different output instructions:** Flat requests one category
from the complete vocabulary; DCI additionally permits `None`. DCI retains this
instruction in its final local call, including the single-call case $N=B$.

[`dci/validation.py`](dci/validation.py) applies the manuscript's literal raw-output
rule. The untouched response is selected only when it is exactly equal to one
candidate string supplied in that call. The literal response `None` is recorded
as a null decision; every other response is invalid. Validation does not trim
whitespace, remove answer wrappers, punctuation, quotes, or code fences, change
case, correct spelling, split text, or perform substring, synonym, or semantic
matching. For example, `Greek_salad` is invalid when the local candidate is
`greek_salad`, and `tiger cat` is not accepted as `cat`. Each ImageNet-21K
candidate is the first listed name of a WNID after exact-name deduplication;
other synonyms in the source catalog are not accepted as alternate responses.
For DCI, a label that belongs to the global vocabulary is still invalid when it
is absent from the current local group, and it is removed before the next level.
The same validation applies to Flat, every recursive DCI level, and the final
local call. Invalid successful responses are not retried; a null final result
counts as incorrect and stays in the denominator.

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

| Dataset | CLI name | Paper evaluation split | Candidates | Images | Default $B$ |
|:--|:--|:--|--:|--:|--:|
| CIFAR-100 | `cifar100` | complete official test split | 100 | 10,000 | 10 |
| CUB-200-2011 | `cub200` | complete official test split | 200 | 5,794 | 10 |
| Food-101 | `food101` | complete official test split | 101 | 25,250 | 10 |
| ImageNet-1K | `imagenet1k` | official validation split | 1,000 | 50,000 | 10 |
| ImageNet-21K | `imagenet21k` | available full-vocabulary stress-test pool | 20,101 | 1,000 sampled/run | 100 |

For CUB-200-2011, keep `images.txt`, `image_class_labels.txt`, `classes.txt`,
and `train_test_split.txt` beside the `images/` directory. For Food-101, keep
`meta/test.txt` beside `images/`. The runner reads these official files directly
instead of using unrelated train or repository-specific splits. See
[`data/README.md`](data/README.md) for exact layouts.

The ImageNet-21K vocabulary is constructed from all 21,843 WNID entries by taking
the first comma-separated name after each WNID, preserving case and underscores,
and removing exact duplicate names in source order. This produces **20,101
distinct candidate names**. WNIDs with the same first name share one target
label; classification accuracy is computed against these mapped targets. See
[`docs/IMAGENET21K_LABEL_PROTOCOL.md`](docs/IMAGENET21K_LABEL_PROTOCOL.md) and the
exported [`first_names.txt`](data/metadata/imagenet21k/first_names.txt).

Each run samples 1,000 distinct images uniformly without replacement from the
available ImageNet-21K pool. If the pool is arranged under WNID directories,
build a reusable index:

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

The consolidated model, dataset, sampling, validation, failure-handling,
latency, cost, and provenance specifications are in
[`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md).

The defaults use five runs, random grouping, and the paper's dataset-specific
$B$. The CLI starts from seed 0 unless overridden and derives one run seed per
run. Use the same initial seed, candidate sizes, images, model endpoint, and
decoding configuration for paired baseline/DCI commands.
The baseline flag selects the Flat template independently of $B$. Optional
`--temperature`, `--top-p`, and `--max-tokens` overrides are recorded in the run
manifest; when omitted, the serving defaults are retained. Record the endpoint's
model revision, preprocessing, context limit, batching, GPU allocation and cache
configuration alongside the output directory. The CLI's default concurrency of
10 is a configurable software default, not a recovered paper measurement setting.
All Flat and DCI responses use exact raw-string validation as described above.

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
classes and evaluates all 50 official validation images per selected class.
Subsets are not nested across different values of $N$. At $N=1000$, it uses all
classes and all 50,000 images; grouping and model generation are repeated across
runs.

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

The 1,000-image per-run sample is automatic for `imagenet21k`. For new runs, the
default `--max-retries 0` follows the paper's retry-until-response rule; set a
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

- JSONL files resume without recomputing completed images. Changed prompts,
  validation rules, candidate selections or decoding settings require a fresh
  output directory to prevent mixing incompatible predictions.
- Per-image groupings are deterministic from the run seed and the image's stable
  dataset-relative identifier, so moving an unchanged dataset does not alter its
  partitions and a resumed run uses the same groups.
- Manifests record the exact candidate list, run seed, prompt hash, full catalog
  hash, environment, command arguments, Git revision, and whether tracked files
  in the worktree were dirty. API keys are omitted.
- Each JSONL record stores the prediction, target, call count and per-image
  end-to-end latency. `--save-traces` also records local candidates, raw responses
  and their `selected` / `none` / `invalid` validation outcomes.
- `summary.json` reports run-level correct/total counts, accuracies and mean ± SD
  for newly executed runs.
  `--sd-ddof 0` (default) uses population SD; `--sd-ddof 1` uses sample SD. The
  selected convention is stored explicitly. Accuracy is calculated from counts
  without step snapping. Stored image timings support complete-run latency
  summaries after a resumed run.
- The ImageNet-1K loader disambiguates repeated display names with class IDs,
  preserving its 1,000 candidate classes. ImageNet-21K uses first names with
  exact-name deduplication: its WNID catalog retains 21,843 entries and its
  candidate list contains 20,101 names. No WNID suffix is added to these names.

## Results

The [reported-results release](data/results/README.md) is an aggregate-data
snapshot synchronized with the current manuscript. Reported means and SDs were
copied from the reporting workbooks, whose correction history is retained.
Original five-run predictions and correct/total records are unavailable, and
the 20,101-name ImageNet-21K protocol synchronization did not rerun models or
rescore those historical aggregates. The release therefore documents the
author-reported manuscript values; it is not presented as output regenerated by
the current CLI. New CLI runs produce separate, auditable measurements and do
not overwrite this release.

Across the 24 model-dataset combinations in the main comparison, DCI has higher
observed mean accuracy in every case, with an overall macro-average gain of
4.67 percentage points.

<p align="center">
  <img src="assets/figures/dci_main.png" width="100%" alt="DCI results on four benchmarks and six MLLMs">
</p>

At $N=1000$ on the ImageNet-1K candidate-scaling experiment, the reported
absolute gains are +9.77 pp (Qwen2.5-VL-7B-Instruct), +10.82 pp
(Qwen3-VL-2B-Instruct), +3.60 pp (Qwen3-VL-8B-Instruct), +18.54 pp
(DeepSeek-VL-7B-chat), +6.58 pp (Kimi-VL-A3B-Instruct), and +5.21 pp
(Gemma-4-E4B).

At $N=B=10$, the six-model mean gain is **−1.05 pp**. Kimi-VL-A3B-Instruct
has Flat **94.28%** and DCI **93.16%** (−1.12 pp). The small-set comparison
retains the different Flat/DCI instructions; both methods make one call.

<p align="center">
  <img src="assets/figures/dci_suppression.png" width="100%" alt="Candidate-set scaling results for DCI and monolithic inference">
</p>

In the full-vocabulary ImageNet-21K stress test, DCI improves all evaluated
local and API-served models. Selected configurations also reduce measured
latency relative to monolithic inference; this is a measured hardware-dependent
result rather than a universal latency guarantee.

At $B=100$, Qwen2.5-VL-7B-Instruct changes from **1.56%** to **37.88%**
(+36.32 pp). At $B=500$, Qwen3-VL-2B-Instruct obtains **19.70%** at
**5.99 s/image**, compared with Flat **2.12%** at **7.96 s/image**;
Gemma-4-E4B obtains **15.82%** at **6.88 s/image**, compared with Flat
**2.14%** at **16.39 s/image**. Full mean ± SD values are in the data release.

<p align="center">
  <img src="assets/figures/imagenet21k_dci_advantage.png" width="100%" alt="Full-vocabulary ImageNet-21K stress-test results">
</p>

The paper's complete figure set is available as source PDF plus web PNG under
[`assets/figures/`](assets/figures/). The project page mirrors the PNGs under
`docs/assets/`.

### Qualitative examples (Figure 12)

Both examples use **Qwen3-VL-2B-Instruct, $B=10$**. The candidate counts are
**200 → 20 → 2 → 1** for CUB-200-2011 and **101 → 5 → 1** for Food-101.
`None` and invalid outputs are removed before the next level.

<p align="center">
  <img src="assets/figures/dci_B10.png" width="100%" alt="Combined B=10 CUB-200-2011 and Food-101 candidate-pruning trajectories">
</p>

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
  author = {Zhipeng Ye and Jiaqi Huang and Feng Jiang and Qiufeng Wang and Yikang Duan and Dawei Wang and Kaixin Liu and Hao Li},
  year   = {2026},
  note   = {Manuscript under review; source code available at \url{https://github.com/FourierAI/DCI}}
}
```

The code is released under the [MIT License](LICENSE). Dataset images remain
subject to their respective licenses and are not redistributed.
