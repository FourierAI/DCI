# Paper-to-repository alignment

This document tracks the public repository against **Divide-and-Conquer
Inference for Large-Scale Image Classification with Multimodal Large Language
Models**. The implementation follows the submitted manuscript's Algorithm 1
and Section 4 protocol.

## Method and implementation

| Manuscript requirement | Repository behavior |
|:--|:--|
| Candidate set size is $N$; maximum local group size is $B$ | CLI and outputs use `N` and `B`; `--k-values` is only a compatibility alias |
| Randomly partition each non-terminal set into disjoint groups of at most $B$ | A deterministic per-image/run RNG shuffles before chunking; changing the run seed regenerates groups |
| Use one fixed MLLM and one fixed prompt across levels | `DCIClassifier` reuses the client, model, prompt, and decoding arguments |
| Every local output is one listed label or `None` | Normalization accepts exactly one valid candidate; `None`, zero matches, or multiple matches become null |
| Empty combined set returns null | `classify` returns Python `None`, serialized as JSON `null`, without a fallback global call |
| Singleton set returns its only label | Returned without another model call |
| Two through $B$ survivors receive one final local call | The same prompt, including the `None` option, is used for the final call |
| Same-level group calls are independent and optionally parallel | `ThreadPoolExecutor` runs group requests concurrently up to `--max-workers` |
| API failures are retried; invalid successful outputs are not | `--max-retries 0` retries API errors until a response; invalid content is immediately treated as null |

## Experimental protocol

| Manuscript setting | Repository support |
|:--|:--|
| Five run-level measurements and mean ± SD | `--runs 5` is the default; `summary.json` reports run values, mean, and population SD |
| Standard experiments use random grouping and $B=10$ | Default for CIFAR-100, CUB-200-2011, Food-101, and ImageNet-1K |
| CIFAR-100 complete official test split | 10,000 images; validated before inference |
| CUB-200-2011 complete official test split | Reads official `train_test_split.txt`; requires exactly 5,794 images |
| Food-101 complete official test split | Reads official `meta/test.txt`; requires exactly 25,250 images |
| ImageNet-1K official validation split | Bundled 50,000-image index with all 1,000 class IDs preserved |
| Candidate sizes $N\in\{10,20,100,200,500,1000\}$ | `--candidate-counts`; class subsets are independently sampled for each run and $N$ |
| All 50 validation images for each selected ImageNet-1K class | Default behavior when no image cap is supplied |
| Full 21,843-synset ImageNet-21K vocabulary | Full descriptions plus deterministic WNID disambiguation retain exactly 21,843 candidates |
| 1,000 distinct ImageNet-21K images sampled uniformly per run | Automatic dataset default; uses the indexed ImageNet-21K pool, not ImageNet-1K images |
| Paired baseline/DCI use identical classes and images | Separate commands with the same seed and candidate counts select the same problem instances |
| Sampling metadata retained | Manifests store candidate labels, seeds, catalog/prompt hashes, arguments, environment, and Git revision |

## Claims and terminology

- The repository uses **candidate-space performance degradation**, matching the
  manuscript. It does not present “Performance Collapse in Long Sequence
  Recognition” as a formal named phenomenon.
- Candidate interference and limited long-context utilization are described as
  plausible contributors, not proven mechanisms.
- Complexity statements refer to the manuscript's squared-attention **proxy**.
  They are not presented as measured FLOPs or a universal runtime guarantee.
- Latency claims are limited to selected configurations in the full-vocabulary
  ImageNet-21K stress test. Smaller label spaces can incur a latency-accuracy
  trade-off.
- Reported gains are observed means for the evaluated models, datasets, and
  serving configurations; they are not universal guarantees.

## Figure synchronization

Every manuscript figure supplied with the final paper is versioned as a vector
PDF and a 2,600-pixel-wide PNG in `assets/figures/`. The project page mirrors
the PNG files in `docs/assets/`.

| Manuscript figure/content | Source asset basename |
|:--|:--|
| Candidate-space scaling overview (Fig. 1) | `scaling_performance` |
| DCI framework (Fig. 2) | `method` |
| Conquer prompt (Fig. 3) | `prompt` |
| Squared-attention proxy (Fig. 4) | `dci_complexity_analysis` |
| Main benchmark results (Fig. 5) | `dci_main` |
| Candidate-set scaling with DCI (Fig. 6) | `dci_suppression` |
| Full-vocabulary ImageNet-21K comparison (Fig. 7) | `imagenet21k_dci_advantage` |
| ImageNet-21K group-size trade-off (Fig. 8) | `dci_imagenet21k_tradeoff` |
| Standard-space group-size trade-off (Fig. 9) | `dci_group_size_tradeoff` |
| Test-time scaling comparison (Fig. 10) | `tts_accuracy_latency_tradeoff` |
| Grouping-strategy ablation (Fig. 11) | `gs_ablation` |
| CUB-200-2011 pruning trajectory (Fig. 12) | `cub_dci` |
| Food-101 pruning trajectory (Fig. 13) | `food_dci` |

The legacy descriptive filenames `dci-framework`, `conquer-prompt`,
`complexity-analysis`, and `scaling-results` are synchronized aliases so
existing external links do not show stale figures.
