# Paper-to-repository alignment

This document tracks the public repository against **Divide-and-Conquer
Inference for Large-Scale Image Classification with Multimodal Large Language
Models**, synchronized to the current manuscript on 2026-09-05. Version 0.4.0
aligns the executable Flat/DCI templates, recursive procedure, dataset loaders,
and exact raw-string validation with the current method description. Summary
results are preserved from the author's reporting workbooks; no quantitative
benchmark evaluation was rerun for this synchronization.

The consolidated settings and explicitly unrecovered fields are listed in
[`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md).

## Method and implementation

| Manuscript requirement | Repository behavior |
|:--|:--|
| Candidate set size is $N$; maximum local group size is $B$ | CLI and outputs use `N` and `B`; `--k-values` is only a compatibility alias |
| Randomly partition each non-terminal set into disjoint groups of at most $B$ | A deterministic per-image/run RNG shuffles before chunking; changing the run seed regenerates groups |
| Use one fixed MLLM and one fixed prompt across levels | `DCIClassifier` reuses the client, model, prompt, and decoding arguments |
| Raw output can be a local label, `None`, or invalid | `exact-raw-string-v1`: the untouched response must equal a supplied candidate exactly; only the literal `None` has null status; all other strings are invalid |
| `None` and invalid outputs are filtered | Both map to null; optional traces distinguish `none` and `invalid` |
| Flat and DCI share the task instruction but differ in the `None` option | Separate author-supplied `FLAT_PROMPT` and `DCI_PROMPT`; `--baseline` invokes `classify_flat` |
| Empty combined set returns null | `classify` returns Python `None`, serialized as JSON `null`, without a fallback global call |
| Singleton set returns its only label | Returned without another model call |
| Two through $B$ survivors receive one final local call | The same prompt, including the `None` option, is used for the final call |
| Same-level group calls are independent and optionally parallel | `ThreadPoolExecutor` runs group requests concurrently up to `--max-workers` |
| API failures are retried; invalid successful outputs are not | `--max-retries 0` retries API errors until a response; invalid content is immediately treated as null |

Exact raw-string validation performs no whitespace trimming, wrapper or
punctuation removal, quote or code-fence stripping, case conversion, spelling
correction, synonym lookup, substring extraction, or semantic remapping. It is
used by Flat inference, every recursive DCI level, and the final local call. A
global-vocabulary label that is absent from the current DCI group is invalid and
is removed before the next level.

## Experimental protocol

| Manuscript setting | Repository support |
|:--|:--|
| Five run-level measurements and mean ± SD | `--runs 5`; new summaries record `sd_ddof` (default 0, optional 1). Published SDs are retained unchanged without inferring their convention |
| Standard experiments use random grouping and $B=10$ | Default for CIFAR-100, CUB-200-2011, Food-101, and ImageNet-1K |
| CIFAR-100 complete official test split | 10,000 images; validated before inference |
| CUB-200-2011 complete official test split | Reads official `train_test_split.txt`; requires exactly 5,794 images |
| Food-101 complete official test split | Reads official `meta/test.txt`; requires exactly 25,250 images |
| ImageNet-1K official validation split | Bundled 50,000-image index with all 1,000 class IDs preserved |
| Candidate sizes $N\in\{10,20,100,200,500,1000\}$ | `--candidate-counts`; class subsets are independently sampled for each run and $N$ |
| All 50 validation images for each selected ImageNet-1K class | Default behavior when no image cap is supplied |
| Candidate scaling at $N=1000$ | Uses the fixed complete 1,000-class, 50,000-image validation set; grouping and model generation are repeated across runs |
| ImageNet-21K vocabulary derived from all 21,843 WNIDs | Select each WNID's first name; preserve case and underscores; remove exact duplicate names in source order, yielding 20,101 candidate names |
| ImageNet-21K classification accuracy | Retain every catalog WNID-to-first-name mapping; map each indexed image through its WNID; same-name WNIDs share a target label |
| 1,000 distinct ImageNet-21K images sampled uniformly per run | Automatic dataset default; uses the indexed ImageNet-21K pool, not ImageNet-1K images |
| Paired baseline/DCI use identical classes and images | Separate commands with the same seed and candidate counts select the same problem instances |
| Model-specific default decoding | Temperature, top-p, and maximum-token overrides are omitted unless explicitly supplied; manifests record their values, including null when the endpoint default is retained |
| Metadata for new evaluations | Manifests store candidate labels, image lists, seeds, catalog/prompt hashes, validation version, arguments, environment, Git revision, and tracked-worktree state |
| Consistent resumed evaluations | Existing records cannot be reused after changes to prompts, validation, sampling, model or decoding configuration |
| End-to-end latency per image | Image preprocessing and all required model calls are timed; complete stored image timings are aggregated after resuming |

## Claims and terminology

- The repository uses **candidate-space performance degradation**, matching the
  current empirical framing in the manuscript.
- Candidate interference and limited long-context utilization are described as
  plausible contributors to the observed difficulty.
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
| Combined CUB-200-2011 / Food-101 trajectories, both $B=10$ (Fig. 12) | `dci_B10` |

The legacy descriptive filenames `dci-framework`, `conquer-prompt`,
`complexity-analysis`, and `scaling-results` are synchronized aliases so
existing external links do not show stale figures.
The old `cub_dci` and `food_dci` URLs now resolve to copies of the combined
Figure 12; no current Figure 13 is claimed.

## Reported data and reproducibility boundary

The first-name construction is documented in
[`IMAGENET21K_LABEL_PROTOCOL.md`](IMAGENET21K_LABEL_PROTOCOL.md). This update
changes the executable label protocol and its description to match the current
manuscript; it does not rerun quantitative benchmarks or rescore historical
aggregate results.

[`../data/results/`](../data/results/README.md) contains the current summary
workbook, its source workbooks, machine-readable current figure data, and a
file-hash manifest. Earlier source entries and correction notes remain in their
original worksheets. The machine-readable tables transcribe the current
reporting worksheets; they are not regenerated from the CLI.

Original five-run prediction logs and correct/total records are unavailable.
Consequently, this public snapshot cannot independently reconstruct the five
run-level measurements, infer the historical SD convention, recover the exact
ImageNet-21K image samples and group partitions, or demonstrate that the
reported aggregates were generated by the current executable protocol. In
particular, adopting the 20,101-name first-name protocol did not trigger a model
rerun or historical rescoring. The aggregates are retained as the manuscript's
author-reported values, not claimed as a repository reproduction. New runs save
their own records and must be treated as separate measurements.

The two author-confirmed recorded inference traces illustrate individual
examples only. They record exact raw validity outcomes and are separate from the
full-benchmark summary results. New runs can save raw group responses using
`--save-traces`, without changing the reported aggregates.

This repository implements random-grouping DCI and Flat inference. It does not
yet include executable CoT/PaS/SC/SA/D&A or semantic-grouping evaluation pipelines;
their reported summary values and figures are included in the data release.

## Synchronization checks for version 0.4.0

- Exact template-string equality with the manuscript's accompanying helper file.
- Offline tests of Flat/DCI dispatch, exact raw-string validation, null/invalid
  pruning, termination, pairing, metadata and resumed-run timing.
- Current figure data matched against the author's reporting workbook; all
  current vector figures copied without content edits and web PNGs regenerated.
- Author order, affiliations, current $B=10$ qualitative examples and citation
  metadata synchronized across the README and project page.

The runtime does not modify reported means or apply nearest-step rounding to new
experimental results. New mean accuracy and SD are computed from run records.
