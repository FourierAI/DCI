# Reported experimental results

Aggregate-data snapshot synchronized on **2026-09-03**; protocol documentation
synchronized with the current manuscript on **2026-09-05**.
These files contain the author's reported aggregate results. Means, standard
deviations, latency and cost values are preserved from the current reporting
worksheets; no quantitative benchmark experiments were rerun during this
synchronization. The qualitative files below are recorded inference traces.
The full current protocol and the list of unrecovered historical settings are in
[`../../docs/EXPERIMENT_PROTOCOL.md`](../../docs/EXPERIMENT_PROTOCOL.md).

## Files

- `reported_results.json`: typed numeric values for the eight current quantitative
  figure-data tables, with source worksheet names and column labels.
- `workbooks/DCI数据整理汇总.xlsx`: unchanged consolidated reporting workbook.
- `workbooks/`: eight accompanying source workbooks. The public copy of
  `scaling_performance_data.xlsx` replaces one private source-script path with
  its filename; every other cell value is preserved. Other workbooks are
  byte-identical copies.
- `qualitative/`: the two author-confirmed recorded inference traces for Figure
  12, both with $B=10$. They record untouched responses, exact local-membership
  outcomes, labels, and groupings; private absolute image-directory prefixes are
  omitted.
- `manifest.json`: file hashes for checking this release against its sources.

| Figure / table | JSON table / consolidated worksheet |
|:--|:--|
| Figure 1 | `Scaling Wide` |
| Figure 5 and Table 1 | `Main DCI` (includes six macro-average rows) |
| Figure 6 | `Suppression` |
| Figure 7 and Table 2 | `IN21K Main` |
| Figure 8 | `IN21K Group Size` |
| Figure 9 | `Group Size` |
| Figure 10 | `TTS` (CUB-200-2011 and ImageNet-1K only) |
| Figure 11 | `Grouping` |
| Figure 12 | `qualitative/cub200_B10.json`, `qualitative/food101_B10.json` |

The original workbooks retain older source values and correction notes in
historical worksheets. Use the current reporting sheets identified by each
workbook's README, not a sheet merely named `Raw Data`. Those historical entries
are summary records, not original model prediction logs. The entropy worksheets
and historical ImageNet-21K TTS panel are not part of the current manuscript.

## Units and statistical scope

- Accuracy means are percentages; accuracy SDs and gains are in percentage points.
- Latency is seconds per image; API cost is USD per image.
- The manuscript reports five run-level measurements per setting and summarizes
  them by mean and SD. ImageNet-1K candidate-scaling runs use $50N$ images for
  $N<1000$; the full ImageNet-21K stress test samples 1,000 distinct images per
  run.
- Standard datasets use the complete test/validation splits listed in
  [`../README.md`](../README.md).
- ImageNet-21K uses the first name of each of the 21,843 WNIDs, with exact-name
  deduplication in source order, yielding 20,101 candidate names. WNIDs sharing
  the same first name share a target label; classification accuracy is computed
  against these mapped targets. Synchronizing this protocol did not recompute
  the stored results. See
  [`../../docs/IMAGENET21K_LABEL_PROTOCOL.md`](../../docs/IMAGENET21K_LABEL_PROTOCOL.md).
- Flat and DCI outputs are valid only when the untouched response exactly equals
  one candidate in the corresponding full list or local group. No normalization,
  regular-expression extraction, case conversion, spelling correction, synonym
  mapping, or other post-processing is part of the current paper protocol.
- Original five-run correct/total and prediction logs are unavailable. The means
  include reporting corrections documented in the workbooks; SDs follow the
  manuscript reporting version and have not been recomputed.
- The two author-confirmed qualitative inference traces document individual
  examples, not the five-run benchmark statistics. CUB-200-2011 follows
  $200\to20\to2\to1$ candidates; Food-101 follows $101\to5\to1$.
  Their recorded logger uses `invalid_response` for the same invalid outcome
  named `invalid` by the current validator.

## Provenance boundary

`reported_results.json` is a typed transcription of the current reporting
worksheets. It is not reconstructed from per-image outputs and is not an output
of the current CLI. Because the original five-run predictions and correct/total
records are unavailable, this release cannot independently recover the
run-level measurements, historical SD convention, exact sampled images or group
partitions, or model/backend decoding manifests.

The 20,101-name ImageNet-21K construction and exact raw-string validation are
the current manuscript and executable protocol. They were not used to rerun or
rescore the stored historical aggregates during synchronization. Accordingly,
the files in this directory document author-reported manuscript values; they do
not claim an independent end-to-end reproduction under the current code.
Inferring hypothetical run-level records from rounded means and SDs would not
repair that evidence gap.

## New evaluations

The current CLI computes accuracies directly from `correct / total`, aggregates
unrounded run accuracies and records its chosen `--sd-ddof` convention. It never
snaps experimental accuracies to a preferred reporting step. `--save-traces`
stores untouched local responses and exact validation outcomes for future
audits. New runs
are written to a separate output directory and do not replace these reported
summary files automatically.

The accompanying code implements Flat and random-grouping DCI. Other TTS and
semantic-grouping methods currently have reported data here, not complete
executable reproductions in the CLI.
