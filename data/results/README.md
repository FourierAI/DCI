# Reported experimental results

Snapshot synchronized with the current 36-page manuscript on **2026-09-03**.
These files contain the author's reported aggregate results. Means, standard
deviations, latency and cost values are preserved from the current reporting
worksheets; no model experiments were rerun during this synchronization.

## Files

- `reported_results.json`: typed numeric values for the eight current quantitative
  figure-data tables, with source worksheet names and column labels.
- `workbooks/DCI数据整理汇总.xlsx`: unchanged consolidated reporting workbook.
- `workbooks/`: eight accompanying source workbooks. The public copy of
  `scaling_performance_data.xlsx` replaces one private source-script path with
  its filename; every other cell value is preserved. Other workbooks are
  byte-identical copies.
- `qualitative/`: the two supplied individual $B=10$ traces for Figure 12. Only
  private absolute image-directory prefixes were removed; responses, labels and
  grouping records are preserved.
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
- Reported summaries use five runs. ImageNet-1K candidate-scaling runs use $50N$
  images; the full ImageNet-21K stress test samples 1,000 distinct images per run.
- Standard datasets use the complete test/validation splits listed in
  [`../README.md`](../README.md).
- Original five-run correct/total or prediction logs are unavailable. The means
  include author-confirmed reporting corrections, documented in the workbooks;
  SDs follow the manuscript reporting version and have not been recomputed.
- The two qualitative traces document individual examples, not the five-run
  benchmark statistics. They preserve the original recorded grouping order.

## New evaluations

The current CLI computes accuracies directly from `correct / total`, aggregates
unrounded run accuracies and records its chosen `--sd-ddof` convention. It never
snaps experimental accuracies to a preferred reporting step. `--save-traces`
stores raw local responses and validation outcomes for future audits. New runs
are written to a separate output directory and do not replace these reported
summary files automatically.

The accompanying code implements Flat and random-grouping DCI. Other TTS and
semantic-grouping methods currently have reported data here, not complete
executable reproductions in the CLI.
