# DCI project page

This page mirrors the current manuscript title, terminology, experimental
protocol, scoped claims, and all 12 manuscript figures as synchronized on
2026-09-05. Figure 12 combines the two qualitative $B=10$ trajectories. Web PNGs in
`docs/assets/` are synchronized from the vector sources in
`../assets/figures/`.

Reported summary results and their provenance are available in
[`../data/results/`](../data/results/README.md). See
[`PAPER_ALIGNMENT.md`](PAPER_ALIGNMENT.md) for the synchronized release details.
The complete method and experiment matrix is in
[`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md). Response validation uses
untouched, case-sensitive raw-string membership with no normalization.
ImageNet-21K uses 20,101 distinct first-name candidates derived from all 21,843
WNIDs; the label construction and classification accuracy calculation are specified in
[`IMAGENET21K_LABEL_PROTOCOL.md`](IMAGENET21K_LABEL_PROTOCOL.md).

The reported-results directory preserves the manuscript's aggregate workbooks
and machine-readable transcriptions. The first-name and validation protocol sync
did not rerun models or rescore those aggregates, and original five-run
prediction/correct-total logs are unavailable. Those files are therefore
author-reported values rather than a regenerated CLI reproduction.

This project page is adapted from the
[Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template),
which incorporates components from the
[Nerfies project page](https://nerfies.github.io/).

The website source in this directory is distributed under the
[Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).
The DCI research code in the parent repository remains licensed under the MIT
License.
