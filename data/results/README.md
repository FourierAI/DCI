# Final experimental results

This directory contains the final aggregate numerical results reported in
**Divide-and-Conquer Inference for Large-Scale Image Classification with
Multimodal Large Language Models**.

[`reported_results.json`](reported_results.json) provides eight tables:

| Table | Content | Figure |
|:--|:--|:--|
| Main DCI | Main model-dataset accuracy comparisons | 5 |
| Scaling Wide | Accuracy across candidate-set sizes | 1 |
| Suppression | Flat and DCI candidate-set scaling | 6 |
| IN21K Main | Full-vocabulary ImageNet-21K comparison | 7 |
| IN21K Group Size | Large-vocabulary accuracy/efficiency trade-offs | 8 |
| Group Size | Standard-vocabulary group-size comparison | 9 |
| TTS | Test-time scaling comparisons | 10 |
| Grouping | Grouping-strategy comparison | 11 |

Accuracy means are percentages. Accuracy standard deviations and absolute gains
are in percentage points; latency is seconds per image and cost is USD per
image. Column names identify the units, model, dataset, and setting. `Avg.`
rows are dataset macro-averages; `null` denotes a field not reported or not
applicable, not zero.

Only final aggregate results are distributed here. Working spreadsheets,
individual-run outputs, intermediate traces, and experiment-process documents
are outside the public release.

The final figures are in [`assets/figures/`](../../assets/figures/), with a
visual overview on the [project page](https://fourierai.github.io/DCI/).
