# Current manuscript experiment protocol

This document consolidates the method and experimental settings stated in the
current manuscript, synchronized on 2026-09-05. It also distinguishes those
settings from implementation defaults and from details that cannot be recovered
from the retained historical records.

## DCI and Flat inference

Let $N$ be the current candidate-set size and $B\geq2$ the maximum local group
size. Flat inference sends the complete candidate set in one prompt. DCI uses
the following procedure:

1. If the set is empty, return null without a model call. If it is a singleton,
   return that label without a model call.
2. If $2\leq N\leq B$, make one final local call and apply the same validity rule
   used at all earlier levels.
3. If $N>B$, randomly partition the active candidates into disjoint groups of
   at most $B$ labels. All but at most one group contain $B$ labels.
4. Query every group independently with the same image, frozen MLLM, prompt
   template, and decoding configuration. Same-level queries may run in parallel.
5. Retain at most one valid response from each group, combine the survivors, and
   recurse.

The Flat and DCI templates share the classification instruction, output-format
constraint, and one-shot example. DCI additionally asks the model to choose the
most likely listed category whenever possible and permits the literal response
`None`. DCI retains this instruction for its final local call, including the
single-call case $N=B$.

### Exact raw-output validity

The protocol is `exact-raw-string-v1`:

- Keep the model response untouched.
- Record `selected` only when the raw response is exactly equal to one candidate
  string supplied in that call.
- Record `none` only for the literal raw response `None`.
- Record every other response as `invalid`.

There is no whitespace trimming, answer-prefix removal, regular-expression
extraction, punctuation removal, quote or code-fence stripping, case conversion,
spelling correction, comma splitting, synonym lookup, substring matching, or
semantic remapping. For example, `Greek_salad` is invalid when the supplied
candidate is `greek_salad`; `greek_salad.` is also invalid. Flat uses the full
candidate list for this check, while DCI uses the current local group. A null or
invalid final result counts as incorrect and remains in the denominator. A
response may be a legitimate member of the original global vocabulary and still
be invalid for DCI when it is absent from the current call's local group; that
response is deleted rather than promoted to the next level.

## Common evaluation protocol

The manuscript reports classification accuracy separately for five runs and
then reports the mean and standard deviation of the five run-level values. The
run is the unit of variability. In paired Flat/DCI comparisons, both methods use
the same model, candidate vocabulary, evaluation images, and model-specific
decoding configuration within a run.

Unless an experiment below states otherwise, DCI uses random grouping and
$B=10$.

| Dataset | Evaluation images | Full candidate vocabulary | Default $B$ |
|:--|:--|--:|--:|
| CIFAR-100 | complete official test split, 10,000 images | 100 | 10 |
| CUB-200-2011 | complete official test split, 5,794 images | 200 | 10 |
| Food-101 | complete official test split, 25,250 images | 101 | 10 |
| ImageNet-1K | complete official validation split, 50,000 images, 50/class | 1,000 | 10 |
| ImageNet-21K stress test | 1,000 distinct images sampled/run from the complete available pool | 20,101 | 100 |

### ImageNet-1K candidate-set scaling

Both the candidate-space evaluation and the DCI scaling comparison use
$N\in\{10,20,100,200,500,1000\}$.

- For every run and every $N<1000$, independently sample $N$ classes and evaluate
  all 50 official validation images for each selected class, giving $50N$ images.
- Class subsets are not nested across different values of $N$.
- At $N=1000$, use all classes and all 50,000 validation images. The class set and
  image set are fixed; grouping and model generation are repeated across runs.
- Paired Flat/DCI measurements use the same sampled classes and images within a
  run. DCI uses $B=10$.

### ImageNet-21K first-name stress test

The supplied catalog contains 21,843 WNID entries. For each WNID, take the first
comma-separated name, remove only the catalog field's surrounding whitespace,
preserve its case, spaces, and underscores, and remove exact duplicate candidate
strings in source order. This yields 20,101 candidates. WNIDs sharing a first
name map to the same target name. The complete catalog retains every WNID-to-name
mapping, and each indexed image obtains its target through its WNID. Prediction
scoring compares the exact validated name with that mapped target.

For every run, sample 1,000 distinct images uniformly without replacement from
the complete available WNID-indexed pool. Samples may overlap across runs. Flat
and DCI use the same 1,000 images and all 20,101 candidate names within a run.
The reported accuracy is micro-averaged over sampled images. The default DCI
setting is $B=100$. See
[`IMAGENET21K_LABEL_PROTOCOL.md`](IMAGENET21K_LABEL_PROTOCOL.md) for catalog
hashes, collision counts, and the target mapping.

## Experiment matrix and model names

The names below match the manuscript's checkpoint names and API aliases;
they do not identify an unrecovered model revision.

| Manuscript experiment | Models / methods | Dataset and $N/B$ |
|:--|:--|:--|
| Candidate-space evaluation (Fig. 1) | Llama-3.2-11B-Vision-Instruct; Gemma-3-4B-IT; Gemma-4-E4B; DeepSeek-VL-7B-chat; Kimi-VL-A3B-Instruct; Qwen2.5-VL-7B-Instruct; Qwen2.5-VL-32B-Instruct; Qwen2.5-VL-MAX; Qwen3-VL-2B-Instruct; Qwen3-VL-8B-Instruct | ImageNet-1K; $N\in\{10,20,100,200,500,1000\}$ |
| Main Flat/DCI comparison (Fig. 5) | Qwen2.5-VL-7B-Instruct; Qwen3-VL-{2B,4B,8B}-Instruct; Kimi-VL-A3B-Instruct; Gemma-4-E2B | ImageNet-1K, CIFAR-100, CUB-200-2011, Food-101; full vocabularies; $B=10$ |
| DCI candidate scaling (Fig. 6) | Qwen2.5-VL-7B-Instruct; Qwen3-VL-{2B,8B}-Instruct; DeepSeek-VL-7B-chat; Kimi-VL-A3B-Instruct; Gemma-4-E4B | ImageNet-1K; six $N$ values above; $B=10$ |
| ImageNet-21K comparison (Fig. 7) | Local: Qwen2.5-VL-7B-Instruct, Qwen3-VL-{2B,4B,8B}-Instruct, Gemma-4-E4B. API: GPT-4.1, Claude Opus 4.5, Qwen3-VL-Plus, Kimi-K2.5, Llama-4-Maverick | $N=20{,}101$; $B=100$; 1,000 images/run |
| ImageNet-21K group-size sweep (Fig. 8) | Qwen3-VL-2B-Instruct; Gemma-4-E4B | $B\in\{50,100,500,1000,5000\}$ plus Flat |
| Standard-space group-size sweep (Fig. 9) | Qwen3-VL-8B-Instruct | CIFAR-100 and ImageNet-1K; $B\in\{2,5,10,20,50\}$ plus Flat |
| Test-time scaling comparison (Fig. 10) | Flat, CoT, Plan-and-Solve, Describe-and-Answer, Self-Consistency, Self-Aggregation, DCI; Qwen3-VL-8B-Instruct | CUB-200-2011 and ImageNet-1K; DCI $B=10$; SC uses five parallel responses; SA uses four parallel responses and one aggregation call |
| Grouping ablation (Fig. 11) | Qwen2.5-VL-7B-Instruct; Qwen3-VL-4B-Instruct | ImageNet-1K; $B=10$; random, most-similar, least-similar grouping; semantic similarities use CLIP text embeddings |
| Author-confirmed recorded qualitative traces (Fig. 12) | Qwen3-VL-2B-Instruct | $B=10$; CUB $200\to20\to2\to1$; Food-101 $101\to5\to1$ |

The current CLI implements Flat inference and random-grouping DCI. It does not
yet implement the complete CoT, Plan-and-Solve, Describe-and-Answer,
Self-Consistency, Self-Aggregation, or semantic-grouping evaluation pipelines.
Their aggregate tables are included only as reported manuscript data.

## Inference environment and measurement definitions

The manuscript states that local models were evaluated in BF16 on one server
with an AMD EPYC 7642 CPU, four NVIDIA RTX 4090 GPUs, and 256 GB DDR4 RAM. The
reported software environment is Ubuntu 22.04.3 LTS, Python 3.10, PyTorch 2.8.0,
Transformers 4.57.3, and vLLM 0.15.0. Depending on model compatibility, local
inference used either vLLM or Hugging Face Transformers; API models used their
official endpoints.

The manuscript says that each checkpoint or endpoint retained its default
decoding configuration and that no generation parameter was tuned specifically
for DCI. A paired Flat/DCI comparison used the same configuration for a given
model. It reports no prompt truncation; the longest ImageNet-21K Flat prompt was
approximately 90K input tokens, and every model used there supported at least
120K tokens.

- **Accuracy:** correct predictions divided by all evaluated images. Null and
  invalid results count as incorrect.
- **ImageNet-21K accuracy:** micro-average over the 1,000 sampled images in each
  run.
- **Latency:** end-to-end wall-clock seconds per image, including image
  preprocessing and every recursive call needed for the final result.
- **Parallel latency:** same-level DCI groups run concurrently. The manuscript
  says concurrency was set to saturate the four-GPU server. SC used five
  concurrent calls; SA used four concurrent calls and then one aggregation call.
- **API cost:** provider-billed USD normalized per image.
- **Failures:** transient execution, timeout, and API failures were retried until
  a response was obtained; the manuscript reports no unrecoverable execution
  failure. Invalid successfully returned content was not retried.

## Known and unknown configuration details

The following distinction prevents repository convenience defaults from being
mistaken for recovered historical settings.

| Detail | Current status |
|:--|:--|
| Flat/DCI prompts | Exact current templates are versioned in `dci/prompts.py` |
| Validation | Exact and versioned as `exact-raw-string-v1` |
| New-run grouping | Current CLI uses a deterministic per-image/run shuffle derived from the run seed and dataset-relative image identifier; seed defaults to 0 and run seeds advance by run |
| Historical grouping seeds and partitions | Unknown; original run manifests are unavailable |
| Historical sampled ImageNet-1K classes and ImageNet-21K images | Unknown; original run manifests are unavailable |
| Historical checkpoint/API revisions | Unknown; only manuscript checkpoint names and aliases are available |
| Historical model-to-backend mapping | Unknown for individual models; the manuscript states vLLM or Transformers according to compatibility |
| Historical temperature, top-p, top-k, maximum generation tokens, do-sample, and generation seed | Endpoint/checkpoint defaults were retained according to the manuscript, but their concrete per-model values are not recoverable from the available records |
| Current CLI decoding fields | Optional `temperature`, `top_p`, and `max_tokens`; null means the request leaves that setting to the serving endpoint |
| Historical DCI worker count / scheduling | Exact values are unknown; only saturation of the four-GPU server is stated. The CLI default worker count is a convenience default, not a recovered setting |
| Historical SD degrees of freedom | Unknown; the manuscript reports run-level SD but the original five values and convention are unavailable |
| ImageNet-21K upstream release name and available-pool size/filtering | Not specified in the manuscript or recoverable manifests; the shipped 21,843-entry catalog and its hashes define the public label artifact |

New-run manifests record the selected labels and images, run seed, prompt and
catalog hashes, validation version, available decoding arguments, environment,
command arguments, Git revision, and tracked-worktree state. Users should additionally record the exact
checkpoint or API revision, preprocessing, context limit, unexposed decoding
defaults, batching, GPU allocation, cache configuration, and service date.

## Reported-results provenance boundary

The files under [`../data/results/`](../data/results/README.md) preserve the
current manuscript's aggregate reporting workbooks and typed figure-table
transcriptions. Original five-run prediction and correct/total logs are
unavailable. The stored means include reporting corrections documented in the
workbooks, and the SD values were retained rather than recomputed.

No quantitative benchmark evaluation or historical aggregate rescoring was
performed when the executable protocol was synchronized to the 20,101-name
ImageNet-21K construction and exact raw-string validation. The retained
aggregates therefore document
author-reported manuscript values; they are not claimed to have been regenerated
by this version of the CLI. New runs produce separate measurements and must not
overwrite or be silently combined with this snapshot.
