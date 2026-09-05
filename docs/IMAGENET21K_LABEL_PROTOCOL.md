# ImageNet-21K first-name protocol

Protocol: `imagenet21k-first-name-v1`. Synchronized with the current manuscript
on 2026-09-05.

## Candidate construction

The source is `data/metadata/imagenet21k/im21K.txt`, relative to either code
repository root. Each line contains a WNID followed by comma-separated names.
Select only the first name following the WNID, strip surrounding whitespace,
and preserve its original spelling, capitalization, internal spaces, and
underscores. Remove exact duplicate candidate strings in source order.

| Source entry | Candidate name |
| --- | --- |
| `n00004475,organism, being` | `organism` |
| `n00007846,person, individual, someone, somebody, mortal, soul` | `person` |
| `n00326094,rock_climbing` | `rock_climbing` |

This is first-name selection from the supplied metadata. It does not select
the shortest synonym, choose a name using a model, concatenate synonyms, or
append a WNID to the candidate name. The complete source file remains intact.

| Quantity | Count |
| --- | ---: |
| Source entries / distinct WNIDs | 21,843 |
| Distinct first-name candidates after exact deduplication | 20,101 |
| First-name groups shared by more than one WNID | 1,444 |
| WNIDs in those shared-name groups | 3,186 |

For example, `n02012849` and `n03126707` both map to `crane`. Both WNID mappings
remain in the catalog, while the candidate list contains one `crane` entry.
The candidate count is **N = 20,101**; **ImageNet-21K** remains the dataset name.
In the current manuscript, “full vocabulary” in this stress test means all
20,101 distinct first-name candidates.

## Targets and scoring

The current runner in `github-dci/` maps each image's WNID to its first name.
WNIDs sharing that name share one target label. We report classification
accuracy over the sampled images, comparing the validated predicted name with
this mapped target. The metric does not separately resolve same-name WNIDs.

Candidate deduplication and response validation both use exact, case-sensitive
string equality, but at different stages. Deduplication removes a candidate only
when its first-name string exactly duplicates an earlier string. Validation uses
`exact-raw-string-v1`: an untouched model response is accepted only when it is
exactly equal to one candidate supplied in that call. The literal `None` is a
null decision. No whitespace trimming, answer-wrapper removal, punctuation or
quote stripping, case conversion, spelling correction, synonym lookup,
substring extraction, or ancestor-category remapping is performed.

The vocabulary contains 31 pairs that collide only after case folding (62
names, associated with 68 WNIDs), such as `Cardigan` and `cardigan`. Exact
deduplication retains both strings as distinct candidates, and exact validation
requires the response to preserve their case. This statistic is documented to
make clear that case-insensitive validation would define a different protocol.

## Current and legacy runners

The public runner uses a WNID-indexed available ImageNet-21K image pool and the
current manuscript's image-sampling protocol. An earlier local expanded-vocabulary
script used ImageNet-1K validation-image metadata instead. That script does not
implement the current full-pool protocol and must not be treated as a reproduction
of it.

The legacy metadata has 50 images with target `teddy`, which is absent from the
first-name candidates. The source file contains `teddy` only as a synonym of
`chemise`; that is not a valid semantic replacement for the toy-bear target.
That legacy loader warns about the missing target and preserves every image and
original target. This coverage issue requires reliable class/WNID metadata before
any such legacy run can be treated as a closed-set result.

## Reproducible artifacts and scope

Each repository includes these files under `data/metadata/imagenet21k/`:

- `im21K.txt`: unmodified source entries and synonyms.
- `first_names.txt`: the ordered, deduplicated 20,101-name candidate list.
- `first_name_protocol.json`: construction rule, counts, and SHA-256 hashes.

The source SHA-256 is
`66e637be9c3dc9c6a3850ec04ceae807d7225aece331581a72b5ca681016829d`.
The exported candidate-list SHA-256 is
`3577ce3def0f09abc5718569084487b8fe8543bbf0c2475f717209194d4b4c13`.
Repository tests check the loader against this export and manifest. These hashes
verify label construction only; they do not verify any reported accuracy,
latency, or cost aggregate.

This synchronization updates the public executable protocol and its description.
Existing result workbooks, numerical results, figures, and author-confirmed
recorded qualitative traces are preserved. No quantitative benchmark experiment,
historical aggregate rescoring, or token measurement was repeated. Original
five-run prediction and correct/total logs
are unavailable, so the historical aggregates cannot be independently rebuilt
under this 20,101-name protocol. They remain author-reported manuscript values,
not claimed outputs of a new repository reproduction. New evaluations create
separate manifests, predictions, and summaries.

## 中文说明

每个 WNID 只取其后第一个逗号分隔名称，去掉两端空白，保留大小写、下划线及名称
内部空格，再按原顺序对完全相同的名称去重。21,843 个 WNID 最终对应 20,101 个
不同候选名称。图像的真实标签由 WNID 映射为第一名称；第一名称相同的 WNID 共用
一个目标标签。论文报告分类准确率，将校验后的预测名称与上述目标标签比较。
模型原始输出仅在与当前候选字符串完全相等时有效；不清理空白、前后缀、标点或
引号，不转换大小写，也不做拼写、同义词或子串映射。本次同步没有重新运行模型或
重新计算历史实验数据，且原始五轮预测与 correct/total 日志不可用。因此，现有
汇总值是论文报告值，不应表述为由当前 20,101 名称协议重新复现的结果。旧版脚本
的图像池和 `teddy` 覆盖问题见上文，不能与当前 WNID 图像池混用。
