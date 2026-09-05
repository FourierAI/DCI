# Changelog

## 0.4.0 - 2026-09-05

- Match the manuscript's `exact-raw-string-v1` validation rule. A response is
  selected only when it exactly equals a label in the current call's candidate
  list; only the literal `None` is treated as a null response.
- Preserve model response text unchanged in traces. Whitespace, wrappers,
  punctuation, quotes, capitalization variants, out-of-group labels, and other
  non-matching outputs are invalid.
- Use the ImageNet-21K first-name protocol: 21,843 WNIDs map to 20,101 distinct
  candidate names after exact duplicate removal in source order.
- Synchronize the current Figure 12 traces and assets: CUB-200-2011 follows
  `200 -> 20 -> 2 -> 1`, and Food-101 follows `101 -> 5 -> 1`, both with `B=10`.
- Add a consolidated experiment-protocol document and explicit provenance
  boundaries for the author-reported aggregate results.
- Record protocol metadata and hashes needed to keep resumed and new
  evaluations separate from incompatible configurations.
- Derive random groups from stable dataset-relative image identifiers, reject
  initial configurations with $B>N$, and record tracked-worktree state in run
  manifests.

## 0.3.0 - 2026-09-03

- Added the paper-aligned Flat/DCI runner, dataset loaders, result snapshot,
  figures, and project page.
