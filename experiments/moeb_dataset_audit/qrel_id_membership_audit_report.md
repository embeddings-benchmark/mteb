# Qrel ID membership audit (Issue #5021)

Scope: whether retrieval-task qrels ever reference `query-id`/`corpus-id` values
that don't actually exist in the task's loaded `queries`/`corpus` datasets — an
invariant that is **not** enforced anywhere in MTEB's loading or evaluation
code (see "Why this isn't already caught" below). Audited 45 retrieval-family
tasks drawn from the de-duplicated task inventory across the four curated
omni-modal benchmarks (`MTEB(eng, v2)`, `MIEB(lite)`, `MAEB(beta)`,
`MVEB(beta)`; see [`moeb_task_inventory.json`](moeb_task_inventory.json)),
plus `AudioCapsT2ARetrieval`/`AudioCapsA2TRetrieval`.

Script: [`audit_qrel_id_membership.py`](audit_qrel_id_membership.py) (one-off,
not part of any PR). For each task it re-derives the config-resolution and
split-fallback logic from `RetrievalDatasetLoader`
(`mteb/abstasks/retrieval_dataset_loaders.py`), but pulls **only the `id`
column** from corpus/queries — never a media column — so it needs no
torchcodec/ffmpeg and works identically across text/image/audio/video tasks.
Computes `qrel_query_ids - query_ids` and `qrel_corpus_ids - corpus_ids` per
task. Full raw results: [`qrel_id_membership_audit.json`](qrel_id_membership_audit.json).

## Results

35/45 tasks loaded successfully; 10 failed with a script-level (not data-level)
limitation — see "Coverage gaps" below.

| Task | corpus | qrel rows | missing corpus IDs | result |
|---|---:|---:|---:|---|
| **OVENIT2TRetrieval** | 676,667 | 492,654 | **442,650 rows (89.9%)** | **FOUND** |
| **InfoSeekIT2TRetrieval** | 611,651 | 73,869 | **7,547 rows (10.2%)** | **FOUND** |
| **ArguAna** | 8,674 | 1,406 | **5 IDs** | **FOUND** |
| AskUbuntuDupQuestions, CIRRIT2IRetrieval, CQADupstackGamingRetrieval, CQADupstackUnixRetrieval, CUB200I2IRetrieval, ClimateFEVERHardNegatives, ClothoT2ARetrieval.v2, ClothoA2TRetrieval.v2, CommonVoice/AudioCaps (both directions), FEVERHardNegatives, Fashion200kI2TRetrieval, FiQA2018, GigaSpeechT2ARetrieval, HotpotQAHardNegatives, MACST2ARetrieval, MindSmallReranking, NIGHTSI2IRetrieval, RP2kI2IRetrieval, SCIDOCS, SpokenSQuADT2ARetrieval, TRECCOVID, Touche2020Retrieval.v3, UrbanSound8KT2ARetrieval, VQA2IT2TRetrieval, VisualNewsI2TRetrieval, WebQAT2ITRetrieval | — | — | 0 | clean |

**Coverage gaps (script limitation, not evidence of correctness):** 10 tasks
use HF repo config-naming schemes the script doesn't parse yet —
per-language configs (`CommonVoiceMini21T2ARetrieval`, `FleursT2ARetrieval`:
raw language codes as config names), `merged`/`pure` prefixes
(`JamAltArtistA2ARetrieval`, `JamAltLyricA2TRetrieval`), and single-`"default"`-config
repos (`AVMemeExamAT2VRetrieval`, `ActivityNetCaptionsT2VRetrieval`,
`AudioCapsAVVA2TRetrieval`, `AudioCapsAVVT2ARetrieval`, `HatefulMemesI2TRetrieval`,
`MSVDT2VRetrieval`, `VALOR32KT2VARetrieval`, `VATEXV2ARetrieval`,
`VATEXVA2TRetrieval`, `VGGSoundAVA2VRetrieval`, `YouCook2T2VARetrieval`).
These are **unaudited**, not confirmed clean.

## Finding 1: OVEN / InfoSeek — systematic cross-source ID collision (M-BEIR family)

Both are `task6` exports from the M-BEIR merged benchmark
(`mteb/mbeir_oven_task6` @ `8899473562`, `mteb/mbeir_infoseek_task6` @ `4510aa3b45`).
Verified directly against the HF repos (not just via the audit script):

```python
# OVEN
corpus id prefix counts: Counter({'5': 676667})
qrel corpus-id prefix counts: Counter({'6': 442650, '5': 50004})

# InfoSeek
corpus id prefix counts: Counter({'6': 611651})
qrel corpus-id prefix counts: Counter({'6': 66322, '5': 7547})
```

Each task's `corpus` config contains only **one** M-BEIR source prefix, but
the qrels (inherited wholesale from the original merged M-BEIR relevance
judgments) reference **both**. Concrete example (OVEN):

```
qrel row: {query-id: '5:35085', corpus-id: '6:1000', score: 1}
'6:1000' in corpus['id']  ->  False
```

**Per-query reachability** (does the query have *any* qrel-labeled positive
that actually exists in corpus?):

| | total queries | 0 reachable positives (metrics forced to 0 regardless of model) | some reachable + some dangling | fully clean |
|---|---:|---:|---:|---:|
| OVEN | 50,004 | 0 | 36,549 (73.1%) | 13,455 |
| InfoSeek | 11,323 | 0 | 7,547 (66.7%) | 3,776 |

No query has zero reachable positives, so MRR/Hit-Rate are largely unaffected
(see evaluation-path analysis below) — but Recall@K/NDCG@K/MAP are
structurally capped below their true achievable maximum for the majority of
queries in both tasks.

## Finding 2: ArguAna — small, likely a stale corpus-doc deletion

`mteb/arguana` @ `c22ab2a510`, one of the most widely used BEIR datasets.
5/8674 corpus docs referenced by qrels are missing. Concrete example:

```
query "test-education-ufsdfkhbwu-con03a" -> qrel corpus-id "test-education-ufsdfkhbwu-con03b"
corpus actually contains for this topic: con01b, con02b, pro01b, pro02b, pro03b
"con03b" is absent
```

Looks like a single document was dropped during corpus construction/dedup
without removing or repointing the corresponding qrel. Small in scale (5
IDs) but notable because ArguAna has been a standard benchmark for years.

## Why this isn't already caught

Traced the full evaluation path (`mteb/abstasks/retrieval.py::_evaluate_subset`,
`mteb/_evaluators/retrieval_evaluator.py`, `mteb/models/search_wrappers.py`,
`mteb/_evaluators/retrieval_metrics.py`):

- `_filter_queries_without_positives` (`retrieval.py:67-80`) only drops
  queries with an *empty* qrels dict — never checks whether qrel doc IDs
  exist in corpus.
- The search index (`SearchEncoderWrapper.index()`) is built strictly from
  `corpus["id"]`; qrels never participate in indexing or search. A qrel
  corpus-id absent from `corpus["id"]` is unretrievable by any model, by
  construction.
- `RetrievalDatasetLoader._load_qrels` (`retrieval_dataset_loaders.py:187-223`)
  loads qrels independently with no join/validation against corpus IDs.
- Scoring uses `pytrec_eval.RelevanceEvaluator` (`retrieval_metrics.py:588`),
  standard TREC semantics: a qrel doc absent from the run is silently
  counted as "not retrieved" (contributes to the denominator, never the
  numerator). No exception, no warning.

## Metric impact (toy example, k=2, one query)

Qrels `{docA: 1, docB: 1}`, `docB` unreachable. Best possible model ranks
`docA` at rank 1 (retrieval = `[docA, docC]`).

| metric | A: qrels has dangling `docB` | B: dangling entry removed | affected? |
|---|---:|---:|---|
| Recall@2 | 0.50 | 1.00 | yes — denominator inflated |
| MAP | 0.50 | 1.00 | yes — R inflated |
| NDCG@2 | 0.613 | 1.00 | yes — IDCG assumes docB reachable |
| MRR@2 | 1.00 | 1.00 | no — only cares about first reachable hit |
| Precision@2 | 0.50 | 0.50 | no — denominator is k, not qrel count |

A model already at the ceiling of what's achievable given the actual corpus
is scored as if it failed to retrieve something that was never retrievable.

## Conclusion / next steps

This has moved from a theoretical invariant gap to a confirmed, real defect
with quantified severity. Recommended next steps (not yet started):
1. Expand the audit to the remaining M-BEIR-family tasks not yet covered
   (other `mteb/mbeir_*_task*` exports) to check whether the prefix-mismatch
   pattern generalizes across the whole family.
2. Adapt the audit script's config-resolution to cover the 10 skipped tasks'
   non-standard schemas (per-language configs, `merged`/`pure` prefixes,
   single-`default`-config repos) for full coverage.
3. Decide on a fix for OVEN/InfoSeek specifically: either regenerate their
   `corpus` config to include both M-BEIR source prefixes, or drop
   unreachable qrel rows at the source and document the resulting metric
   discontinuity for any already-published scores.
4. Consider whether a permanent CI check (mirroring this script's logic,
   without media decoding) belongs in `tests/test_tasks/test_task_quality.py`
   alongside the checks added in PR #5090 — deferred until the above
   scoping/fix decisions are made.
