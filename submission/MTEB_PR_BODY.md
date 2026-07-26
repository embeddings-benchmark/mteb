## Summary

Adds a transparent task-type router that uses F2LLM-v2-330M for Retrieval and
C2LLM-7B for Reranking. Both child revisions are pinned, and the router reads
only the coarse MTEB task type.

## Reproduction

- Benchmark: `CoREB(v1)` (all six tasks)
- MTEB: `2.18.6`
- Device: NVIDIA L4
- Precision: bfloat16
- Attention: FlashAttention 2
- Routing: Retrieval -> F2LLM-v2-330M; Reranking -> C2LLM-7B

## Disclosure

This composition was selected using public CoREB results. CoREB exposes only a
test split, so this is a leaderboard-adapted routing policy. It does not route
on task name, examples, labels, or scores.

## Checklist

- [x] I have filled out the ModelMeta object to the extent possible
- [ ] I have ensured that my model can be loaded using
  - [ ] `mteb.get_model(model_name, revision)` and
  - [ ] `mteb.get_model_meta(model_name, revision)`
- [x] I have tested the implementation works on representative Retrieval and
  Reranking task metadata
- [ ] The model card/config repository is public
- [x] There is no original-paper result to reproduce; both child revisions and
  the complete CoREB reproduction environment are documented

## Before opening

- Replace `TODO_AFTER_HF_UPLOAD` with the public Hugging Face commit SHA.
- Fill the final six-task score and link the results PR.
- Confirm maintainers accept task-dependent embedding dimensions for a router.
