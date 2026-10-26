# Qwen and NuExtract last-prompt text embeddings

These registrations use publicly available generative checkpoints as text
encoders through [vLLM-Last-Hidden-State](https://github.com/numindai/vLLM-Last-Hidden-State).
The names identify embedding recipes, not additional Hub repositories.

| MTEB name | Base checkpoint | Revision |
| --- | --- | --- |
| `Qwen/Qwen3.5-4B-e5v-last-prompt` | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| `numind/NuExtract3-revised-last-prompt` | `numind/NuExtract3` | `c99dc8f5641b866aa0192b6ea78f84bf9f3535f1` |

Each request returns the final active prompt token's post-final-normalization
hidden state, used to predict the first generated token. The 2560-dimensional
vector is cast to float32 without pooling or client L2 normalization. Similarity
uses cosine. Both queries and documents use the same recipe. Only text inputs
are registered here.

Qwen receives the exact user string `{text}\n Summary above sentence in one word: `,
including the space after the newline and the trailing space. This is inspired
by [E5-V](https://arxiv.org/abs/2407.12580), applied to Qwen3.5-4B; it is not a
reproduction of the original paper's trained model.
NuExtract receives raw text with `mode="embeddings"` and `enable_thinking=False`
as chat-template kwargs. Its template asks for a stable document identity and
ends with an assistant short-description JSON prefill. Exact whitespace and
special tokens are preserved in the bundled templates.

## Setup and use

Install the `last-prompt` client extra. In a separate serving environment,
install the extension with its supported vLLM release using its
[setup instructions](https://github.com/numindai/vLLM-Last-Hidden-State#quick-start).
Pin the extension commit and vLLM version for new evaluations; the current
extension release is not evidence of the version used for the original results.

The exact templates and a known prompt/token example are included in
`mteb/models/model_implementations/last_prompt_resources/` and packaged in the wheel.
For example, from an MTEB checkout, after configuring the serving environment:

```sh
vllm-last-hidden-state Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --tokenizer-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --chat-template mteb/models/model_implementations/last_prompt_resources/qwen.jinja \
  --host 127.0.0.1 --port 8000
```

Or, for NuExtract:

```sh
vllm-last-hidden-state numind/NuExtract3 \
  --revision c99dc8f5641b866aa0192b6ea78f84bf9f3535f1 \
  --tokenizer-revision c99dc8f5641b866aa0192b6ea78f84bf9f3535f1 \
  --chat-template mteb/models/model_implementations/last_prompt_resources/nuextract-revised.jinja \
  --host 127.0.0.1 --port 8000
```

These illustrate loading pinned weights/templates, not recovered historical
launch commands. Choose compatible hardware, precision and context settings,
and record them with new results. The adapter does not truncate input.

```python
import mteb

model = mteb.get_model("Qwen/Qwen3.5-4B-e5v-last-prompt")
try:
    report = mteb.mock_run(model)
    assert report.all_passed
finally:
    model.close()
```

Set `LAST_PROMPT_BASE_URL` (default `http://localhost:8000/v1`),
`LAST_PROMPT_SERVED_MODEL` for a serving alias, and optionally
`LAST_PROMPT_API_KEY`. The adapter verifies the model root and exact token IDs
via `/tokenize` before sending embedding requests. This detects template drift;
it cannot attest to the loaded weight bytes. Requests use one completion token,
`n=1`, temperature zero, no streaming, `add_generation_prompt=True`, and
`kv_transfer_params.return_last_hidden_state=True`.

## Evaluation provenance

Both recipes completed all 41 `MTEB(eng, v2)` tasks with MTEB 2.24.14.
The original task JSONs retain that version. Task names, dataset revisions,
selected splits and subsets match MTEB 2.26.24 metadata; this does not establish
identical evaluator code across versions. Descriptive unweighted task means
multiplied by 100 are 47.09853 (Qwen) and 42.30302 (NuExtract).

Original manifests did not record an immutable checkpoint revision. The
operator subsequently confirmed the latest weights, resolved on 2026-10-26
to the above Hub SHAs; both Hub modification dates predate evaluation.
Pinned tokenizers reproduce the saved prompt token IDs exactly; the supplied
revised NuExtract template reproduces its saved rendered prompt. This is
post-run provenance, not an independent inspection of historical loaded weights.

Historical server launch commands, precision, GPU/backend/runner settings and
vLLM/extension revisions were not recorded. Training overlap is unspecified
(`training_datasets=None`). These limits remain relevant when reproducing or
comparing the scores.

The original clients used `mteb.evaluate(..., cache=None)` and wrote task JSONs
themselves. Accompanying `run_settings.jsonl` files were reconstructed from
frozen manifests and client source on 2026-10-26. Each entry labels this origin,
includes source hashes and records only supported settings; they are not
original runtime logs. The task result JSONs are byte-identical to the originals.

On 2026-10-26 both adapters passed all 28 compatible live text mock tasks;
see the committed `mteb_mock_run_results.md`. The live server reported vLLM
0.30.0, which does not establish the version used for the original evaluations.
