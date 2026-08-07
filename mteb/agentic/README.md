# Answer-mode retrieval benchmark (`mteb.agentic`)

MTEB's retrieval tasks score a retriever's *ranking* quality (ndcg, recall).
This package scores the end-to-end *system*: given a question and a fixed
corpus, produce an answer, graded on correctness, cost, latency, gold-doc
recall, and calibration. The same dataset supports both modes, so retrievers
and full systems stay comparable.

## Quickstart

```bash
pip install "mteb[agentic]"        # in-process paradigms; add [agentic-agents] for containerized agents
```

```python
from mteb.agentic import evaluate, OpenAIChatModel

model = OpenAIChatModel("Qwen/Qwen3-32B", base_url="http://localhost:8010/v1", api_key="EMPTY")
result = evaluate("rag", "BrowseCompPlus", model=model, retriever="bm25")
print(result.scores.accuracy, result.scores.mean_latency_s)
```

Each task carries its official metric as the default judge (BrowseComp-Plus:
LLM grading; HotpotQA/MuSiQue: token F1; LongBench-v2: option accuracy;
OOLONG: numeric tolerance). Pass `judge=` to override.

`evaluate(system, task, *, model=, retriever=, judge=, ...)` is the single
front door. `system` and `task` are registry names (`list_systems()`,
`list_tasks()`) or objects. `model` is a `ChatModel` or a model name (a name
uses `OPENAI_BASE_URL` / `OPENAI_API_KEY` from the environment). Every call
returns an `AnswerEvaluationResult`: aggregate scores plus per-question records.

To compare several systems, use the batch form. It loads the task once and
reuses compatible corpus representations, including the built retrieval index:

```python
results = evaluate(
    task="BrowseCompPlus",
    systems=["rag", "iterative-rag", "search-agent"],
    model=model,
    retriever="bm25",
    limit=25,
)
print(results["rag"].scores.accuracy)  # dict keyed by system name
```

## Setup

Systems fall in two tiers with different requirements, installed via extras:

- **In-process systems** (`closed-book`, `full-context`, `windowed-full-context`,
  `rag`, `iterative-rag`, `search-agent`,
  `oracle`): `pip install "mteb[agentic]"` (pulls the OpenAI client and bm25s).
  Need only Python and an OpenAI-compatible endpoint for the answerer and judge.
  **No Docker, no Harbor.** For late-interaction retrievers add `mteb[pylate]`.
- **Containerized agents** (`claude-code`, `codex`, `mini-swe-agent`,
  `openhands`, `hermes`): `pip install "mteb[agentic-agents]"` (adds Harbor).
  You never invoke Harbor — `mteb.agentic` generates the task, drives the run,
  and reads answers back; it is a pip dependency, not part of your workflow.
  A container backend is required: local Docker (start Docker Desktop / the
  daemon), or a Harbor cloud sandbox (e2b / daytona) to skip local Docker.
- **`rlm`**: `pip install "mteb[agentic-rlm]"` (PyPI package `rlms`, imported as
  `rlm`). Defaults to rlm's **in-process `local`** execution (no Docker) — the
  model-written code runs in your process. For isolation, pass
  `environment="docker"` (needs a `rlm-sandbox` image:
  `docker build -t rlm-sandbox -f Dockerfile.sandbox .`) or a cloud backend
  (e2b, daytona, modal).

`evaluate()` raises a clear error if the `harbor` CLI is missing; a stopped
Docker daemon surfaces as a failed `harbor run`. In-process runs never require
any of this.

## Auth

**Answerer / judge** (in-process systems): build the `ChatModel` with its
endpoint and key, e.g. `OpenAIChatModel(name, base_url=..., api_key=...)`, or
pass a model name and set `OPENAI_BASE_URL` / `OPENAI_API_KEY` in the
environment.

**Containerized agents**: set your provider's standard variable and `evaluate()`
forwards it into the container automatically (no `agent_env` wiring):

| Provider | Env var |
| --- | --- |
| Claude subscription | `CLAUDE_CODE_OAUTH_TOKEN` (from `claude setup-token`) |
| Anthropic API | `ANTHROPIC_API_KEY` |
| OpenAI / Codex | `OPENAI_API_KEY` (or a CLI sign-in) |
| Local / self-hosted model | `OPENAI_API_BASE`, e.g. `http://host.docker.internal:8011/v1` |

`ANTHROPIC_BASE_URL`, `OPENAI_BASE_URL`, `GEMINI_API_KEY`, and `GOOGLE_API_KEY`
are forwarded too. Containers reach a model running on your host via
`host.docker.internal`, so a server bound to `localhost` must be made reachable
from containers (bind `0.0.0.0` or front it with a proxy). Per-token API auth
gives exact cost accounting; subscriptions suit exploration.

## Paradigms

Systems differ on two axes: how much agency they have, and how they access the
corpus. All run through the same evaluator and judge.

| System | Corpus access | Description |
| --- | --- | --- |
| `closed-book` | none | Floor: parametric memory only. |
| `full-context` | whole corpus in prompt | Long-context rival to RAG. N/A when the corpus exceeds the window (reported as coverage, not as wrong). |
| `windowed-full-context` | sliding windows | Per-window answers aggregated; always applies (bounded by `max_windows`). |
| `rag` | retriever | Retrieve top-k once, answer. |
| `iterative-rag` | retriever | Decompose-retrieve-reason loop (Self-Ask/IRCoT), then answer. |
| `search-agent` | retriever tools | Reason-act loop over search + get_document (BrowseComp-Plus setup). |
| `rlm` | REPL over raw corpus | Recursive Language Model: writes code against the corpus, no retriever. |
| `claude-code`, `codex`, `mini-swe-agent`, `openhands`, `hermes` | raw files | Containerized agents run by Harbor that grep the mounted corpus directly, no retriever. |
| `oracle` | gold documents | Ceiling: answers from the labeled evidence. |

Every paradigm runs through the same `evaluate(system, task, ...)` call:

```python
evaluate("closed-book", task, model=m, judge=j)                          # no corpus
evaluate("full-context", task, model=m, judge=j)                         # or windowed-full-context
evaluate("rag", task, model=m, judge=j, retriever="bm25")                # or iterative-rag, search-agent
evaluate("oracle", task, model=m, judge=j)                               # ceiling
evaluate("rlm", task, model=m, judge=j)                                  # needs mteb[agentic-rlm]
evaluate("claude-code", task, model="claude-sonnet-5", judge=j)          # agent: mteb[agentic-agents] + Docker
evaluate("mini-swe-agent", task, model=m, judge=j, agent_retriever=True)  # agent + BM25 tool
```

The retriever axis is orthogonal and reuses MTEB models: `retriever="bm25"`
(`mteb/baseline-bb25`), any dense encoder name (wrapped in `SearchEncoderWrapper`),
or a late-interaction model such as `colbert-ir/colbertv2.0`. Only retriever-based
systems read it. The corpus is indexed once and, in the batch form, reused across
all retriever systems.

Retrieval *strategies* that transform the query or reorder candidates are
retrievers, not answer paradigms, so `rag` composes with any of them and they
stay scorable on ranking metrics (e.g. MTEB/OBLIQ retrieval tasks):

```python
from mteb.agentic import QueryRewriteRetriever, HyDERetriever, RerankRetriever
import mteb
bm25 = mteb.get_model("mteb/baseline-bb25")
evaluate("rag", data, model=model, judge=judge, retriever=RerankRetriever(bm25, model))
```

`QueryRewriteRetriever` (LLM rewrites the query), `HyDERetriever` (LLM writes a
hypothetical passage to retrieve with), and `RerankRetriever` (LLM listwise
reranks a candidate pool) each wrap a base retriever.

## Scoring

Correctness comes from a `Judge`; each task declares its official metric as
the default (see Quickstart). Cost and latency come from `Usage` accounting on
each answer. Two BrowseComp-Plus secondary metrics ride along: `mean_recall`
(gold docs among the cited ones; N/A for systems that cite nothing) and
`calibration_error` (ECE over stated confidences). `aggregate` reduces a run
to `AggregateScores`; `to_scores_dict` bridges to MTEB's scores dict with
accuracy as the main score.

Feasibility-gated systems (`full-context`) mark questions they cannot attempt
as not applicable; those are excluded from accuracy and reported as `coverage`.

## External agents

`rlm` and the Harbor agents wrap external runtimes behind the same
`AnswerSystem` contract. Harbor runs one batch `harbor run` job over all
questions (corpus bind-mounted read-only); the answer artifacts are read back
and scored by the same `Judge` as every other system. See **Setup** and
**Auth** for requirements.

## Extending

- **Task**: add a module under `tasks/` that builds an `AnswerTaskData` (via
  `from_mteb_retrieval` for a shared corpus or `from_per_question` for one
  corpus per question) and declares a `TaskMeta`; list it in
  `tasks.TASK_REGISTRY`.
- **System**: add a module under `systems/` implementing
  `AnswerSystem.answer(question, corpus) -> AnswerResult` and list a
  `SystemMeta` in `systems.SYSTEM_REGISTRY`.
- **Retriever**: any MTEB `SearchProtocol` model works as `retriever=` with no
  adapter.

## Layout

- `interface.py` contract: `AnswerSystem`, `CorpusHandle`, `ChatModel`, results.
- `corpus.py` `InMemoryCorpus`, `RetrievalCorpus` (wraps MTEB retrievers).
- `data.py` `AnswerTaskData`, `TaskMeta`, adapters from MTEB retrieval data.
- `evaluate.py` the single- and multi-system `evaluate()` front door, with
  corpus reuse within batch evaluations.
- `evaluator.py` per-question run loop (resilient, optionally concurrent).
- `metrics.py` judges, recall, calibration, and aggregation.
- `models.py` `OpenAIChatModel`.
- `retrievers.py` LLM retriever wrappers (query rewrite, HyDE, rerank).
- `harbor.py` Harbor dataset export, batch run, and result readers.
- `tasks/` one module per task, plus the task registry (`get_task`).
- `systems/` one module per paradigm, plus the system registry (`get_system`).

## References

- Coding Agents are Effective Long-Context Processors: arXiv 2603.20432
- BrowseComp-Plus: arXiv 2508.06600, `Tevatron/browsecomp-plus`
- RLM: arXiv 2512.24601
- OBLIQ-Bench (retrieval): arXiv 2605.06235
- Harbor: `github.com/harbor-framework/harbor`
