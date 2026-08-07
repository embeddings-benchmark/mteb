"""Harbor adapter: export an answer-mode task as a Harbor dataset, run it batched.

Follows Harbor's canonical adapter pattern (harbor-framework/harbor/adapters):
convert the task into a directory of per-question task folders, run one
`harbor run -p <dataset> -a <agent> -n K` job (isolated trials, K concurrent),
and read each trial's answer artifact back so our Judge scores it host-side.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

logger = logging.getLogger(__name__)

# Host auth env vars forwarded into the agent container (subscription or API key).
_FORWARDED_AUTH_ENV = (
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)

_INSTRUCTION = """Answer the question using only the documents in /corpus, one
text file per document. Explore them with the tools available to you (grep, cat,
and so on). When you are done, write your final answer, and nothing else, to
/workdir/answer.txt.

Question: {question}
"""

# Per-question corpora are baked into each task image; a shared corpus is
# materialized once and bind-mounted at /corpus into every trial.
_DOCKERFILE_COPY = "FROM python:3.11-slim\nWORKDIR /workdir\nCOPY corpus/ /corpus/\n"
_DOCKERFILE_MOUNT = "FROM python:3.11-slim\nWORKDIR /workdir\n"
# With a retriever tool, ship the BM25 search script into the image too.
_DOCKERFILE_COPY_RETRIEVER = _DOCKERFILE_COPY + "COPY search.py /workdir/search.py\n"
_DOCKERFILE_MOUNT_RETRIEVER = _DOCKERFILE_MOUNT + "COPY search.py /workdir/search.py\n"

_RETRIEVER_INSTRUCTION = """Answer the question using only the documents in
/corpus, one text file per document. You also have a BM25 search tool: run
`python /workdir/search.py "your query" 10` to list the most relevant document
files (it builds an index on first use). Use it, grep, cat, or any combination.
When you are done, write your final answer, and nothing else, to
/workdir/answer.txt.

Question: {question}
"""

# Dependency-free BM25 search tool, shipped in when a retriever is requested.
_SEARCH_SCRIPT = r'''#!/usr/bin/env python3
"""BM25 search over /corpus. Usage: python search.py "query" [top_k]."""
import glob
import math
import os
import pickle
import re
import sys

CORPUS = "/corpus"
CACHE = "/workdir/.bm25_index.pkl"
K1, B = 1.5, 0.75


def _tok(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _build():
    files = sorted(glob.glob(os.path.join(CORPUS, "**", "*.txt"), recursive=True))
    postings, lengths = {}, []
    for idx, path in enumerate(files):
        try:
            with open(path, encoding="utf-8", errors="ignore") as handle:
                toks = _tok(handle.read())
        except OSError:
            toks = []
        counts = {}
        for word in toks:
            counts[word] = counts.get(word, 0) + 1
        for word, freq in counts.items():
            postings.setdefault(word, []).append((idx, freq))
        lengths.append(len(toks))
    index = {
        "files": files,
        "lengths": lengths,
        "postings": postings,
        "n": len(files),
        "avgdl": (sum(lengths) / len(lengths)) if lengths else 1.0,
    }
    with open(CACHE, "wb") as handle:
        pickle.dump(index, handle)
    return index


def _load():
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as handle:
            return pickle.load(handle)
    return _build()


def search(query, top_k):
    index = _load()
    n, avgdl = index["n"], index["avgdl"] or 1.0
    scores = {}
    for word in set(_tok(query)):
        posting = index["postings"].get(word)
        if not posting:
            continue
        idf = math.log(1 + (n - len(posting) + 0.5) / (len(posting) + 0.5))
        for doc_idx, freq in posting:
            dl = index["lengths"][doc_idx] or 1
            scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * freq * (K1 + 1) / (
                freq + K1 * (1 - B + B * dl / avgdl)
            )
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
    return [(score, index["files"][idx]) for idx, score in ranked]


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else ""
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    for hit_score, hit_path in search(q, k):
        print(f"{hit_score:.3f}\t{hit_path}")
'''

# Grading is host-side via our Judge (--disable-verification), so this only
# satisfies Harbor's task-layout validation.
_TEST_SH = "#!/usr/bin/env bash\nexit 0\n"

_TASK_TOML = """schema_version = "1.1"
artifacts = []
[task]
name = "{name}"
description = "Answer-mode retrieval question over a fixed corpus."
authors = []
keywords = []
[metadata]
category = "retrieval-qa"
tags = ["mteb", "retrieval", "agentic"]
[verifier]
timeout_sec = 60.0

[verifier.env]
[agent]
timeout_sec = {agent_timeout}
[environment]
build_timeout_sec = 600.0
os = "linux"
cpus = 1
memory_mb = 2048
storage_mb = 8192
gpus = 0
allow_internet = true
mcp_servers = []

[environment.env]
[solution.env]
"""


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def _materialize(documents: Mapping[str, Mapping[str, str]], corpus_dir: Path) -> None:
    # One text file per document, named by docid, under one corpus directory.
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, doc in documents.items():
        (corpus_dir / f"{_safe(doc_id)}.txt").write_text(
            doc.get("text", ""), encoding="utf-8"
        )


def _write_task(
    task_dir: Path,
    name: str,
    question: str,
    documents: Mapping[str, Mapping[str, str]] | None,
    agent_timeout_s: float,
    *,
    retriever_tool: bool = False,
) -> None:
    env_dir, tests_dir = task_dir / "environment", task_dir / "tests"
    for path in (env_dir, tests_dir):
        path.mkdir(parents=True, exist_ok=True)
    # Harbor requires an org/name task name; the slug also names the trial dir.
    (task_dir / "task.toml").write_text(
        _TASK_TOML.format(name=f"mteb-agentic/{name}", agent_timeout=agent_timeout_s),
        encoding="utf-8",
    )
    instruction = _RETRIEVER_INSTRUCTION if retriever_tool else _INSTRUCTION
    (task_dir / "instruction.md").write_text(
        instruction.format(question=question), encoding="utf-8"
    )
    if retriever_tool:
        (env_dir / "search.py").write_text(_SEARCH_SCRIPT, encoding="utf-8")
    if documents is None:  # shared corpus, bind-mounted at /corpus
        dockerfile = (
            _DOCKERFILE_MOUNT_RETRIEVER if retriever_tool else _DOCKERFILE_MOUNT
        )
        (env_dir / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    else:  # per-question corpus, baked into the image
        dockerfile = _DOCKERFILE_COPY_RETRIEVER if retriever_tool else _DOCKERFILE_COPY
        (env_dir / "Dockerfile").write_text(dockerfile, encoding="utf-8")
        _materialize(documents, env_dir / "corpus")
    (tests_dir / "test.sh").write_text(_TEST_SH, encoding="utf-8")


def to_harbor_dataset(
    questions: Mapping[str, str],
    corpus_for: Callable[[str], Mapping[str, Mapping[str, str]]],
    out_dir: str | Path,
    *,
    shared_documents: Mapping[str, Mapping[str, str]] | None = None,
    agent_timeout_s: float = 1800.0,
    retriever_tool: bool = False,
) -> Path | None:
    """Write one Harbor task folder per question, named q0..qN in question order.

    With shared_documents (one corpus for every question) it is materialized once
    and the returned path is bind-mounted into every trial; otherwise each task
    bakes its own corpus_for(qid). read_harbor_answers keys answers by the slug.
    retriever_tool ships a BM25 search script the agent may call.
    """
    out = Path(out_dir)
    mount_dir: Path | None = None
    if shared_documents is not None:
        # A sibling of the dataset dir, so Harbor's task scan never sees it.
        mount_dir = (out.parent / "corpus").resolve()
        _materialize(shared_documents, mount_dir)
    for index, (qid, question) in enumerate(questions.items()):
        slug = f"q{index}"
        docs = None if shared_documents is not None else corpus_for(qid)
        _write_task(
            out / slug,
            slug,
            question,
            docs,
            agent_timeout_s,
            retriever_tool=retriever_tool,
        )
    return mount_dir


def run_harbor(
    dataset_dir: str | Path,
    agent: str,
    model: str | None,
    jobs_dir: str | Path,
    *,
    n_concurrent: int = 8,
    agent_env: list[str] | None = None,
    mount_corpus: str | Path | None = None,
) -> None:
    """Run one batch Harbor job over the dataset.

    Well-known auth env vars set on the host (e.g. CLAUDE_CODE_OAUTH_TOKEN for a
    Claude subscription, ANTHROPIC_API_KEY, OPENAI_API_KEY) are forwarded into
    the agent automatically; explicit agent_env entries override them.
    """
    cmd = [
        "harbor",
        "run",
        "-p",
        str(dataset_dir),
        "-a",
        agent,
        "-o",
        str(jobs_dir),
        "-y",
        "-n",
        str(n_concurrent),
        "--disable-verification",
        "--artifact",
        "/workdir/answer.txt",
    ]
    if model:
        cmd += ["-m", str(model)]
    if mount_corpus is not None:  # shared corpus mounted read-only at /corpus
        cmd += ["--mounts-json", json.dumps([f"{mount_corpus}:/corpus:ro"])]
    explicit_keys = {kv.split("=", 1)[0] for kv in agent_env or []}
    forwarded = [
        f"{var}={os.environ[var]}"
        for var in _FORWARDED_AUTH_ENV
        if var in os.environ and var not in explicit_keys
    ]
    n_secrets = 0
    for kv in (*forwarded, *(agent_env or [])):
        cmd += ["--ae", kv]
        n_secrets += 1
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        # Re-raise without argv so forwarded auth values stay out of tracebacks.
        raise RuntimeError(
            f"harbor run failed with exit code {exc.returncode} "
            f"(command redacted, {n_secrets} env values forwarded)."
        ) from None


def read_harbor_answers(jobs_dir: str | Path) -> dict[str, str]:
    """Read each trial's answer artifact, keyed by the q0..qN task slug."""
    answers: dict[str, str] = {}
    for artifact in Path(jobs_dir).rglob("artifacts/answer.txt"):
        # trial dir is <slug>__<shortid>; the slug recovers as the prefix.
        slug = artifact.parent.parent.name.rsplit("__", 1)[0]
        answers[slug] = artifact.read_text(encoding="utf-8").strip()
    return answers


def _span_seconds(span: object) -> float | None:
    """Elapsed seconds for a Harbor {started_at, finished_at} span."""
    if not isinstance(span, dict):
        return None
    start, end = span.get("started_at"), span.get("finished_at")
    if not start or not end:
        return None
    from datetime import datetime

    try:
        return (
            datetime.fromisoformat(end) - datetime.fromisoformat(start)
        ).total_seconds()
    except ValueError:
        return None


def read_harbor_metrics(jobs_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Read each trial's token, cost, and latency metrics, keyed by the task slug."""
    metrics: dict[str, dict[str, Any]] = {}
    for result in Path(jobs_dir).rglob("result.json"):
        slug = result.parent.name.rsplit("__", 1)[0]  # trial dir is q<N>__<shortid>
        if not (slug.startswith("q") and slug[1:].isdigit()):
            continue  # skip the job-level result.json (timestamp dir)
        try:
            data = json.loads(result.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("unreadable Harbor result %s; metrics skipped", result)
            continue
        agent = data.get("agent_result") or {}
        metrics[slug] = {
            "prompt_tokens": agent.get("n_input_tokens") or 0,
            "completion_tokens": agent.get("n_output_tokens") or 0,
            "cost_usd": agent.get("cost_usd"),
            "latency_s": _span_seconds(data.get("agent_execution")),
        }
    return metrics
