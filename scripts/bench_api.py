"""Benchmark the leaderboard API.

Hits every public endpoint, measures cold + warm latency, response size,
gzip ratio, and ETag-based 304 revalidation. Prints one table per
endpoint group so you can spot regressions at a glance.

Usage::

    # against a local server
    python scripts/bench_api.py

    # against a remote URL with more warm samples
    python scripts/bench_api.py --url https://api.example.com --warm-runs 20

    # focus on a single endpoint group
    python scripts/bench_api.py --group summary

The script makes no extra runtime dependencies — stdlib only.
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CallResult:
    status: int
    seconds: float
    raw_bytes: int  # body length on the wire
    decoded_bytes: int  # body length after gunzip
    encoding: str | None  # Content-Encoding header
    etag: str | None
    cache_control: str | None


@dataclass
class Stats:
    label: str
    path: str
    cold: CallResult
    warm_runs: list[CallResult] = field(default_factory=list)
    revalidate: CallResult | None = None

    @property
    def warm_p50_ms(self) -> float:
        return statistics.median(r.seconds for r in self.warm_runs) * 1000

    @property
    def warm_min_ms(self) -> float:
        return min(r.seconds for r in self.warm_runs) * 1000

    @property
    def warm_max_ms(self) -> float:
        return max(r.seconds for r in self.warm_runs) * 1000


def _request(
    url: str,
    *,
    if_none_match: str | None = None,
    accept_encoding: str = "gzip",
    timeout: float = 600.0,
) -> CallResult:
    req = urllib.request.Request(url)
    req.add_header("Accept-Encoding", accept_encoding)
    if if_none_match:
        req.add_header("If-None-Match", if_none_match)

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            elapsed = time.perf_counter() - t0
            status = resp.status
            encoding = resp.headers.get("Content-Encoding")
            decoded_len = (
                len(gzip.decompress(raw)) if encoding == "gzip" and raw else len(raw)
            )
            return CallResult(
                status=status,
                seconds=elapsed,
                raw_bytes=len(raw),
                decoded_bytes=decoded_len,
                encoding=encoding,
                etag=resp.headers.get("ETag"),
                cache_control=resp.headers.get("Cache-Control"),
            )
    except urllib.error.HTTPError as e:
        elapsed = time.perf_counter() - t0
        # 304 is the happy path for revalidation — surface it like a 2xx.
        return CallResult(
            status=e.code,
            seconds=elapsed,
            raw_bytes=0,
            decoded_bytes=0,
            encoding=e.headers.get("Content-Encoding"),
            etag=e.headers.get("ETag"),
            cache_control=e.headers.get("Cache-Control"),
        )


def _bench(base: str, label: str, path: str, *, warm_runs: int) -> Stats | None:
    url = base.rstrip("/") + path
    try:
        cold = _request(url)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {label:35s} cold request failed: {exc}", file=sys.stderr)
        return None
    if cold.status >= 400 and cold.status != 304:
        print(f"  ! {label:35s} HTTP {cold.status} (cold)", file=sys.stderr)
        return None

    warm = [_request(url) for _ in range(warm_runs)]

    revalidate: CallResult | None = None
    if cold.etag:
        revalidate = _request(url, if_none_match=cold.etag)

    return Stats(
        label=label, path=path, cold=cold, warm_runs=warm, revalidate=revalidate
    )


def _fmt_bytes(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.1f} KB"
    return f"{n} B"


def _fmt_ms(s: float) -> str:
    ms = s * 1000
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    if ms >= 10:
        return f"{ms:.0f} ms"
    return f"{ms:.2f} ms"


def _call_to_dict(c: CallResult) -> dict[str, Any]:
    return {
        "status": c.status,
        "seconds": round(c.seconds, 6),
        "ms": round(c.seconds * 1000, 3),
        "raw_bytes": c.raw_bytes,
        "decoded_bytes": c.decoded_bytes,
        "encoding": c.encoding,
        "etag": c.etag,
        "cache_control": c.cache_control,
    }


def _stats_to_dict(s: Stats) -> dict[str, Any]:
    warm_ms = [r.seconds * 1000 for r in s.warm_runs]
    return {
        "label": s.label,
        "path": s.path,
        "cold": _call_to_dict(s.cold),
        "warm": {
            "runs": [_call_to_dict(r) for r in s.warm_runs],
            "summary_ms": {
                "min": round(min(warm_ms), 3) if warm_ms else None,
                "max": round(max(warm_ms), 3) if warm_ms else None,
                "median": round(statistics.median(warm_ms), 3) if warm_ms else None,
                "mean": round(statistics.fmean(warm_ms), 3) if warm_ms else None,
                "p95": (
                    round(sorted(warm_ms)[max(int(len(warm_ms) * 0.95) - 1, 0)], 3)
                    if warm_ms
                    else None
                ),
            },
        },
        "revalidate_304": _call_to_dict(s.revalidate) if s.revalidate else None,
    }


def _write_json(
    path: str,
    base_url: str,
    warm_runs: int,
    samples: dict[str, str],
    groups: list[tuple[str, list[Stats]]],
) -> None:
    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "api_url": base_url,
        "warm_runs": warm_runs,
        "samples": samples,
        "groups": {
            title: [_stats_to_dict(s) for s in items] for title, items in groups
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")


def _print_group(title: str, results: Sequence[Stats]) -> None:
    if not results:
        return
    print(f"\n== {title} ==")
    header = (
        f"{'endpoint':38s}  {'cold':>10s}  {'warm p50':>10s}  "
        f"{'warm min':>9s}  {'warm max':>9s}  {'304':>8s}  "
        f"{'raw':>9s}  {'gzip':>9s}  enc"
    )
    print(header)
    print("-" * len(header))
    for s in results:
        rev = (
            _fmt_ms(s.revalidate.seconds)
            if s.revalidate and s.revalidate.status == 304
            else "—"
        )
        enc = s.cold.encoding or "—"
        print(
            f"{s.label:38s}  "
            f"{_fmt_ms(s.cold.seconds):>10s}  "
            f"{_fmt_ms(s.warm_p50_ms / 1000):>10s}  "
            f"{_fmt_ms(s.warm_min_ms / 1000):>9s}  "
            f"{_fmt_ms(s.warm_max_ms / 1000):>9s}  "
            f"{rev:>8s}  "
            f"{_fmt_bytes(s.cold.decoded_bytes):>9s}  "
            f"{_fmt_bytes(s.cold.raw_bytes):>9s}  "
            f"{enc}"
        )


def _pick_sample_names(base: str) -> dict[str, str]:
    """Grab one benchmark, task, and model name from the API to use as samples."""
    url = base.rstrip("/")

    def _first(path: str, key: str | None = None) -> str | None:
        try:
            with urllib.request.urlopen(url + path, timeout=30) as r:
                body = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                data: Any = json.loads(body)
        except Exception:  # noqa: BLE001
            return None
        if isinstance(data, list) and data:
            first = data[0]
            return first.get(key) if (isinstance(first, dict) and key) else first
        return None

    return {
        "benchmark": _first("/benchmarks", "name") or "MTEB(eng, v2)",
        "task": _first("/tasks", "name") or "STS22.v2",
        "model": _first("/models", "name") or "Qwen/Qwen3-Embedding-8B",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument(
        "--warm-runs",
        type=int,
        default=5,
        help="How many warm requests to time per endpoint (default: 5)",
    )
    parser.add_argument(
        "--group",
        choices=["health", "benchmarks", "tasks", "models", "all"],
        default="all",
        help="Limit the benchmark to one endpoint group",
    )
    parser.add_argument(
        "--benchmark",
        help="Benchmark name to use for /benchmarks/{name}/summary (auto-detected if omitted)",
    )
    parser.add_argument(
        "--task",
        help="Task name to use for /tasks/{name}/scores (auto-detected if omitted)",
    )
    parser.add_argument(
        "--model",
        help="Model name to use for /models/{name}/scores (auto-detected if omitted)",
    )
    parser.add_argument(
        "--json-out",
        help="Also write structured results (per-endpoint + per-run timings) to this path",
    )
    args = parser.parse_args(argv)

    base = args.url
    # Only hit the auto-detect endpoints (which can themselves be slow on a
    # cold cache) for sample names the user didn't supply.
    if args.benchmark and args.task and args.model:
        samples = {
            "benchmark": args.benchmark,
            "task": args.task,
            "model": args.model,
        }
    else:
        samples = _pick_sample_names(base)
        if args.benchmark:
            samples["benchmark"] = args.benchmark
        if args.task:
            samples["task"] = args.task
        if args.model:
            samples["model"] = args.model

    print(f"API:        {base}")
    print(f"Warm runs:  {args.warm_runs}")
    print(f"Samples:    benchmark={samples['benchmark']!r}")
    print(f"            task={samples['task']!r}")
    print(f"            model={samples['model']!r}")

    collected: list[tuple[str, list[Stats]]] = []

    def _run_group(title: str, items: list[Stats | None]) -> None:
        results = [r for r in items if r is not None]
        _print_group(title, results)
        if results:
            collected.append((title, results))

    bench: Callable[[str, str], Stats | None] = lambda label, path: _bench(
        base, label, path, warm_runs=args.warm_runs
    )

    enc = urllib.parse.quote

    if args.group in ("health", "all"):
        _run_group("health", [bench("/health", "/health")])

    if args.group in ("benchmarks", "all"):
        b = samples["benchmark"]
        _run_group(
            "benchmarks",
            [
                bench("/benchmarks/menu", "/benchmarks/menu"),
                bench("/benchmarks", "/benchmarks"),
                bench(f"/benchmarks/{b}", f"/benchmarks/{enc(b)}"),
                bench(
                    f"/benchmarks/{b}/summary",
                    f"/benchmarks/{enc(b)}/summary",
                ),
            ],
        )

    if args.group in ("tasks", "all"):
        t = samples["task"]
        _run_group(
            "tasks",
            [
                bench("/tasks (full)", "/tasks"),
                bench("/tasks?types=Retrieval", "/tasks?types=Retrieval"),
                bench(f"/tasks/{t}", f"/tasks/{enc(t)}"),
                bench(f"/tasks/{t}/scores", f"/tasks/{enc(t)}/scores"),
            ],
        )

    if args.group in ("models", "all"):
        m = samples["model"]
        _run_group(
            "models",
            [
                bench("/models (full)", "/models"),
                bench("/models?modalities=image", "/models?modalities=image"),
                bench(f"/models/{m}", f"/models/{enc(m)}"),
                bench(f"/models/{m}/scores", f"/models/{enc(m)}/scores"),
            ],
        )

    if args.json_out:
        _write_json(args.json_out, base, args.warm_runs, samples, collected)
        print(f"\nWrote JSON results to {args.json_out}")

    print(
        "\nLegend: cold = first hit (in-process cache miss). warm p50 = median of "
        f"{args.warm_runs} subsequent hits. 304 = revalidate with If-None-Match. "
        "raw / gzip = decoded vs on-wire body size."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
