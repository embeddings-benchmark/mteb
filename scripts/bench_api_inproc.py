"""In-process API benchmark.

Imports the FastAPI app, runs warmup, then times warm GETs to a fixed list of
endpoints via httpx + ASGITransport. Designed to be run twice (e.g. against
the working tree vs against ``git stash``ed HEAD) so the JSON outputs can be
diffed.

Usage::

    python scripts/bench_api_inproc.py --label after --out after.json --n 100
    git stash
    python scripts/bench_api_inproc.py --label before --out before.json --n 100
    git stash pop
    python scripts/bench_api_inproc.py --diff before.json after.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import time
import warnings
from pathlib import Path

import httpx

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


ENDPOINTS = [
    ("scores_multilingual", "/v1/benchmarks/MTEB(Multilingual, v2)/scores"),
    ("scores_BEIR", "/v1/benchmarks/BEIR/scores"),
    ("per_language_multilingual", "/v1/benchmarks/MTEB(Multilingual, v2)/per-language"),
    ("benchmarks_list", "/v1/benchmarks"),
    ("benchmarks_menu", "/v1/benchmarks/menu"),
    ("benchmark_detail", "/v1/benchmarks/BEIR"),
    ("tasks_list", "/v1/tasks"),
    ("tasks_filtered", "/v1/tasks?types=Retrieval"),
    ("models_list", "/v1/models"),
    ("task_scores", "/v1/tasks/SciDocsRR/scores"),
    ("model_scores_arctic", "/v1/models/Snowflake/snowflake-arctic-embed-l/scores"),
]


def _fmt(values: list[float]) -> dict:
    s = sorted(values)
    n = len(s)
    return {
        "n": n,
        "min": round(s[0], 3),
        "median": round(statistics.median(s), 3),
        "mean": round(statistics.fmean(s), 3),
        "p95": round(s[max(int(n * 0.95) - 1, 0)], 3),
        "p99": round(s[max(int(n * 0.99) - 1, 0)], 3),
        "max": round(s[-1], 3),
    }


async def _time_path(
    client: httpx.AsyncClient, path: str, n: int
) -> tuple[list[float], int]:
    # Warm twice — first call includes any cold deserialise on the
    # ASGITransport side.
    await client.get(path)
    await client.get(path)
    body_len = 0
    out: list[float] = []
    for _ in range(n):
        t = time.perf_counter()
        r = await client.get(path)
        out.append((time.perf_counter() - t) * 1000.0)
        if r.status_code != 200:
            raise RuntimeError(f"{path} → {r.status_code}")
        body_len = len(r.content)
    return out, body_len


async def _run_bench(n: int) -> dict:
    print("importing app (warmup runs in lifespan)…", flush=True)
    from mteb.api.app import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        await client.get("/health")  # triggers lifespan startup
        results: dict[str, dict] = {}
        for label, path in ENDPOINTS:
            try:
                lat, body_len = await _time_path(client, path, n)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {label}: {exc}")
                results[label] = {"path": path, "error": str(exc)}
                continue
            stats = _fmt(lat)
            results[label] = {"path": path, "bytes": body_len, **stats}
            print(
                f"  {label:<22} bytes={body_len:>10,} "
                f"median={stats['median']:>7.2f}ms p95={stats['p95']:>7.2f}ms "
                f"mean={stats['mean']:>7.2f}ms"
            )
    return results


def _print_diff(before: dict, after: dict) -> None:
    print(
        f"\n{'endpoint':<22} {'before med':>12} {'after med':>12} {'Δ med':>10}  "
        f"{'before p95':>12} {'after p95':>12} {'Δ p95':>10}  speedup"
    )
    print("-" * 110)
    for key in before:
        if key not in after or "error" in before[key] or "error" in after[key]:
            continue
        bm = before[key]["median"]
        am = after[key]["median"]
        bp = before[key]["p95"]
        ap = after[key]["p95"]
        speedup = bm / am if am > 0 else float("inf")
        print(
            f"{key:<22} {bm:>10.2f}ms {am:>10.2f}ms {am - bm:>+8.2f}ms  "
            f"{bp:>10.2f}ms {ap:>10.2f}ms {ap - bp:>+8.2f}ms  {speedup:>5.2f}x"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100, help="warm requests/endpoint")
    parser.add_argument("--label", default="run", help="label for this run")
    parser.add_argument("--out", type=Path, help="dump results JSON here")
    parser.add_argument(
        "--diff", nargs=2, metavar=("BEFORE", "AFTER"), help="diff two result JSONs"
    )
    args = parser.parse_args()

    if args.diff:
        before = json.loads(Path(args.diff[0]).read_text())
        after = json.loads(Path(args.diff[1]).read_text())
        _print_diff(before, after)
        return

    print(f"[{args.label}] running benchmark, n={args.n} per endpoint")
    results = await _run_bench(args.n)
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
