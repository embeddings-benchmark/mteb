"""Render every benchmark / task / model OG hero card by screenshotting
the parameterised template at ``scripts/og-template/template.html``.

Imports the mteb registry directly — no HTTP round-trip needed since
this script ships in the same repo as the data it's rendering. The
template loads from a ``file://`` URL (relative ``dots-icon.png`` resolves
against the same directory), so no local web server has to be running
either. Playwright drives Chromium against the template; for each
entity we mutate the DOM in place and snapshot a PNG.

Incremental: every PNG carries a sidecar ``${slug}.hash`` containing the
SHA-1 of its rendering inputs. On the next run we recompute the hash
per entity, compare to the sidecar, and skip the Playwright trip when
they match. So a steady-state rebuild after one new model lands only
re-renders the one model.

Output layout::

    <OG_DIR>/benchmark/<name>.png
    <OG_DIR>/benchmark/<name>.hash
    <OG_DIR>/task/...
    <OG_DIR>/model/...
    <OG_DIR>/manifest.json

Usage::

    python scripts/generate_og_images.py                       # all defaults
    python scripts/generate_og_images.py --out=/data/og
    python scripts/generate_og_images.py --only benchmarks
    python scripts/generate_og_images.py --concurrency=8
    python scripts/generate_og_images.py --force               # bypass incremental skip
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, TypedDict

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

if TYPE_CHECKING:
    # Type-only imports — runtime users go through ``_load_catalogue``
    # which does the actual import lazily so the script can be imported
    # without mteb on PYTHONPATH (handy for tests).
    from mteb.api.schemas import BenchmarkSchema, ModelMetaSchema, TaskMetaSchema

logger = logging.getLogger("scripts.generate_og_images")

Kind = Literal["benchmark", "task", "model"]


class CardParams(TypedDict):
    """Render-time parameters for one OG card.

    Every key here is read in the browser-side ``_RENDER_JS`` block:
    we hand the dict straight to ``page.evaluate(_RENDER_JS, params)``.
    Stable key set + key names so the JS doesn't have to guard against
    missing fields and so :func:`_params_hash` stays deterministic.
    """

    kind: str
    typeLabel: str
    eyebrow: str
    title: str
    tagline: str
    stats: str
    headlineSize: str


class RenderJob(TypedDict):
    """One ``CardParams`` plus its on-disk artefacts.

    ``hash`` is the SHA-1 of the params (see :func:`_params_hash`);
    we write it to ``hash_path`` so the next run can short-circuit
    when nothing about the inputs changed.
    """

    out: Path
    hash_path: Path
    hash: str
    params: CardParams


# JS evaluated in the page to update DOM in place. Playwright treats
# the string as the body of an async function — we pass the per-entity
# params object as its single argument.
_RENDER_JS = """
(args) => {
    const { kind, typeLabel, eyebrow, title, tagline, stats, headlineSize } = args;
    document.getElementById('eyebrow').textContent = eyebrow;
    document.getElementById('kindBadge').textContent = kind ?? eyebrow ?? '';
    document.getElementById('typeBadge').textContent = typeLabel ?? '';
    document.getElementById('headline').textContent = title;
    document.getElementById('tagline').textContent = tagline ?? '';
    document
        .getElementById('card')
        .style.setProperty('--headline-size', headlineSize || '78px');
    const statsEl = document.getElementById('stats');
    statsEl.replaceChildren();
    const items = (stats || '').split(',').map((s) => s.split('|'));
    items.forEach(([label, num], i) => {
        if (i > 0) {
            const sep = document.createElement('div');
            sep.className = 'stat-sep';
            statsEl.appendChild(sep);
        }
        const wrap = document.createElement('div');
        const n = document.createElement('div');
        n.className = 'stat-num';
        n.textContent = num;
        const l = document.createElement('div');
        l.className = 'stat-label';
        l.textContent = label;
        wrap.appendChild(n);
        wrap.appendChild(l);
        statsEl.appendChild(wrap);
    });
}
"""

_AWAIT_FONTS_AND_ICON_JS = """
async () => {
    await document.fonts.ready;
    const img = document.querySelector('.brand img');
    if (img && !(img.complete && img.naturalWidth > 0)) {
        await new Promise((res) => {
            img.addEventListener('load', res, { once: true });
            img.addEventListener('error', res, { once: true });
        });
    }
}
"""

# Hash input keys. Stable order so re-ordering kwargs in the planners
# never invalidates existing PNG files. Adding a key here is a cache-breaking
# change — every entity re-renders on the next run.
_HASH_FIELDS: Final[tuple[str, ...]] = (
    "kind",
    "typeLabel",
    "eyebrow",
    "title",
    "tagline",
    "stats",
    "headlineSize",
)


def _slug(name: str) -> str:
    """Return ``name`` unchanged for use as a filesystem-relative path.

    Starlette's ``StaticFiles`` percent-decodes the URL path *before*
    the file lookup — so it expects the on-disk filename to use
    literal characters. Earlier versions of this slug called
    ``quote(name)`` and tried to match the frontend's encoded URL
    character-for-character; that 404s because the URL is decoded
    on the way in. Letting ``/`` survive as a real path separator
    is the only transformation we need (it gets us the nested
    ``model/{org}/{name}.png`` layout for free via :mod:`pathlib`).

    On Linux containers — our deployment target — every character
    that can appear in a benchmark / task / model name (``(``, ``)``,
    ``,``, space, ``=``, …) is a legal filename character, so no
    sanitisation is needed.
    """
    return name


def _params_hash(params: CardParams) -> str:
    h = hashlib.sha1(usedforsecurity=False)
    for k in _HASH_FIELDS:
        h.update(f"{k} {params.get(k, '') or ''} ".encode())
    return h.hexdigest()


def _fmt_num(n: int | float | None) -> str:
    # Narrow no-break space as thousands separator — matches the
    # home hero on the frontend.
    if n is None:
        return "—"
    return f"{int(n):,}".replace(",", " ")


def _fmt_params(b: float | None) -> str:
    if not b:
        return "—"
    if b >= 1:
        return f"{b:.1f}B"
    return f"{round(b * 1000)}M"


def _fmt_tokens(n: int | None) -> str:
    if not n:
        return "—"
    if n >= 1000:
        return f"{round(n / 1024)}K"
    return str(n)


def _pick_headline_size(text: str) -> str:
    """Step the headline font size down so long names don't overflow.

    Tuned against the 1200 × 630 card; breakpoints map to roughly-equal
    visual weights across model / task / benchmark titles.
    """
    n = len(text)
    if n <= 18:
        return "92px"
    if n <= 26:
        return "78px"
    if n <= 35:
        return "64px"
    if n <= 48:
        return "54px"
    return "46px"


def _trim_description(desc: str | None, max_len: int = 140) -> str:
    if not desc:
        return ""
    flat = " ".join(desc.split())
    if len(flat) <= max_len:
        return flat
    return flat[: max_len - 1].rstrip() + "…"


def _plan_job(out_dir: Path, kind: Kind, name: str, params: CardParams) -> RenderJob:
    file = _slug(name)
    out_path = out_dir / kind / f"{file}.png"
    # Slugs with embedded `/` (e.g. `microsoft/harrier-...`) write into
    # an org-named subdirectory — ensure it exists before the renderer
    # tries to write the PNG.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return RenderJob(
        out=out_path,
        hash_path=out_dir / kind / f"{file}.hash",
        hash=_params_hash(params),
        params=params,
    )


def plan_benchmarks(
    entries: Iterable[BenchmarkSchema], out_dir: Path
) -> list[RenderJob]:
    """Build the render jobs for benchmark cards.

    Stats are pure registry metadata (task / language / domain counts) —
    the ``num_models`` field is intentionally not surfaced because it
    requires the results polars frame to be populated, and we want
    the generator to run without depending on that download.
    """
    jobs: list[RenderJob] = []
    for b in entries:
        title = b.display_name or b.name
        tagline = _trim_description(
            b.description or "Benchmark on the MTEB Leaderboard."
        )
        stats = ",".join(
            [
                f"Tasks|{_fmt_num(len(b.tasks))}",
                f"Languages|{_fmt_num(len(b.languages))}",
                f"Domains|{_fmt_num(len(b.domains))}",
            ]
        )
        params: CardParams = {
            "kind": "Benchmark",
            # No secondary type for benchmarks — `.type-badge:empty`
            # in the template CSS collapses the empty slot.
            "typeLabel": "",
            "eyebrow": "Benchmark",
            "title": title,
            "tagline": tagline,
            "stats": stats,
            "headlineSize": _pick_headline_size(title),
        }
        jobs.append(_plan_job(out_dir, "benchmark", b.name, params))
    return jobs


def plan_tasks(entries: Iterable[TaskMetaSchema], out_dir: Path) -> list[RenderJob]:
    """Build the render jobs for task cards. Pure registry metadata."""
    jobs: list[RenderJob] = []
    for t in entries:
        tagline = _trim_description(
            t.description or f"{t.type or ''} task on the MTEB Leaderboard."
        )
        stats = ",".join(
            [
                f"Type|{t.type or '—'}",
                f"Languages|{_fmt_num(len(t.languages))}",
                f"Domains|{_fmt_num(len(t.domains))}",
            ]
        )
        params: CardParams = {
            "kind": "Task",
            # Task type (e.g. STS / Classification / Retrieval) goes on
            # the secondary badge — single most useful classifier at a
            # glance without parsing the stats grid.
            "typeLabel": t.type or "",
            "eyebrow": "Task",
            "title": t.name,
            "tagline": tagline,
            "stats": stats,
            "headlineSize": _pick_headline_size(t.name),
        }
        jobs.append(_plan_job(out_dir, "task", t.name, params))
    return jobs


def plan_models(entries: Iterable[ModelMetaSchema], out_dir: Path) -> list[RenderJob]:
    """Build the render jobs for model cards."""
    jobs: list[RenderJob] = []
    for m in entries:
        flags = " · ".join(
            x
            for x in (
                "open weights" if m.open_weights else "proprietary",
                "instruction-tuned" if m.instruction_tuned else None,
            )
            if x
        )
        stats = ",".join(
            [
                f"Params|{_fmt_params(m.total_params_b)}",
                f"Dim|{_fmt_num(m.embedding_dim)}",
                f"Tokens|{_fmt_tokens(m.max_tokens)}",
            ]
        )
        params: CardParams = {
            "kind": "Model",
            # Model type (dense / cross-encoder / late-interaction /
            # sparse / router) on the secondary badge.
            "typeLabel": m.model_type or "",
            "eyebrow": "Model",
            "title": m.name,
            "tagline": _trim_description(flags),
            "stats": stats,
            "headlineSize": _pick_headline_size(m.name),
        }
        jobs.append(_plan_job(out_dir, "model", m.name, params))
    return jobs


def _should_skip(out_path: Path, hash_path: Path, expected: str, force: bool) -> bool:
    if force:
        return False
    if not out_path.is_file():
        return False
    try:
        return hash_path.read_text().strip() == expected
    except OSError:
        return False


async def _boot_worker(ctx: BrowserContext, template_url: str) -> Page:
    page = await ctx.new_page()
    await page.goto(template_url, wait_until="load")
    await page.evaluate(_AWAIT_FONTS_AND_ICON_JS)
    return page


async def _capture(page: Page, job: RenderJob) -> None:
    await page.evaluate(_RENDER_JS, job["params"])
    await page.locator(".card").screenshot(
        path=str(job["out"]), type="png", scale="css"
    )
    job["hash_path"].write_text(job["hash"])


class PoolResult(TypedDict):
    rendered: int
    skipped: int


async def _run_pool(
    ctx: BrowserContext,
    jobs: list[RenderJob],
    label: str,
    template_url: str,
    concurrency: int,
    force: bool,
) -> PoolResult:
    if not jobs:
        return {"rendered": 0, "skipped": 0}
    rendered = 0
    skipped = 0
    cursor = 0

    async def worker() -> None:
        nonlocal cursor, rendered, skipped
        page: Page | None = None
        try:
            while True:
                idx = cursor
                cursor += 1
                if idx >= len(jobs):
                    break
                job = jobs[idx]
                if _should_skip(job["out"], job["hash_path"], job["hash"], force):
                    skipped += 1
                    continue
                # Lazy boot — if every job in this worker's queue was
                # skippable we never pay the cost of opening a tab.
                if page is None:
                    page = await _boot_worker(ctx, template_url)
                await _capture(page, job)
                rendered += 1
                if (rendered + skipped) % 50 == 0:
                    logger.info(
                        "[%s] %s rendered, %s skipped", label, rendered, skipped
                    )
        finally:
            if page is not None:
                await page.close()

    pool_size = min(concurrency, len(jobs))
    await asyncio.gather(*(worker() for _ in range(pool_size)))
    logger.info(
        "[og] %s: %s rendered, %s skipped (%s total)",
        label,
        rendered,
        skipped,
        len(jobs),
    )
    return {"rendered": rendered, "skipped": skipped}


def _load_catalogue(
    only: Sequence[str],
) -> tuple[list[BenchmarkSchema], list[TaskMetaSchema], list[ModelMetaSchema]]:
    """Walk the mteb registry and return schema objects per kind.

    Pure metadata only — no ``num_models`` overlay. We deliberately
    skip :func:`mteb.api.cache.warmup_blocking` and the
    ``_with_num_models`` / ``_with_task_num_models`` route helpers
    because computing real per-entity model counts means building the
    unified polars results frame (~12 s) just to populate one stat.
    The hero cards stick to fields that come straight from the
    registry — task / language / domain counts on benchmarks, task
    type on tasks, params / dim / tokens on models — so the generator
    stays cheap and never depends on the results dataset being
    downloaded.
    """
    import mteb
    from mteb.api.adapters import (
        benchmark_to_schema,
        model_meta_to_schema,
        task_to_meta_schema,
    )

    benches: list[BenchmarkSchema] = []
    tasks: list[TaskMetaSchema] = []
    models: list[ModelMetaSchema] = []
    if "benchmarks" in only:
        benches = [benchmark_to_schema(b) for b in mteb.get_benchmarks()]
    if "tasks" in only:
        tasks = [
            task_to_meta_schema(t)
            for t in mteb.get_tasks(
                exclude_beta=False,
                exclude_private=False,
                exclude_superseded=False,
            )
        ]
    if "models" in only:
        models = [model_meta_to_schema(m) for m in mteb.get_model_metas()]
    return benches, tasks, models


_TEMPLATE_DIR = Path(__file__).resolve().parent / "og-template"


def _template_url() -> str:
    """Return the ``file://`` URL of the bundled template.

    Lives next to this script under ``scripts/og-template/`` so the
    template + ``dots-icon.png`` it references stay co-located with the
    generator that uses them, instead of inside the runtime package.
    Playwright resolves the relative ``dots-icon.png`` against this URL.
    """
    return (_TEMPLATE_DIR / "template.html").as_uri()


async def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind in ("benchmark", "task", "model"):
        (out_dir / kind).mkdir(parents=True, exist_ok=True)

    only = tuple(args.only.split(","))
    template_url = _template_url()

    logger.info(
        "[og] target: %s  pool: %s%s",
        out_dir,
        args.concurrency,
        "  (force re-render)" if args.force else "",
    )

    benches, tasks, models = _load_catalogue(only)
    logger.info(
        "[og] catalogue: %s benchmarks, %s tasks, %s models",
        len(benches),
        len(tasks),
        len(models),
    )

    bench_jobs = plan_benchmarks(benches, out_dir)
    task_jobs = plan_tasks(tasks, out_dir)
    model_jobs = plan_models(models, out_dir)

    started = time.time()
    totals: PoolResult = {"rendered": 0, "skipped": 0}
    async with async_playwright() as p:
        browser: Browser = await p.chromium.launch()
        ctx = await browser.new_context(viewport={"width": 1200, "height": 630})
        try:
            for kind_label, jobs in (
                ("benchmarks", bench_jobs),
                ("tasks", task_jobs),
                ("models", model_jobs),
            ):
                r = await _run_pool(
                    ctx, jobs, kind_label, template_url, args.concurrency, args.force
                )
                totals["rendered"] += r["rendered"]
                totals["skipped"] += r["skipped"]
        finally:
            await browser.close()

    elapsed = time.time() - started
    total = len(bench_jobs) + len(task_jobs) + len(model_jobs)
    logger.info(
        "[og] done: %s rendered, %s skipped of %s total in %.1fs",
        totals["rendered"],
        totals["skipped"],
        total,
        elapsed,
    )

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": {
            "benchmark": len(bench_jobs),
            "task": len(task_jobs),
            "model": len(model_jobs),
        },
        "last_run": {
            "rendered": totals["rendered"],
            "skipped": totals["skipped"],
            "seconds": round(elapsed, 1),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render per-entity Open Graph hero cards for every benchmark / task / model."
    )
    p.add_argument(
        "--out",
        default=os.environ.get("MTEB_API_OG_DIR", "/data/og"),
        help="Output directory for the generated PNG files (defaults to $MTEB_API_OG_DIR or /data/og).",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("MTEB_API_OG_CONCURRENCY", "6")),
        help="Worker pool size (default 6). Higher = faster but more memory / CPU.",
    )
    p.add_argument(
        "--only",
        default="benchmarks,tasks,models",
        help="Comma-separated subset of {benchmarks,tasks,models} to (re)generate.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-render every entity even if the hash sidecar already matches.",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    return asyncio.run(run(_parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
