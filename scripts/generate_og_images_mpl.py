"""Render every benchmark / task / model Open Graph hero card using
matplotlib instead of headless Chromium.

Drop-in alternative to ``generate_og_images.py``: same output layout,
same sidecar-hash incremental skip, same CLI surface (``--workers``
plays the role of ``--concurrency`` for the Playwright variant).
Trades the Playwright + Chromium dependency (~700 MB at build time)
for a pure Python/matplotlib pipeline at the cost of design fidelity
— the gradient text and radial-glow backgrounds are recreated with
numpy + imshow rather than CSS, and the typography uses whatever
Inter-like font matplotlib's font manager can locate.

Parallelism is process-based (``multiprocessing.Pool`` with
``spawn``) because matplotlib's font / freetype state isn't designed
for thread sharing. Each worker re-imports the script and rebuilds
its own background-array cache on first use — the per-worker boot
costs ~1 s, so the in-process path stays faster for batches under
~8 cards.

When to use which:

* ``generate_og_images.py`` (Playwright) — the canonical card design.
  Pixel-identical to whatever the browser shows when previewing the
  template locally. Required if you keep iterating on the visual
  design in HTML/CSS.

* ``generate_og_images_mpl.py`` (this file) — lighter dependency.
  Good for environments where bundling Chromium is wasteful (small
  CI builders, slim Docker images) and the cards don't need pixel
  parity with the HTML template.

Both write to the same filesystem layout under
``$MTEB_API_OG_DIR``, so swapping the generator for a future rebuild
is just swapping the script invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, TypedDict

import matplotlib

# Force the Agg backend before any pyplot import. Agg is the only
# matplotlib backend that runs entirely off-screen without a display
# server, and it's what we'd want in a Docker build anyway.
matplotlib.use("Agg")

import textwrap  # noqa: E402

import numpy as np  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.image import imread  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402

if TYPE_CHECKING:
    # Type-only — actual import happens lazily in ``_load_catalogue``
    # so the script can be imported in environments without mteb on
    # PYTHONPATH.
    from mteb.api.schemas import BenchmarkSchema, ModelMetaSchema, TaskMetaSchema

logger = logging.getLogger("scripts.generate_og_images_mpl")


Kind = Literal["benchmark", "task", "model"]


class CardParams(TypedDict):
    """Render-time parameters for one OG card.

    Mirrors the dict the Playwright sibling script hands to
    ``page.evaluate()``. Kept identical so the SHA-1 input hash is
    portable between the two renderers and changing implementations
    doesn't invalidate the entire on-disk cache.
    """

    kind: str
    typeLabel: str
    eyebrow: str
    title: str
    tagline: str
    stats: str
    headlineSize: str


class RenderJob(TypedDict):
    out: Path
    hash_path: Path
    hash: str
    params: CardParams


class PoolResult(TypedDict):
    rendered: int
    skipped: int


class _FontStyle(TypedDict):
    """Subset of matplotlib text kwargs used to size & style a label.

    Lets us define a font style as a literal once and spread it into
    both ``_measure_text`` and ``_draw_text`` calls without losing
    type info on the way through ``**kwargs``.
    """

    fontsize: int
    fontweight: int
    family: list[str]


# ── Visual constants ───────────────────────────────────────────────
#
# The card is 1200 × 630 (the OG standard). We work in display pixels
# throughout: matplotlib's coordinate system is configured to match,
# so a stat label at "x=720, y=480" lines up with the same pixel in
# the output PNG.

CARD_W = 1200
CARD_H = 630
DPI = 100

BG_BASE = "#0a0d12"
INK_STRONG = "#ffffff"
INK_HEADLINE_TINT = "#d3e0f8"  # bottom of the gradient text colour stop
INK_TAGLINE = "#9aa8c5"
INK_BRAND_NAME = "#e3eaf8"
INK_BRAND_EYEBROW = "#5b8fff"
STAT_NUM_COLOR = "#5b8fff"
STAT_LABEL_COLOR = "#6f7e9e"
STAT_SEP_RGBA = (0.604, 0.659, 0.773, 0.18)

BADGE_KIND_BG = ("#5b8fff", "#3d7bff")  # top → bottom gradient stops
BADGE_KIND_INK = "#ffffff"
BADGE_TYPE_BG_RGBA = (0.357, 0.561, 1.0, 0.14)
BADGE_TYPE_BORDER_RGBA = (0.357, 0.561, 1.0, 0.35)
BADGE_TYPE_INK = "#c8d8ff"

BOTTOM_STRIPE_STOPS = ((0.0, "#5b8fff"), (0.70, "#a06bff"), (1.0, "#5b8fff"))

# Scattered embedding-cloud dots: (x, y, diameter, alpha, hex_color).
# Pulled verbatim from the HTML template so the visual layout matches.
DOT_SPEC: tuple[tuple[int, int, int, float, str], ...] = (
    (880, 60, 70, 0.18, "#6fa1ff"),
    (1000, 200, 48, 0.22, "#3d7bff"),
    (820, 380, 96, 0.12, "#5b8fff"),
    (90, 480, 60, 0.18, "#6fa1ff"),
    (1100, 460, 50, 0.20, "#3d7bff"),
    (980, 110, 38, 0.40, "#5b8fff"),
    (880, 250, 28, 0.50, "#6fa1ff"),
    (1080, 320, 36, 0.40, "#4d83ff"),
    (950, 480, 24, 0.55, "#6fa1ff"),
    (740, 80, 22, 0.45, "#5b8fff"),
    (1130, 80, 14, 0.85, "#87b3ff"),
    (920, 360, 12, 0.80, "#5b8fff"),
    (1050, 540, 16, 0.75, "#6fa1ff"),
    (760, 480, 10, 0.70, "#87b3ff"),
    (1140, 250, 12, 0.70, "#87b3ff"),
)


# ── Hash + plan helpers (identical to the Playwright version) ──────

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

    Starlette's ``StaticFiles`` percent-decodes the URL path before
    the file lookup, so the on-disk filename must use literal
    characters. The only transformation we need is for ``/`` to
    survive as a real path separator — pathlib turns "org/name" into
    a nested ``org/name.png`` for free.
    """
    return name


def _params_hash(params: CardParams) -> str:
    h = hashlib.sha1(usedforsecurity=False)
    for k in _HASH_FIELDS:
        h.update(f"{k} {params.get(k, '') or ''} ".encode())
    return h.hexdigest()


def _fmt_num(n: int | float | None) -> str:
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


def _pick_headline_pt(text: str) -> int:
    """Pick a headline font size (in points) based on length.

    Mirrors the breakpoints used by the Playwright renderer. matplotlib
    works in *points* not pixels, and at our 100 DPI 1 pt ≈ 1.333 px,
    so the integers here are sized to *look like* the CSS px values
    visually. Tuned by eye against the 1200 × 630 box.
    """
    n = len(text)
    if n <= 18:
        return 69  # ~92 px
    if n <= 26:
        return 58  # ~78 px
    if n <= 35:
        return 48  # ~64 px
    if n <= 48:
        return 41  # ~54 px
    return 35  # ~46 px


def _trim_description(desc: str | None, max_len: int = 140) -> str:
    """Collapse whitespace + hard-truncate. Does *not* insert newlines.

    Wrapping into visual lines is a render-time concern (we don't know
    pixel widths at plan time), so the planners only normalise +
    truncate here. ``_wrap_tagline`` handles the actual line break.
    """
    if not desc:
        return ""
    flat = " ".join(desc.split())
    if len(flat) <= max_len:
        return flat
    return flat[: max_len - 1].rstrip() + "…"


def _wrap_tagline(text: str, chars_per_line: int = 70, max_lines: int = 2) -> str:
    """Break ``text`` into at most ``max_lines`` lines that visually fit
    the card's left-column body width (~920 px).

    Word-boundary wrap via :mod:`textwrap`, then keep only the first
    ``max_lines`` segments and append an ellipsis if there's more.
    Sized for the 20-pt tagline at our font + DPI; tighter when the
    chosen font is wider than expected, but a hard cap keeps the
    overflow off the dot scatter on the right side of the card.
    """
    if not text:
        return ""
    lines = textwrap.wrap(text, width=chars_per_line, break_long_words=False)
    if not lines:
        return ""
    if len(lines) > max_lines:
        kept = lines[:max_lines]
        # Trim the last visible line so the ellipsis still fits inside
        # the same character budget.
        tail = kept[-1].rstrip(",.;:- ")
        if len(tail) + 1 > chars_per_line:
            tail = tail[: chars_per_line - 1].rstrip()
        kept[-1] = tail + "…"
        lines = kept
    return "\n".join(lines)


def _plan_job(out_dir: Path, kind: Kind, name: str, params: CardParams) -> RenderJob:
    file = _slug(name)
    out_path = out_dir / kind / f"{file}.png"
    # Nested slugs (e.g. `microsoft/harrier-...`) need their org-named
    # parent directory in place before the renderer writes the PNG.
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
    # Stats are pure registry metadata: task / language / domain counts.
    # We deliberately don't render the model count, which would require
    # the unified results polars frame to be built (~12 s) just to
    # populate one stat.
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
            "typeLabel": "",
            "eyebrow": "Benchmark",
            "title": title,
            "tagline": tagline,
            "stats": stats,
            "headlineSize": str(_pick_headline_pt(title)),
        }
        jobs.append(_plan_job(out_dir, "benchmark", b.name, params))
    return jobs


def plan_tasks(entries: Iterable[TaskMetaSchema], out_dir: Path) -> list[RenderJob]:
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
            "typeLabel": t.type or "",
            "eyebrow": "Task",
            "title": t.name,
            "tagline": tagline,
            "stats": stats,
            "headlineSize": str(_pick_headline_pt(t.name)),
        }
        jobs.append(_plan_job(out_dir, "task", t.name, params))
    return jobs


def plan_models(entries: Iterable[ModelMetaSchema], out_dir: Path) -> list[RenderJob]:
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
            "typeLabel": m.model_type or "",
            "eyebrow": "Model",
            "title": m.name,
            "tagline": _trim_description(flags),
            "stats": stats,
            "headlineSize": str(_pick_headline_pt(m.name)),
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


# ── Renderer ───────────────────────────────────────────────────────


def _pick_font_family() -> list[str]:
    """Return a font fallback chain matching the HTML template's intent.

    The Playwright renderer uses ``Inter`` via Google Fonts. matplotlib
    doesn't have Inter by default, so we pick the closest humanist
    sans-serif we can find and let matplotlib walk its own fallback
    chain. Both Liberation Sans and DejaVu Sans usually ship with
    the matplotlib install and look reasonably close.
    """
    have = {f.name for f in fm.fontManager.ttflist}
    chain = [
        name for name in ("Inter", "Liberation Sans", "DejaVu Sans") if name in have
    ]
    return chain or ["DejaVu Sans"]


_FONT_FAMILY = _pick_font_family()


def _measure_text(ax: Axes, text: str, **text_kwargs: Any) -> tuple[float, float]:
    """Return ``(width_px, height_px)`` for ``text`` in display pixels.

    Drops a throwaway text object off-screen, asks the Agg renderer for
    its bbox, then removes it. Used to size pills around their labels
    instead of guessing pixel widths via character counts — heuristic
    widths were systematically too narrow for non-Inter fallback fonts
    and the labels overflowed their badges.
    """
    # ``Figure.canvas`` is typed as the abstract ``FigureCanvasBase`` in
    # matplotlib's stubs even though we know we constructed an Agg one
    # in ``_render_card``. The ``Agg`` subclass is what exposes
    # ``get_renderer()``.
    canvas: FigureCanvasAgg = ax.figure.canvas  # type: ignore[assignment]
    renderer = canvas.get_renderer()  # type: ignore[no-untyped-call]
    t = ax.text(-9999, -9999, text, **text_kwargs)
    bbox = t.get_window_extent(renderer=renderer)
    t.remove()
    return bbox.width, bbox.height


# Background layers (base color + radial glows + dot scatter + bottom
# stripe) never change between cards, so we render them once into a
# numpy array and blit it into every card. Saves ~80 ms per card.
_BG_CACHE: np.ndarray | None = None

# Brand mark loaded from the same template directory we share with the
# Playwright generator. Read once, ``imshow``'d into every card.
_ICON_CACHE: np.ndarray | None = None
_ICON_PATH = Path(__file__).resolve().parent / "og-template" / "dots-icon.png"


def _icon_array() -> np.ndarray | None:
    """Lazily load + cache the brand mark PNG as an RGBA numpy array.

    Returns ``None`` if the file is missing — the brand row falls back
    to text only, matching how the HTML template's missing-image case
    would render.
    """
    global _ICON_CACHE
    if _ICON_CACHE is not None:
        return _ICON_CACHE
    if not _ICON_PATH.is_file():
        return None
    _ICON_CACHE = imread(str(_ICON_PATH))
    return _ICON_CACHE


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)


def _build_background() -> np.ndarray:
    """Generate the static background layer once and cache it.

    Returns an (H, W, 4) RGBA array in display pixels. Steps:

    1. Fill with ``BG_BASE``.
    2. Apply two radial-gradient glows (top-right + bottom-left) via a
       distance-from-center alpha mask, blended over the base.
    3. Draw the bottom 6-pixel gradient stripe (blue → purple → blue).

    Dots and shadows are drawn per-card by the renderer because they
    sit *over* the foreground text in some layouts (we still cache
    them as patches to add cheaply).
    """
    h, w = CARD_H, CARD_W
    img = np.zeros((h, w, 4), dtype=np.float32)
    base = _hex_to_rgb(BG_BASE)
    img[..., 0] = base[0]
    img[..., 1] = base[1]
    img[..., 2] = base[2]
    img[..., 3] = 1.0

    # Top-right glow — strong, large ellipse.
    yy, xx = np.mgrid[0:h, 0:w]
    glow_color = _hex_to_rgb("#5b8fff")
    d_tr = np.sqrt(((xx - w) / 900) ** 2 + ((yy - 0) / 700) ** 2)
    alpha_tr = np.clip(1 - d_tr, 0, 1) ** 2 * 0.16
    _composite_color(img, glow_color, alpha_tr)

    # Bottom-left glow — softer, slightly offset.
    d_bl = np.sqrt(((xx - 0) / 900) ** 2 + ((yy - h) / 750) ** 2)
    alpha_bl = np.clip(1 - d_bl, 0, 1) ** 2 * 0.08
    _composite_color(img, glow_color, alpha_bl)

    # Bottom stripe — 6 px tall, blue→purple→blue.
    stripe_h = 6
    stripe = np.zeros((stripe_h, w, 3), dtype=np.float32)
    stops = BOTTOM_STRIPE_STOPS
    rgb_stops = [(t, _hex_to_rgb(c)) for t, c in stops]
    for x in range(w):
        u = x / (w - 1)
        # Find the two stops we're between and lerp.
        for (t0, c0), (t1, c1) in zip(rgb_stops, rgb_stops[1:], strict=True):
            if t0 <= u <= t1:
                k = (u - t0) / (t1 - t0)
                stripe[:, x, 0] = c0[0] * (1 - k) + c1[0] * k
                stripe[:, x, 1] = c0[1] * (1 - k) + c1[1] * k
                stripe[:, x, 2] = c0[2] * (1 - k) + c1[2] * k
                break
    img[h - stripe_h : h, :, 0:3] = stripe

    return img


def _composite_color(
    img: np.ndarray, rgb: tuple[float, float, float], alpha: np.ndarray
) -> None:
    """Composite a tinted ``alpha`` mask onto ``img`` in place.

    Source-over: ``out = src * a + dst * (1 - a)`` per channel. ``alpha``
    is the pixel-wise opacity of the tint, ``rgb`` is the tint colour.
    Only the RGB channels are touched — the alpha channel of ``img``
    stays at 1.0.
    """
    inv = 1.0 - alpha
    for c in range(3):
        img[..., c] = rgb[c] * alpha + img[..., c] * inv


def _draw_gradient_text(
    ax: Axes,
    text: str,
    x: float,
    y: float,
    fontsize: int,
    fontweight: int = 900,
    family: list[str] | None = None,
) -> None:
    """Headline text.

    The Playwright/CSS version uses a vertical white→soft-blue gradient
    clipped to the glyph outline (``background-clip: text``). Doing
    that faithfully in matplotlib needs ``TextPath`` glyph extraction +
    a clip-path on an ``imshow`` gradient — meaningful complexity for
    a subtle effect. We keep it simple here and pick the lighter
    headline colour as a solid fill; the result reads as the same
    "soft white headline against dark background" without the
    top-to-bottom fade.
    """
    ax.text(
        x,
        y,
        text,
        fontsize=fontsize,
        fontweight=fontweight,
        color=INK_HEADLINE_TINT,
        family=family or _FONT_FAMILY,
        ha="left",
        va="top",
    )


def _draw_text(
    ax: Axes,
    text: str,
    x: float,
    y: float,
    fontsize: int,
    color: str,
    fontweight: int | str = "normal",
    family: list[str] | None = None,
    ha: str = "left",
    va: str = "top",
    zorder: float = 3,
) -> Text:
    return ax.text(
        x,
        y,
        text,
        fontsize=fontsize,
        fontweight=fontweight,
        color=color,
        family=family or _FONT_FAMILY,
        ha=ha,
        va=va,
        zorder=zorder,
    )


_MplColor = str | tuple[float, float, float] | tuple[float, float, float, float]
"""Color in any form matplotlib's facecolor / edgecolor accept."""


def _draw_pill(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    facecolor: _MplColor,
    edgecolor: _MplColor | None = None,
    zorder: int = 5,
) -> tuple[float, float]:
    """Draw a fully-rounded "pill" rectangle and return its centre.

    Manual construction — ``FancyBboxPatch`` is a poor fit here because
    its ``boxstyle`` params (``pad``, ``rounding_size``) are scaled by
    ``mutation_scale / 72`` instead of treated as data-unit pixels, so
    a 25-px request renders at ~33 px and the patch overflows the
    intended bounding box. Two circles for the end-caps + one rect
    for the body gives us crisp pixel-precise pills with no scaling
    surprises.
    """
    r = h / 2
    # Body rect spans between the two cap centres; the caps fill the
    # left and right ends.
    ax.add_patch(
        Rectangle(
            (x + r, y),
            w - 2 * r,
            h,
            facecolor=facecolor,
            edgecolor="none",
            zorder=zorder,
        )
    )
    for cx in (x + r, x + w - r):
        ax.add_patch(
            Circle(
                (cx, y + r),
                r,
                facecolor=facecolor,
                edgecolor="none",
                zorder=zorder,
            )
        )
    # Border (if any) — same three shapes, no fill, thin stroke.
    if edgecolor:
        ax.add_patch(
            Rectangle(
                (x + r, y),
                w - 2 * r,
                h,
                facecolor="none",
                edgecolor=edgecolor,
                linewidth=1.0,
                zorder=zorder + 0.01,
            )
        )
        for cx in (x + r, x + w - r):
            ax.add_patch(
                Circle(
                    (cx, y + r),
                    r,
                    facecolor="none",
                    edgecolor=edgecolor,
                    linewidth=1.0,
                    zorder=zorder + 0.01,
                )
            )
    return x + w / 2, y + h / 2


def _draw_dots(ax: Axes) -> None:
    for cx, cy, diameter, alpha, color in DOT_SPEC:
        ax.add_patch(
            Circle(
                (cx, cy),
                diameter / 2,
                facecolor=color,
                alpha=alpha,
                edgecolor="none",
                zorder=2,
            )
        )


def _draw_brand(ax: Axes) -> None:
    """Top-left brand row: dots icon + eyebrow + site name.

    Matches the HTML template's geometry — 36 × 36 icon, 14 px gap,
    eyebrow above site name. When the icon file isn't present (e.g.
    in tests using a stripped-down ``scripts/og-template/`` directory)
    the text alone still renders sensibly on the left edge.
    """
    icon = _icon_array()
    text_x = 80
    icon_top = 56
    icon_size = 36
    if icon is not None:
        # Y axis is inverted, so extent's upper Y is icon_top and lower
        # Y is icon_top + icon_size — same convention as the rest of
        # the card's coordinates.
        ax.imshow(
            icon,
            extent=(80, 80 + icon_size, icon_top + icon_size, icon_top),
            zorder=2,
            interpolation="bilinear",
        )
        text_x = 80 + icon_size + 14
    _draw_text(
        ax,
        "MTEB LEADERBOARD",
        text_x,
        icon_top + 4,
        fontsize=10,
        color=INK_BRAND_EYEBROW,
        fontweight="bold",
        ha="left",
        va="top",
    )
    _draw_text(
        ax,
        "mteb-leaderboard.hf.space",
        text_x,
        icon_top + 22,
        fontsize=14,
        color=INK_BRAND_NAME,
        fontweight="bold",
        ha="left",
        va="top",
    )


def _draw_badges(ax: Axes, kind: str, type_label: str) -> None:
    """Top-right stacked pills: kind + optional secondary type.

    Pill widths are measured from the actual text bbox (not estimated
    from character count) so the label always fits inside its pill
    regardless of which font matplotlib falls back to.
    """
    kind_text = (kind or "").upper()
    kind_font: _FontStyle = {"fontsize": 18, "fontweight": 800, "family": _FONT_FAMILY}
    text_w, _ = _measure_text(ax, kind_text, **kind_font)
    pad_x = 28
    kind_h = 50
    kind_w = int(text_w + 2 * pad_x)
    kind_x = CARD_W - 80 - kind_w
    kind_y = 56
    cx, cy = _draw_pill(
        ax, kind_x, kind_y, kind_w, kind_h, facecolor=BADGE_KIND_BG[0], zorder=5
    )
    _draw_text(
        ax,
        kind_text,
        cx,
        cy,
        color=BADGE_KIND_INK,
        ha="center",
        va="center",
        # Sit above the pill (zorder=5) so the label isn't painted over.
        zorder=6,
        **kind_font,
    )

    if type_label:
        type_text = type_label.upper()
        type_font: _FontStyle = {
            "fontsize": 14,
            "fontweight": 700,
            "family": _FONT_FAMILY,
        }
        text_w, _ = _measure_text(ax, type_text, **type_font)
        pad_x = 22
        type_h = 38
        type_w = int(text_w + 2 * pad_x)
        type_x = CARD_W - 80 - type_w
        type_y = kind_y + kind_h + 10
        tcx, tcy = _draw_pill(
            ax,
            type_x,
            type_y,
            type_w,
            type_h,
            facecolor=BADGE_TYPE_BG_RGBA,
            edgecolor=BADGE_TYPE_BORDER_RGBA,
            zorder=5,
        )
        _draw_text(
            ax,
            type_text,
            tcx,
            tcy,
            color=BADGE_TYPE_INK,
            ha="center",
            va="center",
            zorder=6,
            **type_font,
        )


def _draw_stats(ax: Axes, stats: str) -> None:
    """Render the bottom-left stats row.

    Layout: ``num`` on top (39-pt blue, tabular figures), ``LABEL``
    sitting just under its baseline (small uppercase). Items are
    separated by thin 1-px vertical rules, mirroring the HTML
    ``.stat-sep`` element.

    Column advance is the *wider* of the number and the label — using
    only the number's width meant single-char stats like "1" with
    long labels like "LANGUAGES" got squashed into the next column.
    """
    items = [s.split("|", 1) for s in stats.split(",") if s]
    if not items:
        return
    num_font: _FontStyle = {"fontsize": 39, "fontweight": 800, "family": _FONT_FAMILY}
    label_font: _FontStyle = {"fontsize": 12, "fontweight": 700, "family": _FONT_FAMILY}
    # Measure every number + label up front so column widths react to
    # whichever is wider per item. One renderer-bbox call per text
    # token; cheap even for the 4-stat model cards.
    column_widths: list[float] = []
    for label, num in items:
        num_w, _ = _measure_text(ax, num, **num_font)
        label_w, _ = _measure_text(ax, label.upper(), **label_font)
        column_widths.append(max(num_w, label_w))

    x: float = 80
    # Number baseline + label top sit ~10 px apart so the pair reads
    # as a single tight unit, not two separate columns.
    y_num = CARD_H - 88
    y_label = CARD_H - 78
    gap = 32
    sep_h = 56
    for i, ((label, num), col_w) in enumerate(zip(items, column_widths, strict=True)):
        if i > 0:
            # Separator straddles the number's vertical centre.
            ax.add_patch(
                Rectangle(
                    (x, y_num - sep_h + 12),
                    1,
                    sep_h,
                    facecolor=STAT_SEP_RGBA,
                    edgecolor="none",
                    zorder=4,
                )
            )
            x += gap
        _draw_text(
            ax,
            num,
            x,
            y_num,
            color=STAT_NUM_COLOR,
            ha="left",
            va="bottom",
            **num_font,
        )
        _draw_text(
            ax,
            label.upper(),
            x,
            y_label,
            color=STAT_LABEL_COLOR,
            ha="left",
            va="top",
            **label_font,
        )
        x += col_w + gap


def _render_card(params: CardParams) -> bytes:
    """Render a single card to PNG bytes."""
    global _BG_CACHE
    if _BG_CACHE is None:
        _BG_CACHE = _build_background()

    fig = Figure(figsize=(CARD_W / DPI, CARD_H / DPI), dpi=DPI)
    canvas = FigureCanvasAgg(fig)
    ax: Axes = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0, CARD_W)
    ax.set_ylim(0, CARD_H)
    ax.invert_yaxis()  # y=0 at top, matches CSS / DOM convention
    ax.set_axis_off()

    # Background blit
    ax.imshow(
        _BG_CACHE, extent=(0, CARD_W, CARD_H, 0), zorder=0, interpolation="bilinear"
    )

    # Dots overlay (drawn over background, under everything else)
    _draw_dots(ax)

    # Brand
    _draw_brand(ax)

    # Headline (gradient text)
    headline_pt = int(params.get("headlineSize") or 58)
    _draw_gradient_text(ax, params["title"], 80, 250, headline_pt)

    # Tagline — wrap to ≤2 lines so a long description doesn't run off
    # the right side of the card into the dot scatter.
    tagline = _wrap_tagline(params.get("tagline") or "")
    if tagline:
        _draw_text(
            ax,
            tagline,
            80,
            340,
            fontsize=20,
            color=INK_TAGLINE,
            fontweight=500,
            ha="left",
            va="top",
        )

    # Stats row
    _draw_stats(ax, params.get("stats") or "")

    # Top-right kind + type badges
    _draw_badges(ax, params.get("kind") or "", params.get("typeLabel") or "")

    buf = io.BytesIO()
    canvas.print_png(buf)  # type: ignore[no-untyped-call]
    return buf.getvalue()


# ── Catalogue load + entry point ──────────────────────────────────


def _load_catalogue(
    only: Sequence[str],
) -> tuple[list[BenchmarkSchema], list[TaskMetaSchema], list[ModelMetaSchema]]:
    """Walk the mteb registry and return schema objects per kind.

    Pure metadata only — no ``num_models`` overlay and no
    ``warmup_blocking`` call. Computing real per-entity model counts
    would require the unified polars results frame to be built
    (~12 s); we skip it entirely so the generator runs even when the
    results dataset hasn't been downloaded.
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
        tasks = [task_to_meta_schema(t) for t in mteb.get_tasks()]
    if "models" in only:
        models = [model_meta_to_schema(m) for m in mteb.get_model_metas()]
    return benches, tasks, models


def _render_one(job: RenderJob) -> Path:
    """Render a single job and persist its PNG + hash sidecar.

    Top-level (not nested) so :mod:`multiprocessing` workers can pickle
    it. Each worker process gets its own ``_BG_CACHE`` / ``_ICON_CACHE``
    because module globals reset on import — the cost is one extra
    background-array build per worker, not per job.
    """
    png = _render_card(job["params"])
    out = job["out"]
    out.write_bytes(png)
    job["hash_path"].write_text(job["hash"])
    return out


def _split_evenly(items: list[RenderJob], n_chunks: int) -> list[list[RenderJob]]:
    """Slice ``items`` into ``n_chunks`` near-equal-length sublists."""
    if n_chunks <= 1 or len(items) <= 1:
        return [items]
    k, r = divmod(len(items), n_chunks)
    chunks: list[list[RenderJob]] = []
    cursor = 0
    for i in range(n_chunks):
        size = k + (1 if i < r else 0)
        chunks.append(items[cursor : cursor + size])
        cursor += size
    return [c for c in chunks if c]


def _render_chunk(jobs_chunk: list[RenderJob]) -> int:
    """Render every job in a worker-private chunk. Returns the count.

    Lives at module scope so the spawn-mode pool can find it after
    re-importing this script in the worker.
    """
    for job in jobs_chunk:
        _render_one(job)
    return len(jobs_chunk)


def _execute_jobs(
    jobs: list[RenderJob],
    label: str,
    workers: int,
    force: bool,
) -> PoolResult:
    """Render every job, optionally across a multiprocessing pool.

    Skip checks run in the main process — they're filesystem-only and
    cheaper than the IPC roundtrip would be. The remaining jobs are
    farmed out to the pool; with ``workers <= 1`` we stay in-process
    (no spawn overhead for small batches).
    """
    if not jobs:
        return PoolResult(rendered=0, skipped=0)
    to_render = [
        j for j in jobs if not _should_skip(j["out"], j["hash_path"], j["hash"], force)
    ]
    skipped = len(jobs) - len(to_render)
    rendered = 0
    if to_render:
        if workers <= 1 or len(to_render) < 8:
            # Single-process path. Below ~8 jobs the multiprocessing
            # spawn cost outweighs any savings — every render is ~50 ms
            # but every worker boot is ~500–1500 ms.
            for j in to_render:
                _render_one(j)
                rendered += 1
                if rendered % 50 == 0:
                    logger.info("[%s] %s rendered (in-process)", label, rendered)
        else:
            chunks = _split_evenly(to_render, workers)
            # ``spawn`` is the safe default on macOS (Python 3.14 dropped
            # fork as default) and matches Docker/Linux behaviour.
            ctx = mp.get_context("spawn")
            with ctx.Pool(min(workers, len(chunks))) as pool:
                for done in pool.imap_unordered(_render_chunk, chunks):
                    rendered += done
                    logger.info(
                        "[%s] +%s rendered (pool); %s/%s done",
                        label,
                        done,
                        rendered,
                        len(to_render),
                    )
    logger.info(
        "[og-mpl] %s: %s rendered, %s skipped (%s total)",
        label,
        rendered,
        skipped,
        len(jobs),
    )
    return PoolResult(rendered=rendered, skipped=skipped)


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind in ("benchmark", "task", "model"):
        (out_dir / kind).mkdir(parents=True, exist_ok=True)

    only = tuple(args.only.split(","))
    logger.info(
        "[og-mpl] target: %s  font chain: %s%s",
        out_dir,
        ", ".join(_FONT_FAMILY),
        "  (force re-render)" if args.force else "",
    )

    benches, tasks, models = _load_catalogue(only)
    logger.info(
        "[og-mpl] catalogue: %s benchmarks, %s tasks, %s models",
        len(benches),
        len(tasks),
        len(models),
    )

    bench_jobs = plan_benchmarks(benches, out_dir)
    task_jobs = plan_tasks(tasks, out_dir)
    model_jobs = plan_models(models, out_dir)

    started = time.time()
    rendered = 0
    skipped = 0

    for kind_label, jobs in (
        ("benchmarks", bench_jobs),
        ("tasks", task_jobs),
        ("models", model_jobs),
    ):
        result = _execute_jobs(jobs, kind_label, args.workers, args.force)
        rendered += result["rendered"]
        skipped += result["skipped"]

    elapsed = time.time() - started
    total = len(bench_jobs) + len(task_jobs) + len(model_jobs)
    logger.info(
        "[og-mpl] done: %s rendered, %s skipped of %s total in %.1fs",
        rendered,
        skipped,
        total,
        elapsed,
    )

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "renderer": "matplotlib",
        "counts": {
            "benchmark": len(bench_jobs),
            "task": len(task_jobs),
            "model": len(model_jobs),
        },
        "last_run": {
            "rendered": rendered,
            "skipped": skipped,
            "seconds": round(elapsed, 1),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render OG hero cards via matplotlib (no headless browser)."
    )
    p.add_argument(
        "--out",
        default=os.environ.get("MTEB_API_OG_DIR", "/data/og"),
        help="Output directory for the generated PNG files.",
    )
    p.add_argument(
        "--only",
        default="benchmarks,tasks,models",
        help="Comma-separated subset of {benchmarks,tasks,models} to (re)generate.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=int(
            os.environ.get(
                "MTEB_API_OG_WORKERS", str(max(1, (os.cpu_count() or 4) // 2))
            )
        ),
        help=(
            "Process-pool size for parallel rendering. Defaults to half the CPU "
            "count. Set to 1 to stay in-process (faster for small batches; the "
            "pool's spawn overhead dominates below ~8 jobs)."
        ),
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
    return run(_parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
