# syntax=docker/dockerfile:1.7
#
# Multi-stage Dockerfile for the mteb FastAPI service.
#
# Stages split so that layer caching keeps the stable work (pip
# install, Chromium) cached while the data-dependent work (HF dataset
# warmup, per-benchmark frame split, OG rendering) is re-run on every
# daily refresh via the DATA_REFRESH build arg:
#
#   base        — python:3.12-bookworm + a non-root user + the local
#                 checkout installed with the [api] extra.
#   og-deps     — extends `base` with Chromium runtime libs +
#                 Playwright. Depends only on code, so it stays cached
#                 across daily data refreshes.
#   data        — extends `base`; pre-warms the mteb/results parquet
#                 cache and persists the per-benchmark split to
#                 ``~/.cache/mteb/leaderboard/`` so the runtime first
#                 request skips ~30s of cold work. DATA_REFRESH busts
#                 the cache of these layers daily.
#   og-builder  — extends `og-deps`, copies the warmed caches from
#                 `data`, renders one OG hero PNG per benchmark / task
#                 / model into /og-cache, then exits. Nothing
#                 downstream of this stage ships.
#   runtime     — extends `data`, copies the rendered PNG files out of
#                 og-builder. Stays small: no Playwright, no Chromium,
#                 no Node. FastAPI serves the cached PNG files at /og.


# ─── Stage: base ────────────────────────────────────────────────────
FROM python:3.12-bookworm AS base

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl build-essential ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -m -u 1000 user

ENV PATH="/home/user/.local/bin:$PATH" \
    HOME=/home/user \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/user/.cache/huggingface \
    XDG_CACHE_HOME=/home/user/.cache

USER user
WORKDIR /home/user/app

COPY --chown=user:user . /home/user/app

RUN pip install --user --extra-index-url https://download.pytorch.org/whl/cpu ".[api]"


# ─── Stage: og-deps ─────────────────────────────────────────────────
FROM base AS og-deps

ENV PLAYWRIGHT_BROWSERS_PATH=/home/user/.cache/playwright \
    OG_DIR=/og-cache

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
        libxkbcommon0 libatspi2.0-0 libxcomposite1 libxdamage1 \
        libxfixes3 libxrandr2 libgbm1 libdrm2 libasound2 \
        fonts-noto-color-emoji fonts-liberation \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /og-cache && chown user:user /og-cache
USER user

RUN pip install --user "playwright>=1.49.0"
RUN python -m playwright install chromium


# ─── Stage: data ────────────────────────────────────────────────────
FROM base AS data

# Set by CI to the date of the run. Changing it invalidates the layer
# cache from here on, so the daily rebuild re-downloads mteb/results
# and re-splits the frames even when the code (and therefore every
# earlier layer) is unchanged.
ARG DATA_REFRESH=0

# Pre-warm the HF dataset cache from mteb/results so the OG builder
# (which calls warmup_blocking()) and the runtime first request both
# skip the multi-minute cold clone.
RUN echo "data refresh: ${DATA_REFRESH}" \
 && hf download mteb/results --repo-type dataset || true

# Pre-bake the per-benchmark leaderboard frames into the image so the
# runtime skips the ~30s cold (HF download + 72-way split) on first
# request. ``_load_per_benchmark_frames`` reads from the local HF cache
# warmed by the previous RUN, does the split, and persists the result
# to ``$XDG_CACHE_HOME/mteb/leaderboard/`` (~370 MB across 71 parquets
# + a manifest). The runtime stage reads straight from those files —
# warm start drops from ~40s to ~5s.
RUN python -c "from mteb.api.frames import _load_per_benchmark_frames; _load_per_benchmark_frames()" || true


# ─── Stage: og-builder ──────────────────────────────────────────────
FROM og-deps AS og-builder

# Merge the warmed HF + leaderboard caches from the data stage next to
# the Playwright browser cache already present in og-deps.
COPY --from=data --chown=user:user /home/user/.cache /home/user/.cache

RUN python scripts/generate_og_images.py --out=/og-cache


# ─── Stage: runtime ─────────────────────────────────────────────────
FROM data AS runtime

ENV OG_DIR=/data/og

USER root
RUN mkdir -p /data/og && chown -R user:user /data
COPY --from=og-builder --chown=user:user /og-cache /data/og
USER user

EXPOSE 7860

CMD ["uvicorn", "mteb.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
