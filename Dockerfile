# syntax=docker/dockerfile:1.7
#
# Multi-stage Dockerfile for the mteb FastAPI service.
#
# Three stages share the heavy "clone + pip install + HF dataset
# warmup" work via a common base image:
#
#   base        — python:3.12-bookworm + git + a non-root user + mteb
#                 cloned and installed with the [api] extra + the
#                 mteb/results parquet cache pre-warmed.
#   og-builder  — extends `base` with Chromium runtime libs + the
#                 [og] extra (Playwright). Renders one OG hero PNG per
#                 benchmark / task / model into /og-cache, then exits.
#                 Nothing downstream of this stage ships.
#   runtime     — extends `base`, copies the rendered PNG files out of
#                 og-builder. Stays small: no Playwright, no Chromium,
#                 no Node. FastAPI serves the cached PNG files at /og.
#
# Result: the deployed image is the lean base + ~50 MB of pre-rendered
# PNG files, not the ~700 MB hit of baking Chromium into runtime. The repo
# is cloned once, [api] is installed once, the HF dataset cache is
# downloaded once — both downstream stages reuse the same layers.

ARG MTEB_BRANCH=api
ARG MTEB_REPO=https://github.com/embeddings-benchmark/mteb.git


# ─── Stage: base ────────────────────────────────────────────────────
FROM python:3.12-bookworm AS base

ARG MTEB_BRANCH
ARG MTEB_REPO

# Just enough to clone + build pip wheels. Chromium libs are added in
# the og-builder stage so the runtime layer doesn't carry them.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl build-essential ca-certificates \
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
WORKDIR /home/user

# Bust the clone cache whenever the API branch advances upstream. The
# ADD response (latest commit SHA) changes per push, so Docker can no
# longer reuse a stale checkout when you rebuild.
ADD --chown=user:user https://api.github.com/repos/embeddings-benchmark/mteb/commits/${MTEB_BRANCH} /tmp/.git-sha
RUN git clone --depth=1 --branch ${MTEB_BRANCH} ${MTEB_REPO} app
WORKDIR /home/user/app

# Branches that define an [api] extra get fastapi + uvicorn from it; on
# older branches the explicit pins are the fallback. The PyTorch CPU
# wheel index is added alongside default PyPI so the torch / torchvision
# / torchaudio transitive deps resolve to ``2.x.x+cpu`` (~200 MB total)
# instead of the default CUDA wheels (~2 GB across torch +
# nvidia-cu* deps). PEP 440 ranks the ``+cpu`` local version above the
# plain release, so pip picks it without an explicit version pin.
RUN pip install --user --extra-index-url https://download.pytorch.org/whl/cpu ".[api]" \
 || pip install --user --extra-index-url https://download.pytorch.org/whl/cpu . \
        "fastapi>=0.110" "uvicorn[standard]>=0.27"

# Pre-warm the HF dataset cache from mteb/results so the OG builder
# (which calls warmup_blocking()) and the runtime first request both
# skip the multi-minute cold clone. `|| true` keeps the build alive
# when the dataset is still being populated upstream — the API falls
# back to the GitHub clone on first request when the snapshot is empty.
RUN hf download mteb/results --repo-type dataset || true


# ─── Stage: og-builder ──────────────────────────────────────────────
FROM base AS og-builder

ENV PLAYWRIGHT_BROWSERS_PATH=/home/user/.cache/playwright \
    OG_DIR=/og-cache

# Chromium runtime libs Playwright drives. Listing them explicitly
# instead of running `playwright install --with-deps` keeps the layer
# cacheable across rebuilds (the deps list rarely changes).
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

# Add Playwright on top of the [api] install already in base. We
# install the bare package instead of re-resolving ``.[og]`` because
# pip will skip extras when ``.`` is already "satisfied" from the
# base stage's install, which silently drops playwright. Pinning the
# bare package guarantees it lands.
RUN pip install --user "playwright>=1.49.0"
# Invoke via ``python -m playwright`` instead of the ``playwright``
# shim — the shim lives at /home/user/.local/bin/playwright and
# ``$PATH`` does include that directory, but Docker's ``RUN`` shells
# sometimes resolve PATH before the pip layer's new bin entry is
# visible. ``-m`` skips the shim entirely and is the recommended
# invocation in Playwright's own Docker docs.
RUN python -m playwright install chromium

# Render every per-entity OG card. The script imports the mteb
# registry directly (no HTTP, no uvicorn boot) and loads the template
# from a file:// URL under scripts/og-template/, so this is a single
# self-contained Python invocation. Output lands in /og-cache and the
# runtime stage copies it out.
RUN python scripts/generate_og_images.py --out=/og-cache


# ─── Stage: runtime ─────────────────────────────────────────────────
FROM base AS runtime

ENV OG_DIR=/data/og

# CORS deliberately left to its default (``*``) — every endpoint here
# is public read-only, and the OG hero images are meant to be embedded
# cross-origin by share-card validators and chat clients. Lock down
# via ``CORS_ORIGINS`` only if a specific deployment needs it.

# Mount point for the rendered OG hero PNG files. /data is the conventional
# Spaces persistent-volume mount: if Spaces mounts an empty volume over
# /data at runtime, the served /og 404s until someone re-runs the
# generator. With no mount, the baked-in cache wins.
USER root
RUN mkdir -p /data/og && chown -R user:user /data
COPY --from=og-builder --chown=user:user /og-cache /data/og
USER user

EXPOSE 7860

CMD ["uvicorn", "mteb.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
