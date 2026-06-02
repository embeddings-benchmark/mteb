# Hugging Face Spaces Dockerfile for the mteb FastAPI service.
#
# Clones embeddings-benchmark/mteb @ api, installs the project with its
# [api] extra (fastapi + uvicorn), and serves it on :7860 as the
# non-root `user` (UID 1000) that Spaces expects.
#
# Everything is hardcoded — Spaces does not pass build args. Edit and
# rebuild to change repo or branch.

FROM python:3.12-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl build-essential \
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

# Frontend Space origin allowed through CORS. Add others comma-separated
# if you front this API with a different host.
ENV MTEB_API_CORS_ORIGINS="https://embeddings-benchmark-leaderboard-frontend.hf.space,http://localhost:5173,http://localhost:4173"

USER user
WORKDIR /home/user

RUN git clone --depth=1 --branch api \
        https://github.com/embeddings-benchmark/mteb.git app
WORKDIR /home/user/app

# Branches that define an [api] extra get fastapi + uvicorn from it; on
# older branches the explicit pins are the fallback.
RUN pip install --user ".[api]" \
 || pip install --user . "fastapi>=0.110" "uvicorn[standard]>=0.27"

EXPOSE 7860

CMD ["uvicorn", "mteb.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
