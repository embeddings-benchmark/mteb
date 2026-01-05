FROM python:3.12-bookworm

RUN apt update && apt install -y git make curl
RUN useradd -m -u 1000 user
ENV PATH="/home/user/.local/bin:$PATH"

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy the current directory contents into the container
COPY --chown=user:user . /mteb

USER user
WORKDIR /mteb

# Use uv to install dependencies with leaderboard extras
RUN uv sync --extra leaderboard

# ENV XDG_CACHE_HOME=/home/user/.cache
ENV GRADIO_SERVER_NAME="0.0.0.0"
EXPOSE 7860

CMD ["make", "run-leaderboard"]
