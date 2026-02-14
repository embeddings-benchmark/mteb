# Installation

Installation is as simple as:

=== "pip"
    ```bash
    pip install mteb
    ```

=== "uv"
    ```bash
    uv add mteb
    ```

## Model Specific Installations

If you want to run certain models implemented within mteb you will often need some additional dependencies. These can be installed using:

=== "pip"
    ```bash
    pip install mteb[cohere]
    ```

=== "uv"
    ```bash
    uv add "mteb[cohere]"
    ```

If a specific model requires a dependency it will raise an error with the recommended installation. To see full list of available models you can look at the [models overview](./overview/available_models/text.md).

## Audio Tasks

If you want to run audio tasks, install the audio dependencies:

=== "pip"
    ```bash
    pip install mteb[audio]
    ```

=== "uv"
    ```bash
    uv add "mteb[audio]"
    ```

### Additional Requirements for `datasets>=4`

If you are using `datasets>=4`, you will need to:

1. **Install FFmpeg**: The `datasets` library version 4+ uses `torchcodec` for audio processing, which requires FFmpeg to be installed on your system.

    === "macOS"
        ```bash
        brew install ffmpeg
        ```

    === "Ubuntu/Debian"
        ```bash
        sudo apt-get install ffmpeg
        ```

    === "Windows"
        Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to your PATH.

2. **Use `transformers>=4.57.6`**: Due to compatibility issues with `datasets>=4`, you need a recent version of transformers:
    ```bash
    pip install "transformers>=4.57.6"
    ```

If you are using `datasets<4`, no additional requirements are needed beyond the `mteb[audio]` installation.

## Migrating to uv (for Contributors)

If you're a contributor currently using pip, here's how to migrate to uv for faster dependency management:

### Why uv?
- **Faster**: 10-100x faster dependency resolution
- **Reliable**: Deterministic builds with uv.lock
- **Simpler**: One tool for virtual environments and packages

### Migration Steps
1. **Install uv**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Replace your workflow**:
   - `pip install -e .` → `uv sync`
   - `pip install mteb[extra]` → `uv sync --extra extra`
   - `python -m pytest` → `uv run pytest`

### Development Groups
For contributors, uv provides organized dependency groups:

- `uv sync --group test` - Install test dependencies
- `uv sync --group docs` - Install documentation dependencies
- `uv sync --group typing` - Install type checking dependencies
- `uv sync --group lint` - Install linting dependencies
- `uv sync --group dev` - Install all development dependencies (recommended)
