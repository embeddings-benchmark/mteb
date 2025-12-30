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
