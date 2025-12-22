# Repository Guidelines

## Project Structure & Module Organization

The MTEB (Massive Text Embedding Benchmark) repository is organized as follows:

- `mteb/` - Core source code containing:
  - `abstasks/` - Abstract task definitions
  - `tasks/` - Task implementations (19+ subdirectories for different task types)
  - `models/` - Model wrappers and implementations
  - `benchmarks/` - Benchmark definitions
  - `cli/` - Command-line interface tools
  - `results/` - Result handling and storage
  - `evaluate.py` - Main evaluation logic
  - `get_tasks.py` - Task registry and retrieval
- `tests/` - Test suite organized by module (test_abstasks/, test_benchmarks/, etc.)
- `docs/` - Documentation (API docs, contributing guides, usage examples)

## Build, Test, and Development Commands

**Installation:**

`make install` - Install package in editable mode with dev dependencies + pre-commit hooks

**Testing:**

`make test` - Run tests in parallel (excludes test_datasets and leaderboard_stability markers)

`make test-with-coverage` - Run tests with coverage reporting

`pytest tests/test_cli.py -v` - Run specific test file with verbose output

`pytest -k test_name` - Run tests matching pattern

`pytest --durations=10` - Show 10 slowest tests

**Linting & Formatting:**

`make lint` - Format with ruff, fix linting issues, check typos (for local development)

`make lint-check` - Check formatting and linting without modifying files (used in CI)

**Pre-PR Check:**

`make pr` - Runs linting + tests (comprehensive check before submitting PR)

## Coding Style & Naming Conventions

**Formatter & Linter:**
- Uses `ruff` for both formatting and linting
- Uses `typos` for spell-checking
- Pre-commit hooks automatically enforce style on commit

**Style Guidelines:**
- Follow existing patterns in the codebase
- Use type hints throughout
- Keep changes minimal and focused
- Private module-level constants use `_UPPER_CASE` naming
- When adding `# noqa` comments, use specific codes (e.g., `# noqa: N806`)

**Import Conventions:**
- MTEB uses lazy loading via `__getattr__` in `mteb/__init__.py`
- Always use `from mteb import X` instead of `from mteb.X import Y` for top-level exports
- Keep heavy imports (torch, transformers) inside functions when possible to minimize startup time

## Testing Guidelines

**Framework:**
- Uses `pytest` with parallel execution (`pytest-xdist`)
- Test coverage tracked via `pytest-cov`

**Test Organization:**
- Test files mirror source structure (e.g., `tests/test_cli.py` tests `mteb/cli/`)
- Use markers for expensive tests: `@pytest.mark.test_datasets`, `@pytest.mark.leaderboard_stability`
- Mock models available in `tests/mock_models.py` and `tests/mock_tasks.py`

**Running Tests:**
- Test in Python 3.10 (lowest supported version per pyproject.toml) to ensure compatibility
- Use `pytest -n auto` for parallel execution
- Exclude slow markers in regular development: `-m "not (test_datasets or leaderboard_stability)"`

## Commit & Pull Request Guidelines

**Commit Message Convention (Semantic Versioning):**

MTEB uses semantic versioning with automatic releases. Use these prefixes to trigger version bumps:

- `fix:` - Bug fixes (triggers PATCH version bump)
- `model:` - New model additions (triggers MINOR version bump)
- `dataset:` - New datasets/benchmarks (triggers MINOR version bump)
- `feat:` - New features (triggers MINOR version bump)
- `breaking:` - Breaking changes (triggers MAJOR version bump)

Other prefixes (no version bump):
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `chore:` - Build/tooling changes
- `test:` - Test additions/modifications

**Commit Message Format:**

Line 1: prefix: short description

Line 3+: optional detailed explanation

**Pull Request Requirements:**
- Run `make pr` before submitting (linting + tests must pass)
- Link related issues in PR description
- Ensure all pre-commit hooks pass
- Follow the PR template in `.github/pull_request_template.md`
- CI must pass (includes linting check and full test suite)

## Architecture & Performance Notes

**Lazy Loading:**
- `mteb/__init__.py` uses `__getattr__` to defer expensive imports
- Heavy dependencies (torch, transformers) should be imported inside functions when possible
- This pattern reduces CLI startup time from ~4.4s to ~2.0s

**Module Imports:**
- A `_ModuleWrapper` class handles module/function name shadowing
- When adding new top-level exports, update the `_LAZY_MODULES` mapping in `mteb/__init__.py`
- Use `importlib.import_module()` with `# noqa: N806` for accessing private registries

**Testing Performance:**
- CLI tests can be slow due to subprocess spawning and module imports
- Use mock models for unit tests instead of downloading real models
- Avoid testing unrelated functionality when fixing specific issues
