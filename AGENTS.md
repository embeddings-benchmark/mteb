# Repository Guidelines

## Project Structure & Module Organization

The MTEB repository is organized as follows:

- `mteb/` - Main package containing core functionality
  - `abstasks/` - Abstract task definitions
  - `tasks/` - Concrete task implementations (18+ task categories)
  - `models/` - Model wrappers and interfaces
  - `benchmarks/` - Benchmark configurations
  - `evaluate.py` - Core evaluation logic
- `tests/` - Comprehensive test suite with mock tasks and models
- `scripts/` - Utility scripts for data processing and analysis
- `docs/` - Documentation source files

## Build, Test, and Development Commands

Key commands (run from project root):

- `make install` - Install project with dev dependencies and pre-commit hooks
- `make test` - Run test suite (excludes dataset and stability tests)
- `make test-with-coverage` - Run tests with coverage reporting
- `make lint` - Format and lint code (auto-fixes issues)
- `make lint-check` - Check linting without modifying files
- `pip install -e ".[image]"` - Install with image support

## Coding Style & Naming Conventions

- **Formatter**: Ruff (configured in `pyproject.toml`)
- **Linter**: Ruff with automatic fixes
- **Type checking**: Use type hints for all public functions
- **Naming**: Snake_case for functions/variables, PascalCase for classes
- **Indentation**: 4 spaces (Python standard)
- Run `typos` for spell checking before commits

## Testing Guidelines

- **Framework**: pytest with parallel execution (`pytest -n auto`)
- **Test organization**: Mirror source structure in `tests/`
- **Mock data**: Use `mock_tasks.py` and `mock_models.py` for fixtures
- **Test naming**: `test_<module_name>.py` files, `test_<functionality>()` functions
- **Markers**: Use `@pytest.mark` for categorizing tests (e.g., `test_datasets`, `leaderboard_stability`)
- Coverage reports available via `make test-with-coverage`

## Commit & Pull Request Guidelines

Based on project history:

- **Commit format**: `<type>: <description>` (e.g., `fix: add typecheck`, `feat: optimize validate filter`)
- **Types**: `fix`, `feat`, `docs`, `test`, `refactor`, `perf`
- **PR requirements**: Link issues, run tests locally, ensure linting passes
- **Version updates**: Semantic versioning in dedicated commits (e.g., `2.5.4`)
- Pre-commit hooks automatically run formatting and linting checks

## Dependencies & Environment

- **Python**: 3.10-3.14 supported
- **Core deps**: sentence-transformers, torch, datasets, scikit-learn
- **Optional**: Install extras via `pip install mteb[image,faiss-cpu,bm25s]`
- Virtual environment recommended for development
