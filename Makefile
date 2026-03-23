install:
	@echo "--- 🚀 Installing project dependencies ---"
	uv sync --extra image --group dev

install-for-tests:
	@echo "--- 🚀 Installing project dependencies for test ---"
	@echo "This ensures that the project is not installed in editable mode"
	uv sync --extra bm25s --extra image --extra audio --extra leaderboard --extra faiss-cpu --group dev

lint:
	@echo "--- 🧹 Running linters ---"
	uv run --no-sync ruff format . 			# running ruff formatting
	uv run --no-sync ruff check . --fix --exit-non-zero-on-fix  	# running ruff linting # --exit-non-zero-on-fix is used for the pre-commit hook to work
	uv run --no-sync typos

lint-check:
	@echo "--- 🧹 Check is project is linted ---"
	# Required for CI to work, otherwise it will just pass
	uv run --no-sync ruff format . --check
	uv run --no-sync ruff check .
	uv run --no-sync typos --diff

test:
	@echo "--- 🧪 Running tests ---"
	uv run --no-sync --group test pytest -n auto -m "not (test_datasets or leaderboard_stability)"


test-with-coverage:
	@echo "--- 🧪 Running tests with coverage ---"
	uv run --no-sync --group test pytest -n auto --cov-report=term-missing --cov-config=pyproject.toml --cov=mteb

pr:
	@echo "--- 🚀 Running requirements for a PR ---"
	make lint
	make test

build-docs: build-docs-overview
	@echo "--- 📚 Building documentation ---"
	uv run --no-sync --group docs zensical build --clean


build-docs-overview:
	@echo "--- 📚 Building documentation overview ---"
	uv run --no-sync --group docs python docs/overview/create_available_tasks.py
	uv run --no-sync --group docs python docs/overview/create_available_models.py
	uv run --no-sync --group docs python docs/overview/create_available_benchmarks.py


serve-docs:
	@echo "--- 📚 Serving documentation ---"
	uv run --no-sync --group docs zensical serve


model-load-test:
	@echo "--- 🚀 Running model load test ---"
	uv sync --extra pylate --group dev
	uv run --no-sync python scripts/extract_model_names.py $(BASE_BRANCH) --return_one_model_name_per_file
	uv run --no-sync python tests/test_models/model_loading.py --model_name_file scripts/model_names.txt


dataset-load-test:
	@echo "--- 🚀 Running dataset load test ---"
	uv run --no-sync --group test pytest -m test_datasets

dataset-load-test-pr:
	@echo "--- 🚀 Running dataset load test for PR ---"
	eval "$$(uv run --no-sync python -m scripts.extract_datasets $(BASE_BRANCH))" && uv run --no-sync --group test pytest -m test_datasets

leaderboard-build-test:
	@echo "--- 🚀 Running leaderboard build test ---"
	uv run --group test --extra leaderboard pytest -n auto -m leaderboard_stability

leaderboard-test-all:
	@echo "--- 🧪 Running all leaderboard tests ---"
	uv run --group test --extra leaderboard pytest tests/test_leaderboard/ -v

run-leaderboard:
	@echo "--- 🚀 Running leaderboard locally ---"
	uv run --extra leaderboard python -m mteb leaderboard

format-citations:
	@echo "--- 🧹 Formatting citations ---"
	uv run --no-sync python scripts/format_citations.py benchmarks
	uv run --no-sync python scripts/format_citations.py tasks


.PHONY: check
check: ## Run code quality tools.
	@echo "--- 🧹 Running code quality tools ---"
	@uv run --no-sync pre-commit run -a

.PHONY: typecheck
typecheck:
	@echo "--- 🔍 Running type checks ---"
	uv run --no-sync mypy mteb
