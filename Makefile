install:
	@echo "--- 🚀 Installing project dependencies ---"
	pip install -e ".[dev,docs,image]"
	pre-commit install

install-for-tests:
	@echo "--- 🚀 Installing project dependencies for test ---"
	@echo "This ensures that the project is not installed in editable mode"
	pip install ".[dev,bm25s,pylate,image]"

lint:
	@echo "--- 🧹 Running linters ---"
	ruff format . 			# running ruff formatting
	ruff check . --fix --exit-non-zero-on-fix  	# running ruff linting # --exit-non-zero-on-fix is used for the pre-commit hook to work

lint-check:
	@echo "--- 🧹 Check is project is linted ---"
	# Required for CI to work, otherwise it will just pass
	ruff format . --check						    # running ruff formatting
	ruff check **/*.py 						        # running ruff linting

test:
	@echo "--- 🧪 Running tests ---"
	pytest -n auto -m "not test_datasets"


test-with-coverage:
	@echo "--- 🧪 Running tests with coverage ---"
	pytest -n auto --cov-report=term-missing --cov-config=pyproject.toml --cov=mteb

pr:
	@echo "--- 🚀 Running requirements for a PR ---"
	make lint
	make test


build-docs:
	@echo "--- 📚 Building documentation ---"
	# since we do not have a documentation site, this just build tables for the .md files
	python docs/create_tasks_table.py
	python docs/create_benchmarks_table.py

serve-docs:
	@echo "--- 📚 Serving documentation ---"
	python -m mkdocs serve


model-load-test:
	@echo "--- 🚀 Running model load test ---"
	pip install ".[dev, pylate,gritlm,xformers,model2vec]"
	python scripts/extract_model_names.py $(BASE_BRANCH) --return_one_model_name_per_file
	python tests/test_models/model_loading.py --model_name_file scripts/model_names.txt


dataset-load-test:
	@echo "--- 🚀 Running dataset load test ---"
	pytest -n auto -m test_datasets


run-leaderboard:
	@echo "--- 🚀 Running leaderboard locally ---"
	python -m mteb.leaderboard.app

format-citations:
	@echo "--- 🧹 Formatting citations ---"
	python scripts/format_citations.py benchmarks
	python scripts/format_citations.py tasks


.PHONY: check
check: ## Run code quality tools.
	@echo "--- 🧹 Running code quality tools ---"
	@pre-commit run -a
