install:
	@echo "--- 🚀 Installing project dependencies ---"
	pip install -e ".[image]" --group dev
	pre-commit install

install-for-tests:
	@echo "--- 🚀 Installing project dependencies for test ---"
	@echo "This ensures that the project is not installed in editable mode"
	pip install ".[bm25s,pylate,image,codecarbon,faiss-cpu]" --group dev

lint:
	@echo "--- 🧹 Running linters ---"
	ruff format . 			# running ruff formatting
	ruff check . --fix --exit-non-zero-on-fix  	# running ruff linting # --exit-non-zero-on-fix is used for the pre-commit hook to work
	typos

lint-check:
	@echo "--- 🧹 Check is project is linted ---"
	# Required for CI to work, otherwise it will just pass
	ruff format . --check						    # running ruff formatting
	ruff check .    						        # running ruff linting
	typos --diff

test:
	@echo "--- 🧪 Running tests ---"
	pytest -n auto -m "not (test_datasets or leaderboard_stability)"


test-with-coverage:
	@echo "--- 🧪 Running tests with coverage ---"
	pytest -n auto --cov-report=term-missing --cov-config=pyproject.toml --cov=mteb

pr:
	@echo "--- 🚀 Running requirements for a PR ---"
	make lint
	make test

build-docs:
	@echo "--- 📚 Building documentation ---"
	make build-docs-overview
	python -m mkdocs build


build-docs-overview:
	@echo "--- 📚 Building documentation overview ---"
	python docs/overview/create_available_tasks.py
	python docs/overview/create_available_models.py
	python docs/overview/create_available_benchmarks.py


serve-docs:
	@echo "--- 📚 Serving documentation ---"
	python -m mkdocs serve


model-load-test:
	@echo "--- 🚀 Running model load test ---"
	pip install ".[pylate,gritlm,xformers,model2vec]" --group dev
	python scripts/extract_model_names.py $(BASE_BRANCH) --return_one_model_name_per_file
	python tests/test_models/model_loading.py --model_name_file scripts/model_names.txt


dataset-load-test:
	@echo "--- 🚀 Running dataset load test ---"
	pytest -m test_datasets

dataset-load-test-pr:
	@echo "--- 🚀 Running dataset load test for PR ---"
	eval "$$(python -m scripts.extract_datasets $(BASE_BRANCH))" && pytest -m test_datasets

leaderboard-build-test:
	@echo "--- 🚀 Running leaderboard build test ---"
	pytest -n auto -m leaderboard_stability

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

.PHONY: typecheck
typecheck:
	@echo "--- 🔍 Running type checks ---"
	mypy mteb
