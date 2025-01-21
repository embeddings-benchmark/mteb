install:
	@echo "--- 🚀 Installing project dependencies ---"
	pip install -e ".[dev,docs]"

install-for-tests:
	@echo "--- 🚀 Installing project dependencies for test ---"
	@echo "This ensures that the project is not installed in editable mode"
	pip install ".[dev,speedtask,bm25s,pylate]"

lint:
	@echo "--- 🧹 Running linters ---"
	ruff format . 			# running ruff formatting
	ruff check . --fix  	# running ruff linting

lint-check:
	@echo "--- 🧹 Check is project is linted ---"
	# Required for CI to work, otherwise it will just pass
	ruff format . --check						    # running ruff formatting
	ruff check **/*.py 						        # running ruff linting

test:
	@echo "--- 🧪 Running tests ---"
	pytest -n auto --durations=5

test-with-coverage:
	@echo "--- 🧪 Running tests with coverage ---"
	pytest -n auto --durations=5 --cov-report=term-missing --cov-config=pyproject.toml --cov=mteb

pr:
	@echo "--- 🚀 Running requirements for a PR ---"
	make lint
	make test


build-docs:
	@echo "--- 📚 Building documentation ---"
	# since we do not have a documentation site, this just build tables for the .md files
	python docs/create_tasks_table.py

serve-docs:
	@echo "--- 📚 Serving documentation ---"
	python -m mkdocs serve


model-load-test:
	@echo "--- 🚀 Running model load test ---"
	pip install ".[dev, speedtask, pylate,gritlm,xformers,model2vec]"
	python scripts/extract_model_names.py $(BASE_BRANCH) --return_one_model_name_per_file
	python tests/test_models/model_loading.py --model_name_file scripts/model_names.txt