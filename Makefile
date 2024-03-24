install:
	@echo "--- 🚀 Installing project dependencies ---"
	pip install -e ".[dev]"

lint:
	@echo "--- 🧹 Running linters ---"
	ruff format . 			# running ruff formatting
	ruff check . --fix  	# running ruff linting

test:
	@echo "--- 🧪 Running tests ---"
	pytest

test-parallel:
	@echo "--- 🧪 Running tests ---"
	@echo "Note that parallel tests can sometimes cause issues with some tests."
	pytest -n auto --dist=loadfile -s -v

pr:
	@echo "--- 🚀 Running requirements for a PR ---"
	make lint
	make test-parallel
