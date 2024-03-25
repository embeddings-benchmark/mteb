install:
	@echo "--- 🚀 Installing project dependencies ---"
	pip install -e ".[dev]"

lint:
	@echo "--- 🧹 Running linters ---"
	ruff format . 			# running ruff formatting
	ruff check . --fix  	# running ruff linting

test:
	@echo "--- 🧪 Running tests ---"
	pytest -n auto --durations=5

pr:
	@echo "--- 🚀 Running requirements for a PR ---"
	make lint
	make test-parallel
