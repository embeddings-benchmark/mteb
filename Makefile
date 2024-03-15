.PHONY: modified_only_fixup quality style fixup tests

check_dirs := tests mteb scripts

modified_only_fixup:
	$(eval modified_py_files := $(shell python utils/get_modified_files.py $(check_dirs)))
	@if test -n "$(modified_py_files)"; then \
		echo "Checking/fixing $(modified_py_files)"; \
		black --preview $(modified_py_files); \
		isort $(modified_py_files); \
		flake8 $(modified_py_files); \
	else \
		echo "No library .py files were modified"; \
	fi

# Super fast fix and check target that only works on relevant modified files since the branch was made
fixup: modified_only_fixup


# This installs all the required dependencies
install:
	pip install -e .
	pip install -r requirements.dev.txt

# this target runs checks on all files
quality:
	black --check --preview $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)


# this target runs checks on all files and potentially modifies some of them
style:
	black --preview $(check_dirs)
	isort $(check_dirs)

# runs the same lints as the github actions
lint:
	# stop the build if there are Python syntax errors or undefined names
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	# exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Run tests for the library
test:
	pytest -s -v ./tests/

# add parllel test for faster execution (can sometimes cause issues with some tests)
test-parallel:
	pytest -n auto --dist=loadfile -s -v ./tests/