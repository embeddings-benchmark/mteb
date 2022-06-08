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


# this target runs checks on all files

quality:
	black --check --preview $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)

# this target runs checks on all files and potentially modifies some of them

style:
	black --preview $(check_dirs)
	isort $(check_dirs)

# Super fast fix and check target that only works on relevant modified files since the branch was made

fixup: modified_only_fixup

# Run tests for the library

test:
	pytest -n auto --dist=loadfile -s -v ./tests/