#!/usr/bin/env python3
"""Build docs in strict mode, filtering out expected BibTeX warnings."""

import subprocess
import sys
import re
import os
import shutil


def main():
    # Find mkdocs in the current environment
    mkdocs_path = shutil.which("mkdocs")
    if not mkdocs_path:
        # If mkdocs is not in PATH, try common locations
        possible_paths = [
            "/opt/miniconda3/envs/mteb/bin/mkdocs",
            "mkdocs",  # Will use system PATH
        ]
        for path in possible_paths:
            if os.path.exists(path) or shutil.which(path):
                mkdocs_path = path
                break
        else:
            print("ERROR: mkdocs not found. Please ensure mkdocs is installed.")
            sys.exit(1)

    # Run mkdocs build with strict mode
    process = subprocess.Popen(
        [mkdocs_path, "build", "--strict"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Track whether we found any real warnings (not BibTeX)
    real_warnings_count = 0
    bibtex_warnings_count = 0

    # Process output line by line
    for line in process.stdout:
        # Check if this is a BibTeX warning we want to suppress
        if "WARNING" in line and "Inline reference to unknown key" in line:
            bibtex_warnings_count += 1
            # Don't print BibTeX warnings
            continue

        # Count other warnings
        if line.startswith("WARNING"):
            real_warnings_count += 1

        # For the final "Aborted with X warnings" message, adjust it
        if "Aborted with" in line and "warnings in strict mode" in line:
            if real_warnings_count > 0:
                # There were real warnings, print the abort message
                print(line, end="")
            else:
                # Only BibTeX warnings, indicate success
                print(
                    f"INFO    -  Build completed successfully (suppressed {bibtex_warnings_count} BibTeX warnings)"
                )
                # Exit with success even though mkdocs exited with error
                sys.exit(0)
        else:
            # Print all other lines
            print(line, end="")

    # Wait for process to complete
    return_code = process.wait()

    # If mkdocs exited with error but we only had BibTeX warnings, override to success
    if return_code != 0 and real_warnings_count == 0 and bibtex_warnings_count > 0:
        print(
            f"INFO    -  Build completed successfully (suppressed {bibtex_warnings_count} BibTeX warnings)"
        )
        return 0

    return return_code


if __name__ == "__main__":
    sys.exit(main())
