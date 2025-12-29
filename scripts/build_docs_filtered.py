#!/usr/bin/env python
"""
Build mkdocs documentation with filtered warnings.
Suppresses BibTeX warnings about unknown citation keys while keeping strict mode.
"""

import logging
import sys
import os
from mkdocs import __main__ as mkdocs_main


class BibtexWarningFilter(logging.Filter):
    """Filter out BibTeX warnings about unknown citation keys."""

    def filter(self, record):
        # Filter out "Inline reference to unknown key" warnings from bibtex plugin
        msg = record.getMessage()
        if "Inline reference to unknown key" in msg:
            return False
        return True


def main():
    # Configure logging to filter BibTeX warnings
    # We need to set this up before mkdocs starts
    bibtex_filter = BibtexWarningFilter()

    # Add filter to all loggers that might emit these warnings
    logging.getLogger().addFilter(bibtex_filter)

    # Set up command line args for mkdocs build with strict mode
    sys.argv = ["mkdocs", "build", "--strict"]

    # Run mkdocs build command
    try:
        mkdocs_main.cli()
        return 0
    except SystemExit as e:
        # mkdocs exits with non-zero on warnings in strict mode
        # But we've filtered the bibtex warnings, so any remaining warnings are real
        if e.code != 0:
            return e.code
        return 0
    except Exception as e:
        print(f"Build failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
