from __future__ import annotations

import argparse
import ast
import logging
from pathlib import Path

import bibtexparser
from bibtexparser.bwriter import BibTexWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


class KeywordLiteralFinder(ast.NodeVisitor):
    def __init__(self, target_function_name: str, target_keyword_arg: str):
        self.target_function_name = target_function_name
        self.target_keyword_arg = target_keyword_arg
        self.locations: list[tuple[int, int, int, int]] = []
        self.keyword_found_anywhere = False

    def visit_Call(self, node: ast.Call):
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name != self.target_function_name:
            self.generic_visit(node)
            return

        for keyword in node.keywords:
            if keyword.arg != self.target_keyword_arg:
                continue
            self.keyword_found_anywhere = True
            if not isinstance(keyword.value, ast.Constant) or not isinstance(
                keyword.value.value, str
            ):
                continue

            if (
                keyword.value.end_lineno is not None
                and keyword.value.end_col_offset is not None
            ):
                self.locations.append(
                    (
                        keyword.value.lineno,
                        keyword.value.col_offset,
                        keyword.value.end_lineno,
                        keyword.value.end_col_offset,
                    )
                )
            else:
                logger.warning(
                    f"Could not get end location for a {self.target_keyword_arg} string. Skipping this instance."
                )
        self.generic_visit(node)


def extract_string_literal(
    lines: list[str], location: tuple[int, int, int, int]
) -> tuple[str | None, str | None]:
    start_line, start_col, end_line, end_col = location
    start_line_0, end_line_0 = start_line - 1, end_line - 1

    if (
        start_line_0 < 0
        or end_line_0 >= len(lines)
        or start_col > len(lines[start_line_0])
        or end_col > len(lines[end_line_0])
    ):
        return None, None

    if start_line == end_line:
        literal = lines[start_line_0][start_col:end_col]
    else:
        first_line = lines[start_line_0][start_col:]
        middle_lines = (
            lines[start_line_0 + 1 : end_line_0]
            if start_line_0 + 1 <= end_line_0
            else []
        )
        last_line = lines[end_line_0][:end_col]
        literal = "\n".join([first_line] + middle_lines + [last_line])

    quote_types = ['"""', "'''", '"', "'"]
    for quote in quote_types:
        for prefix in [f"r{quote}", quote]:
            if literal.startswith(prefix) and literal.endswith(quote):
                return literal[len(prefix) : -len(quote)], quote

    return None, None


def format_bibtex(bibtex_str: str) -> str | None:
    parser = bibtexparser.bparser.BibTexParser(
        common_strings=True, ignore_nonstandard_types=False, interpolate_strings=False
    )

    try:
        bib_database = bibtexparser.loads(bibtex_str, parser=parser)
        if not bib_database.entries:
            logger.warning(f"No entries found in BibTeX string. {bibtex_str}")
            return None
        bib_database.comments = []

        writer = BibTexWriter()
        writer.indent = "  "
        writer.comma_first = False
        writer.add_trailing_comma = True

        return writer.write(bib_database).strip()
    except Exception as e:
        logger.warning(f"Failed to parse BibTeX: {e}")
        return None


def process_file(
    file_path: Path,
    target_function_name: str,
    target_keyword_arg: str,
    dry_run: bool,
) -> tuple[bool, bool, int, bool, bool]:
    file_modified = file_error = skipped_no_keyword = skipped_no_locations = False
    num_modified_in_file = 0
    replacements_for_file = []

    try:
        content = file_path.read_text()
        tree = ast.parse(content, filename=str(file_path))

        finder = KeywordLiteralFinder(target_function_name, target_keyword_arg)
        finder.visit(tree)

        if not finder.keyword_found_anywhere:
            return False, False, 0, True, False

        if not finder.locations:
            return False, False, 0, False, True

        content_lines = content.splitlines()
        content_lines_with_endings = content.splitlines(True)

        for location in finder.locations:
            literal_value, quote_type = extract_string_literal(content_lines, location)

            if literal_value is None or quote_type is None:
                logger.error(
                    f"In {file_path.name}: Could not extract {target_keyword_arg} string literal at {location}"
                )
                file_error = True
                continue

            literal_str = literal_value.strip()
            if not literal_str:
                continue

            formatted_literal = format_bibtex(literal_str)
            if formatted_literal is None:
                logger.error(
                    f"In {file_path.name}: Failed to parse/format {target_keyword_arg} at {location}"
                )
                file_error = True
                continue

            if literal_str == formatted_literal:
                continue

            new_literal = f'r"""\n{formatted_literal}\n"""'

            start_line, start_col, end_line, end_col = location
            start_char_index = (
                sum(len(line) for line in content_lines_with_endings[: start_line - 1])
                + start_col
            )
            end_char_index = (
                sum(len(line) for line in content_lines_with_endings[: end_line - 1])
                + end_col
            )

            original_slice = content[start_char_index:end_char_index]
            matched_prefix = ""
            if original_slice.startswith(f"r{quote_type}"):
                matched_prefix = "r"

            full_original_literal = (
                f"{matched_prefix}{quote_type}{literal_value}{quote_type}"
            )

            try:
                actual_start = content.index(full_original_literal, start_char_index)
                actual_end = actual_start + len(full_original_literal)
                replacements_for_file.append((actual_start, actual_end, new_literal))
                num_modified_in_file += 1
            except ValueError:
                logger.warning(
                    f"In {file_path.name}: Could not find exact original literal match for {target_keyword_arg} at {location}. Using offset-based replacement."
                )
                replacements_for_file.append(
                    (start_char_index, end_char_index, new_literal)
                )
                num_modified_in_file += 1

        if replacements_for_file:
            replacements_for_file.sort(key=lambda x: x[0], reverse=True)
            new_content = content
            for start, end, literal in replacements_for_file:
                new_content = new_content[:start] + literal + new_content[end:]

            if not dry_run:
                file_path.write_text(new_content)
            file_modified = True

    except SyntaxError as e:
        logger.error(f"SyntaxError in {file_path.name}: {e}")
        file_error = True
    except Exception as e:
        logger.error(f"Unexpected error in {file_path.name}: {e}")
        import traceback

        traceback.print_exc()
        file_error = True

    return (
        file_modified,
        file_error,
        num_modified_in_file,
        skipped_no_keyword,
        skipped_no_locations,
    )


def tasks(args):
    tasks_dir = Path(args.tasks_dir)
    dry_run = args.dry_run
    error_on_change = args.error_on_change

    modified_files = error_files = skipped_files = processed_files = bibtex_modified = 0
    task_files = sorted(tasks_dir.rglob("*.py"))

    if not task_files:
        logger.error(f"No Python files found in {tasks_dir}")
        raise RuntimeError

    logger.info(f"Found {len(task_files)} Python files in {tasks_dir}. Processing...")

    for file_path in task_files:
        if file_path.name == "__init__.py":
            continue

        processed_files += 1
        file_modified, file_error, num_modified, no_keyword, no_locations = (
            process_file(file_path, "TaskMetadata", "bibtex_citation", dry_run)
        )

        if file_error:
            error_files += 1
        elif file_modified:
            modified_files += 1
            bibtex_modified += num_modified
        else:
            skipped_files += 1

    logger.info("\n--- Summary ---")
    logger.info(f"Processed Files: {processed_files}")
    logger.info(f"Modified Files:  {modified_files}")
    logger.info(f"Skipped Files:   {skipped_files}")
    logger.info(f"Error Files:     {error_files}")
    logger.info(f"Total BibTeX Instances Modified: {bibtex_modified}")

    if dry_run:
        logger.info("\nNOTE: Dry run mode was enabled. No files were actually changed.")

    if file_modified and error_on_change:
        raise Exception("Files are modified")

    if error_files > 0:
        logger.warning("Errors occurred during processing. Check logs above.")
        raise RuntimeError


def benchmarks(args):
    benchmarks_file = Path(args.benchmarks_file)
    dry_run = args.dry_run
    error_on_change = args.error_on_change

    logger.info(f"Processing {benchmarks_file}...")

    file_modified, file_error, num_modified, no_keyword, no_locations = process_file(
        benchmarks_file, "Benchmark", "citation", dry_run
    )

    if no_keyword:
        logger.info(f"SKIPPED: No 'citation' keyword found in {benchmarks_file.name}.")
        return
    if no_locations:
        logger.info(
            f"SKIPPED: 'citation' keyword found, but no valid string literals detected in {benchmarks_file.name}."
        )
        return

    logger.info("\n--- Summary ---")
    logger.info(f"Processed File: {benchmarks_file.name}")
    logger.info(f"Modified:        {'Yes' if file_modified else 'No'}")
    logger.info(f"Errors Occurred: {'Yes' if file_error else 'No'}")
    logger.info(f"Citations Modified: {num_modified}")

    if dry_run and file_modified:
        logger.info("\nNOTE: Dry run mode was enabled. File was not actually changed.")

    if file_modified and error_on_change:
        raise Exception("Files are modified")

    if file_error:
        logger.warning("Errors occurred during processing. Check logs above.")
        return
    elif not file_modified and not file_error:
        logger.info("No changes needed.")


def main():
    parser = argparse.ArgumentParser(
        description="Refactor script to use argparse instead of typer."
    )
    subparsers = parser.add_subparsers()

    tasks_parser = subparsers.add_parser("tasks", help="Process tasks directory")
    tasks_parser.add_argument(
        "--tasks_dir",
        type=str,
        default=str(Path("mteb/tasks")),
        help="Directory containing MTEB task Python files.",
    )
    tasks_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform parsing and formatting but do not modify files.",
    )
    tasks_parser.add_argument(
        "--error-on-change",
        action="store_true",
        help="Raise error when files are modified. Need for pre-commit",
    )
    tasks_parser.set_defaults(func=tasks)

    benchmarks_parser = subparsers.add_parser(
        "benchmarks", help="Process benchmarks file"
    )
    benchmarks_parser.add_argument(
        "--benchmarks_file",
        type=str,
        default=str(Path("mteb/benchmarks/benchmarks.py")),
        help="Path to the benchmarks.py file.",
    )
    benchmarks_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform parsing and formatting but do not modify the file.",
    )
    benchmarks_parser.add_argument(
        "--error-on-change",
        action="store_true",
        help="Raise error when files are modified. Need for pre-commit",
    )
    benchmarks_parser.set_defaults(func=benchmarks)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
