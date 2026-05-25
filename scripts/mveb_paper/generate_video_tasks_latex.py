#!/usr/bin/env python3
"""Generate LaTeX table for all video modality tasks in MTEB."""

import json
import sys
from collections import defaultdict
from pathlib import Path

# Add root directory to path to import mteb
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mteb import get_tasks
from mteb.benchmarks.benchmarks.benchmarks import MVEB, MVEB_TEXT_VIDEO, MVEB_VIDEO

# Build sets of task names in each benchmark scope
_MVEB_TASKS = {t.metadata.name for t in MVEB.tasks}
_MVEB_TV_TASKS = {t.metadata.name for t in MVEB_TEXT_VIDEO.tasks}
_MVEB_V_TASKS = {t.metadata.name for t in MVEB_VIDEO.tasks}


def extract_citation_key(citation):
    """Extract citation key from BibTeX citation."""
    if not citation:
        return "UnknownCitation"

    # Find first line with @article{, @inproceedings{, etc.
    lines = citation.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("@"):
            # Extract citation key between { and ,
            start = line.find("{")
            end = line.find(",")
            if start != -1 and end != -1:
                return line[start + 1 : end].strip()

    return "UnknownCitation"


def load_descriptive_stats(task_name, task_type):
    """Load descriptive statistics for a task."""
    stats_dir = Path(__file__).parent.parent.parent / "mteb" / "descriptive_stats"

    # Map task type to directory name
    type_mapping = {
        "VideoClassification": "VideoClassification",
        "VideoClustering": "VideoClustering",
        "VideoZeroshotClassification": "VideoZeroshotClassification",
        "VideoPairClassification": "VideoPairClassification",
        "VideoCentricQA": "VideoCentricQA",
        "Any2AnyRetrieval": "Image/Any2AnyRetrieval",  # Video retrieval tasks are in Image directory
    }

    stats_subdir = type_mapping.get(task_type)
    if not stats_subdir:
        return None

    stats_file = stats_dir / stats_subdir / f"{task_name}.json"

    if not stats_file.exists():
        return None

    try:
        with open(stats_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load stats for {task_name}: {e}")
        return None


def format_duration(total_seconds):
    """Format duration in a readable way."""
    if total_seconds is None:
        return "Unknown"

    if total_seconds < 3600:  # Less than 1 hour
        return f"{int(total_seconds):,}"
    elif total_seconds < 86400:  # Less than 1 day
        hours = int(total_seconds / 3600)
        return f"{hours}h ({int(total_seconds):,}s)"
    else:  # Days
        days = int(total_seconds / 86400)
        return f"{days}d ({int(total_seconds):,}s)"


def format_domains(domains):
    """Format domains list for LaTeX."""
    if not domains:
        return "Unknown"

    # Shorten common domain names
    domain_mapping = {
        "Encyclopedia": "Encycl.",
        "Academic": "Acad.",
        "Non-fiction": "Non-fic.",
        "Reviews": "Rev.",
    }

    formatted = []
    for domain in domains:
        formatted.append(domain_mapping.get(domain, domain))

    return ", ".join(formatted)


def classify_modality(modalities):
    """Classify task modalities into one of three categories."""
    modality_set = set(modalities)

    if modality_set == {"video"}:
        return "video_only"
    elif "text" in modality_set and "audio" in modality_set and "video" in modality_set:
        return "video_audio_text"
    elif "text" in modality_set and "video" in modality_set:
        return "video_text"
    elif "audio" in modality_set and "video" in modality_set:
        # Video-audio goes to video_audio_text category for now
        return "video_audio_text"
    else:
        # Default to video_only for any other combinations
        return "video_only"


def generate_single_table(tasks, caption, label):
    """Generate a single LaTeX table for a specific modality group."""

    if not tasks:
        return ""

    # Determine which benchmark columns are needed for this table
    task_names = {t.metadata.name for t in tasks}
    bench_cols = []  # list of (header, task_set)
    if task_names & _MVEB_TASKS:
        bench_cols.append(("MVEB", _MVEB_TASKS))
    if task_names & _MVEB_TV_TASKS:
        bench_cols.append(("TV", _MVEB_TV_TASKS))
    if task_names & _MVEB_V_TASKS:
        bench_cols.append(("V", _MVEB_V_TASKS))

    n_bench = len(bench_cols)
    # Column spec: l (name) + c (citation) + c*n_bench + r r r + c c
    col_spec = "lc" + "c" * n_bench + "rrrcc"
    total_cols = (
        2 + n_bench + 5
    )  # name, cite, benchmarks, samples, duration, langs, domains, metric

    bench_headers = " & ".join(f"\\textbf{{{h}}}" for h, _ in bench_cols)
    if bench_headers:
        bench_headers = " & " + bench_headers

    latex_content = []

    # Table header
    latex_content.extend(
        [
            f"\\begin{{table*}}[ht]",
            "\\centering",
            "\\resizebox{\\linewidth}{!}{",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
            f"\\textbf{{Dataset}} & \\textbf{{Citation}}{bench_headers} & \\textbf{{N.samples}} & \\textbf{{Total Duration(s)}} & \\textbf{{N.Langs}} & \\textbf{{Domains}} & \\textbf{{Main metric}} \\\\",
            "\\midrule",
        ]
    )

    # Group tasks by type
    tasks_by_type = defaultdict(list)
    for task in tasks:
        task_type = task.metadata.type
        tasks_by_type[task_type].append(task)

    # Define task type order and display names
    type_order = [
        ("VideoClassification", "Video Classification"),
        ("VideoClustering", "Video Clustering"),
        ("VideoZeroshotClassification", "Video Zero-shot Classification"),
        ("VideoPairClassification", "Video Pair Classification"),
        ("VideoCentricQA", "Video-Centric QA"),
        ("Any2AnyRetrieval", "Video Retrieval"),
    ]

    first_section = True
    for task_type, display_name in type_order:
        if task_type not in tasks_by_type:
            continue

        type_tasks = tasks_by_type[task_type]
        if not type_tasks:
            continue

        # Add section header (no midrule before first section)
        if not first_section:
            latex_content.append("\\midrule")
        first_section = False

        latex_content.append(
            f"\\multicolumn{{{total_cols}}}{{l}}{{\\textbf{{{display_name}}}}} \\\\"
        )

        # Sort tasks by name for consistent ordering
        sorted_tasks = sorted(type_tasks, key=lambda t: t.metadata.name)

        for task in sorted_tasks:
            # Load descriptive stats
            stats = load_descriptive_stats(task.metadata.name, task.metadata.type)

            # Extract data
            dataset_name = task.metadata.name or "Unknown"
            citation_key = extract_citation_key(task.metadata.bibtex_citation)

            # Get stats from test split
            n_samples = "Unknown"
            total_duration = "Unknown"

            if stats and "test" in stats:
                test_stats = stats["test"]
                n_samples = test_stats.get("num_samples", "Unknown")
                if n_samples != "Unknown":
                    n_samples = f"{n_samples:,}"

                # Calculate total duration by combining video statistics from different sources
                total_duration_seconds = 0
                duration_found = False

                # Regular tasks
                if (
                    test_stats.get("video_statistics")
                    and "total_duration_seconds" in test_stats["video_statistics"]
                ):
                    total_duration_seconds += test_stats["video_statistics"][
                        "total_duration_seconds"
                    ]
                    duration_found = True

                # Retrieval tasks - add both query and document durations
                if (
                    test_stats.get("documents_video_statistics")
                    and "total_duration_seconds"
                    in test_stats["documents_video_statistics"]
                ):
                    total_duration_seconds += test_stats["documents_video_statistics"][
                        "total_duration_seconds"
                    ]
                    duration_found = True

                if (
                    test_stats.get("queries_video_statistics")
                    and "total_duration_seconds"
                    in test_stats["queries_video_statistics"]
                ):
                    total_duration_seconds += test_stats["queries_video_statistics"][
                        "total_duration_seconds"
                    ]
                    duration_found = True

                # Pair classification tasks - add both video1 and video2 durations
                if (
                    test_stats.get("video1_statistics")
                    and "total_duration_seconds" in test_stats["video1_statistics"]
                ):
                    total_duration_seconds += test_stats["video1_statistics"][
                        "total_duration_seconds"
                    ]
                    duration_found = True

                if (
                    test_stats.get("video2_statistics")
                    and "total_duration_seconds" in test_stats["video2_statistics"]
                ):
                    total_duration_seconds += test_stats["video2_statistics"][
                        "total_duration_seconds"
                    ]
                    duration_found = True

                if duration_found:
                    total_duration = format_duration(total_duration_seconds)

            # Number of languages
            n_langs = len(task.metadata.eval_langs) if task.metadata.eval_langs else 1

            # Format domains
            domains_str = format_domains(task.metadata.domains)

            # Main metric
            main_metric = task.metadata.main_score or "Unknown"

            # Escape underscores and special characters for LaTeX
            dataset_name = dataset_name.replace("_", "\\_")
            domains_str = domains_str.replace("_", "\\_")
            main_metric = main_metric.replace("_", "\\_")

            # Benchmark membership checkmarks
            raw_name = task.metadata.name
            if bench_cols:
                bench_marks = " & " + " & ".join(
                    "\\checkmark" if raw_name in bset else "" for _, bset in bench_cols
                )
            else:
                bench_marks = ""

            # Create table row
            latex_content.append(
                f"{dataset_name} & \\cite{{{citation_key}}}{bench_marks} & {n_samples} & {total_duration} & {n_langs} & {domains_str} & {main_metric} \\\\"
            )

    # Table footer
    latex_content.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table*}",
        ]
    )

    return "\n".join(latex_content)


def generate_bibliography(video_tasks):
    """Generate BibTeX bibliography from task citations."""

    seen_citations = set()
    bib_entries = []

    for task in video_tasks:
        if task.metadata.bibtex_citation and task.metadata.bibtex_citation.strip():
            # Clean up citation - remove extra whitespace
            citation = task.metadata.bibtex_citation.strip()
            if not citation.endswith("}"):
                citation += "}"

            # Deduplicate by citation key
            citation_key = extract_citation_key(citation)
            if citation_key not in seen_citations:
                seen_citations.add(citation_key)
                bib_entries.append(citation)

    return "\n\n".join(bib_entries)


def generate_combined_all_tasks_table(video_tasks):
    """Generate combined all_tasks.tex file with 4 tables."""

    # Classify tasks by modality
    tasks_by_modality = {"video_only": [], "video_text": [], "video_audio_text": []}

    for task in video_tasks:
        modality_class = classify_modality(task.metadata.modalities)
        tasks_by_modality[modality_class].append(task)

    # Separate video-audio-text into retrieval and non-retrieval
    video_audio_text_tasks = tasks_by_modality["video_audio_text"]
    video_audio_text_non_retrieval = []
    video_audio_text_retrieval = []

    for task in video_audio_text_tasks:
        if task.metadata.type == "Any2AnyRetrieval":
            video_audio_text_retrieval.append(task)
        else:
            video_audio_text_non_retrieval.append(task)

    # Generate all 4 tables
    latex_content = []

    # Header
    latex_content.extend(
        [
            "% MVEB Task Overview Tables",
            "% generated by scripts/mveb_paper/generate_video_tasks_latex.py",
            "",
        ]
    )

    # Table 1: Video-only tasks
    table1 = generate_single_table(
        tasks_by_modality["video_only"],
        "Video-only tasks in MVEB. All tasks use video modality exclusively.",
        "tab:mveb-video-only-tasks",
    )
    latex_content.append(table1)
    latex_content.extend(["", ""])

    # Table 2: Video-text tasks
    latex_content.append("% Video-text multimodal tasks table")
    table2 = generate_single_table(
        tasks_by_modality["video_text"],
        "Video-text multimodal tasks in MVEB. Tasks use both video and text modalities.",
        "tab:mveb-video-text-tasks",
    )
    latex_content.append(table2)
    latex_content.extend(["", ""])

    # Table 3: Video-audio(-text) non-retrieval tasks
    latex_content.append("% Video-audio(-text) multimodal tasks table (non-retrieval)")
    table3 = generate_single_table(
        video_audio_text_non_retrieval,
        "Video-audio(-text) multimodal tasks in MVEB (non-retrieval). Tasks use video with audio and optionally text.",
        "tab:mveb-video-audio-text-tasks",
    )
    latex_content.append(table3)
    latex_content.extend(["", ""])

    # Table 4: Video-audio(-text) retrieval tasks
    latex_content.append("% Video-audio(-text) retrieval tasks table")
    table4 = generate_single_table(
        video_audio_text_retrieval,
        "Video-audio(-text) retrieval tasks in MVEB. Tasks use video with audio and optionally text for retrieval.",
        "tab:mveb-video-audio-text-retrieval-tasks",
    )
    latex_content.append(table4)

    return "\n".join(latex_content)


def main():
    """Main function."""
    base_dir = Path(__file__).parent
    output_file = base_dir / "all_tasks.tex"
    bib_file = base_dir / "video_tasks.bib"

    try:
        # Get all video tasks (including beta tasks)
        video_tasks = get_tasks(modalities=["video"], exclude_beta=False)

        print(f"Found {len(video_tasks)} video modality tasks")

        # Classify tasks by modality for summary
        tasks_by_modality = {"video_only": [], "video_text": [], "video_audio_text": []}

        for task in video_tasks:
            modality_class = classify_modality(task.metadata.modalities)
            tasks_by_modality[modality_class].append(task)

        # Print summary by modality
        for modality, tasks in tasks_by_modality.items():
            print(f"  {modality.replace('_', '-')}: {len(tasks)} tasks")

        # Generate combined all_tasks.tex file
        combined_latex = generate_combined_all_tasks_table(video_tasks)

        # Generate bibliography
        bib_content = generate_bibliography(video_tasks)

        # Write all_tasks.tex file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(combined_latex)

        # Write bibliography
        with open(bib_file, "w", encoding="utf-8") as f:
            f.write(bib_content)

        print(f"\nFiles generated:")
        print(f"Combined tables: {output_file} ({output_file.stat().st_size} bytes)")
        print(f"Bibliography: {bib_file} ({bib_file.stat().st_size} bytes)")

        # Show brief summary
        print(f"\nGenerated all_tasks.tex with 4 tables:")
        print(f"  1. Video-only tasks ({len(tasks_by_modality['video_only'])} tasks)")
        print(f"  2. Video-text tasks ({len(tasks_by_modality['video_text'])} tasks)")

        # Count non-retrieval and retrieval for video-audio-text
        non_retrieval = sum(
            1
            for task in tasks_by_modality["video_audio_text"]
            if task.metadata.type != "Any2AnyRetrieval"
        )
        retrieval = sum(
            1
            for task in tasks_by_modality["video_audio_text"]
            if task.metadata.type == "Any2AnyRetrieval"
        )

        print(f"  3. Video-audio(-text) non-retrieval tasks ({non_retrieval} tasks)")
        print(f"  4. Video-audio(-text) retrieval tasks ({retrieval} tasks)")

    except Exception as e:
        print(f"Error generating files: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
