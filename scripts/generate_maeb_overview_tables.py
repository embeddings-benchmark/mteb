#!/usr/bin/env python3
"""
Generate LaTeX overview tables for MAEB benchmarks.

This script creates a single task_overview.tex file containing two tables:
1. Audio-only tasks (a2a modality)
2. Audio-text cross-modal tasks (a2t, t2a modalities)

Each table includes task metadata, benchmark membership, and main metrics.
"""

import re
from typing import Dict, List, Tuple
import pandas as pd
import bibtexparser
from bibtexparser.bparser import BibTexParser

from mteb import get_benchmark, get_tasks


# Manual citation fixes for tasks with missing or incorrect citations
CITATION_FIXES = {
    "BirdCLEF": "birdclef2025",
    "CommonLanguageAgeDetection": "ganesh_sinisetty_2021_5036977",
    "CommonLanguageGenderDetection": "ganesh_sinisetty_2021_5036977",
    "CommonLanguageLanguageDetection": "ganesh_sinisetty_2021_5036977",
    "FSD2019Kaggle": "eduardo_fonseca_2020_3612637",
    "VehicleSoundClustering": "bazilinskyy2018auditory",
    "VocalSound": "Gong_2022",
    "VocalSoundPairClassification": "Gong_2022",
    "VocalSoundAudioReranking": "Gong_2022",
    "GoogleSVQA2TRetrieval": "heigold2025massive",
    "GoogleSVQT2ARetrieval": "heigold2025massive",
}


def extract_citation_key(bibtex_str: str) -> str:
    """Extract citation key from bibtex string."""
    if not bibtex_str:
        return ""

    try:
        parser = BibTexParser()
        parser.ignore_nonstandard_types = True
        parser.homogenize_fields = False
        bib_db = bibtexparser.loads(bibtex_str, parser)

        if bib_db.entries:
            return bib_db.entries[0]["ID"]
    except Exception as e:
        print(f"Warning: Failed to parse bibtex: {e}")
        # Try simple regex as fallback
        match = re.search(r"@\w+\{([^,]+),", bibtex_str)
        if match:
            return match.group(1)

    return ""


def format_languages(languages: List[str]) -> str:
    """Format language list for display."""
    if not languages:
        return "1"

    # Convert BCP-47 to simple language codes
    simple_langs = []
    for lang in languages:
        # Take the first part before any dash
        simple_lang = lang.split("-")[0]
        simple_langs.append(simple_lang)

    # Remove duplicates while preserving order
    seen = set()
    unique_langs = []
    for lang in simple_langs:
        if lang not in seen:
            seen.add(lang)
            unique_langs.append(lang)

    n_langs = len(unique_langs)

    if n_langs == 0:
        return "1"
    else:
        # Only return the number
        return str(n_langs)


def format_domains(domains: List[str]) -> str:
    """Format domains list for display."""
    if not domains:
        return ""

    # Limit to first 3 domains if many
    if len(domains) <= 3:
        return ", ".join(domains)
    else:
        return ", ".join(domains[:3]) + "..."


def get_benchmark_membership(task_name: str, benchmarks: Dict) -> Dict[str, bool]:
    """Check if task is in lite/extended benchmarks."""
    membership = {"lite": False, "extended": False}

    for bench_name, bench in benchmarks.items():
        task_names = [t.metadata.name for t in bench.tasks]
        if task_name in task_names:
            if "lite" in bench_name.lower():
                membership["lite"] = True
            elif "extended" in bench_name.lower():
                membership["extended"] = True

    return membership


def load_benchmark_tasks() -> Tuple[Dict, List]:
    """Load all MAEB benchmark definitions and tasks."""
    # Load benchmark definitions
    benchmarks = {
        "maeb": get_benchmark("MAEB+"),
        "audio_lite": get_benchmark("MAEB(audio, lite)"),
        "audio_extended": get_benchmark("MAEB(audio, extended)"),
        "audio_text_lite": get_benchmark("MAEB(audio-text, lite)"),
        "audio_text_extended": get_benchmark("MAEB(audio-text, extended)"),
    }

    # Get all MAEB tasks
    all_tasks = benchmarks["maeb"].tasks

    return benchmarks, all_tasks


def extract_task_metadata(tasks: List, benchmarks: Dict) -> pd.DataFrame:
    """Extract metadata for all tasks."""
    task_data = []

    for task in tasks:
        metadata = task.metadata

        # Check for manual citation fixes first
        if metadata.name in CITATION_FIXES:
            citation_key = CITATION_FIXES[metadata.name]
        else:
            # Extract citation key from bibtex
            citation_key = extract_citation_key(metadata.bibtex_citation)
            # Skip generic "inproceedings" key
            if citation_key == "inproceedings":
                citation_key = ""

        # Format citation
        if citation_key:
            citation = f"\\cite{{{citation_key}}}"
        else:
            citation = ""

        # Get benchmark membership
        membership = get_benchmark_membership(metadata.name, benchmarks)

        # Format languages
        lang_str = format_languages(metadata.languages)

        # Format domains
        domain_str = format_domains(metadata.domains)

        # Get main metric
        main_metric = metadata.main_score if metadata.main_score else "accuracy"
        # Clean up metric name (remove 'test.' prefix if present)
        main_metric = main_metric.replace("test.", "").replace("_", "\\_")

        # Get task type for grouping
        task_type = metadata.type

        # Simplify task type for display
        display_type = task_type.replace("Audio", "").replace("Any2Any", "")
        if display_type in ["Retrieval", "MultilingualRetrieval"]:
            # Determine retrieval direction
            if metadata.category == "a2t":
                display_type = "A2T Retrieval"
            elif metadata.category == "t2a":
                display_type = "T2A Retrieval"
            elif metadata.category == "a2a":
                display_type = "A2A Retrieval"
        elif display_type == "MultilabelClassification":
            display_type = "Multilabel Clf."
        elif display_type == "ZeroshotClassification":
            display_type = "Zero-shot Clf."
        elif display_type == "PairClassification":
            display_type = "Pair Clf."

        task_data.append(
            {
                "name": metadata.name,
                "citation": citation,
                "lite": "\\checkmark" if membership["lite"] else "",
                "extended": "\\checkmark" if membership["extended"] else "",
                "languages": lang_str,
                "task_type": task_type,
                "display_type": display_type,
                "modality": metadata.category,
                "domains": domain_str,
                "main_metric": main_metric,
            }
        )

    return pd.DataFrame(task_data)


def generate_audio_tasks_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table for audio-only tasks."""
    # Filter for audio-only tasks
    # Include: Classification, Clustering, PairClassification, Reranking, MultilabelClassification
    # and audio-to-audio retrieval (a2a modality)
    # Exclude: ZeroshotClassification and cross-modal retrieval (a2t, t2a)
    audio_types = [
        "AudioClassification",
        "AudioClustering",
        "AudioPairClassification",
        "AudioReranking",
        "AudioMultilabelClassification",
    ]
    # Get tasks by type OR audio-to-audio retrieval
    audio_df = df[
        (df["task_type"].isin(audio_types))
        | (
            (df["task_type"].isin(["Any2AnyRetrieval", "Any2AnyMultilingualRetrieval"]))
            & (df["modality"] == "a2a")
        )
    ].copy()

    if audio_df.empty:
        return ""

    # Normalize task types for grouping (treat Any2AnyMultilingualRetrieval same as Any2AnyRetrieval)
    audio_df["group_type"] = audio_df["task_type"].replace(
        "Any2AnyMultilingualRetrieval", "Any2AnyRetrieval"
    )

    # Sort by normalized task type, then by name
    audio_df = audio_df.sort_values(["group_type", "name"])

    # Start building the table
    latex_lines = [
        "% Audio-Only Tasks Table",
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\caption{MAEB+ Audio-Only Tasks Overview. Tasks are grouped by type and show benchmark membership (Lite/Ext.), language coverage, domains, and main evaluation metric.}",
        "    \\resizebox{\\linewidth}{!}{",
        "    \\begin{tabular}{llccclc}",
        "    \\toprule",
        "    \\textbf{Dataset} & \\textbf{Citation} & \\textbf{Lite} & \\textbf{Ext.} & "
        + "\\textbf{N. Langs} & \\textbf{Domains} & \\textbf{Main Metric} \\\\",
        "    \\midrule",
    ]

    # Group by task type (using normalized group_type)
    current_type = None
    first_section = True
    for _, row in audio_df.iterrows():
        # Add section header for new task type (using normalized group_type)
        if row["group_type"] != current_type:
            current_type = row["group_type"]
            # Format section header (use original task_type for display)
            type_header = (
                row["task_type"]
                .replace("Audio", "")
                .replace("Clustering", "Clustering")
                .replace("Any2AnyMultilingual", "Any2Any")
            )
            # Add hline before section header (except for first section)
            if not first_section:
                latex_lines.append("    \\hline")
            latex_lines.append(
                f"    \\multicolumn{{7}}{{l}}{{\\textit{{{type_header}}}}} \\\\"
            )
            latex_lines.append("    \\hline")
            first_section = False

        # Format task name (escape underscores)
        task_name = row["name"].replace("_", "\\_")

        # Build row
        row_str = (
            f"    {task_name} & {row['citation']} & {row['lite']} & {row['extended']} & "
            + f"{row['languages']} & {row['domains']} & {row['main_metric']} \\\\"
        )
        latex_lines.append(row_str)

    latex_lines.extend(
        [
            "    \\bottomrule",
            "    \\end{tabular}",
            "    }",
            "    \\label{tab:maeb_audio_overview}",
            "\\end{table*}",
            "",
        ]
    )

    return "\n".join(latex_lines)


def generate_audio_text_tasks_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table for audio-text cross-modal tasks."""
    # Filter for audio-text cross-modal tasks
    # Include: ZeroshotClassification and cross-modal retrieval (a2t, t2a)
    # Exclude: audio-to-audio retrieval (a2a)
    audio_text_df = df[
        (df["task_type"] == "AudioZeroshotClassification")
        | (
            (df["task_type"].isin(["Any2AnyRetrieval", "Any2AnyMultilingualRetrieval"]))
            & (df["modality"].isin(["a2t", "t2a"]))
        )
    ].copy()

    if audio_text_df.empty:
        return ""

    # Normalize task types for grouping (treat Any2AnyMultilingualRetrieval same as Any2AnyRetrieval)
    audio_text_df["group_type"] = audio_text_df["task_type"].replace(
        "Any2AnyMultilingualRetrieval", "Any2AnyRetrieval"
    )

    # Sort by normalized task type and modality, then by name
    audio_text_df = audio_text_df.sort_values(["group_type", "modality", "name"])

    # Start building the table
    latex_lines = [
        "% Audio-Text Cross-Modal Tasks Table",
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\caption{MAEB+ Audio-Text Cross-Modal Tasks Overview. Tasks include zero-shot classification and bidirectional retrieval between audio and text modalities.}",
        "    \\resizebox{\\linewidth}{!}{",
        "    \\begin{tabular}{llcccllc}",
        "    \\toprule",
        "    \\textbf{Dataset} & \\textbf{Citation} & \\textbf{Lite} & \\textbf{Ext.} & "
        + "\\textbf{N. Langs} & \\textbf{Modality} & \\textbf{Domains} & \\textbf{Main Metric} \\\\",
        "    \\midrule",
    ]

    # Group by task type and modality (using normalized group_type)
    current_group = None
    first_section = True
    for _, row in audio_text_df.iterrows():
        group_key = (row["group_type"], row["modality"])

        # Add section header for new group
        if group_key != current_group:
            current_group = group_key
            # Format section header
            if "Zeroshot" in row["task_type"]:
                type_header = "Zero-shot Classification"
            elif row["modality"] == "a2t":
                type_header = "Audio-to-Text Retrieval"
            elif row["modality"] == "t2a":
                type_header = "Text-to-Audio Retrieval"
            else:
                type_header = row["task_type"].replace("Audio", "")

            # Add hline before section header (except for first section)
            if not first_section:
                latex_lines.append("    \\hline")
            latex_lines.append(
                f"    \\multicolumn{{8}}{{l}}{{\\textit{{{type_header}}}}} \\\\"
            )
            latex_lines.append("    \\hline")
            first_section = False

        # Format task name (escape underscores)
        task_name = row["name"].replace("_", "\\_")

        # Build row
        row_str = (
            f"    {task_name} & {row['citation']} & {row['lite']} & {row['extended']} & "
            + f"{row['languages']} & {row['modality']} & "
            + f"{row['domains']} & {row['main_metric']} \\\\"
        )
        latex_lines.append(row_str)

    latex_lines.extend(
        [
            "    \\bottomrule",
            "    \\end{tabular}",
            "    }",
            "    \\label{tab:maeb_audio_text_overview}",
            "\\end{table*}",
            "",
        ]
    )

    return "\n".join(latex_lines)


def main():
    """Main function to generate MAEB overview tables."""
    print("Loading MAEB benchmarks and tasks...")
    benchmarks, all_tasks = load_benchmark_tasks()

    print(f"Found {len(all_tasks)} tasks across all MAEB benchmarks")

    print("Extracting task metadata...")
    df = extract_task_metadata(all_tasks, benchmarks)

    # Count tasks by table category
    audio_types = [
        "AudioClassification",
        "AudioClustering",
        "AudioPairClassification",
        "AudioReranking",
        "AudioMultilabelClassification",
    ]
    audio_count = len(
        df[
            (df["task_type"].isin(audio_types))
            | ((df["task_type"] == "Any2AnyRetrieval") & (df["modality"] == "a2a"))
        ]
    )
    audio_text_count = len(
        df[
            (df["task_type"] == "AudioZeroshotClassification")
            | (
                (df["task_type"] == "Any2AnyRetrieval")
                & (df["modality"].isin(["a2t", "t2a"]))
            )
        ]
    )

    print(f"  - Audio-only tasks: {audio_count}")
    print(f"  - Audio-text cross-modal tasks: {audio_text_count}")

    print("Generating LaTeX tables...")

    # Generate tables
    audio_table = generate_audio_tasks_table(df)
    audio_text_table = generate_audio_text_tasks_table(df)

    # Combine into single file
    output_content = [
        "% MAEB+ Task Overview Tables",
        "% Auto-generated by generate_maeb_overview_tables.py",
        "",
        audio_table,
        audio_text_table,
    ]

    # Write to file
    output_file = "task_overview.tex"
    with open(output_file, "w") as f:
        f.write("\n".join(output_content))

    print(f"Successfully generated {output_file}")

    # Print summary statistics
    print("\nSummary:")
    print(f"  Total tasks: {len(df)}")
    print(f"  Tasks in lite benchmarks: {len(df[df['lite'] != ''])}")
    print(f"  Tasks in extended benchmarks: {len(df[df['extended'] != ''])}")

    # Count by task type
    print("\nTasks by type:")
    for task_type in sorted(df["task_type"].unique()):
        count = len(df[df["task_type"] == task_type])
        print(f"  {task_type}: {count}")


if __name__ == "__main__":
    main()
