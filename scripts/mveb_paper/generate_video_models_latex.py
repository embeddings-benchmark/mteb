#!/usr/bin/env python3
"""Generate LaTeX table for all models with video modality support."""

import sys
from pathlib import Path

# Add root directory to path to import mteb
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mteb import get_model_metas


def format_modalities(modalities_list):
    """Format modalities list for LaTeX display with abbreviations."""
    abbreviations = {"audio": "a", "image": "i", "text": "t", "video": "v"}
    abbreviated = [abbreviations.get(mod, mod) for mod in sorted(modalities_list)]
    return ", ".join(abbreviated)


def format_parameters(n_params):
    """Format parameter count in millions."""
    if n_params is None:
        return "Unknown"
    return f"{n_params / 1_000_000:.1f}"


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


def generate_latex_table(video_models):
    """Generate LaTeX table for video models."""

    # Generate LaTeX table
    latex_content = []

    # Table header
    latex_content.extend(
        [
            "\\begin{table}[ht]",
            "\\centering",
            "\\caption{List of all models evaluated in MVEB. Model sizes are in millions of parameters. }",
            "\\label{tab:all-models}",
            "\\begin{tabular*}{\\textwidth}{l@{\\extracolsep{\\fill}}rp{2cm}}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{Parameters (M)} & \\textbf{Modalities} \\\\",
            "\\midrule",
        ]
    )

    # Table rows
    for model in video_models:
        model_name = model.name or "Unknown"

        # Skip citations for random encoders
        if "random" in model_name.lower():
            full_name = model_name
        else:
            citation_key = extract_citation_key(model.citation)
            full_name = f"{model_name} \\cite{{{citation_key}}}"

        param_count = format_parameters(model.n_parameters)
        modalities = format_modalities(model.modalities)

        # Escape underscores for LaTeX
        full_name = full_name.replace("_", "\\_")
        modalities = modalities.replace("_", "\\_")

        latex_content.append(f"{full_name} & {param_count} & {modalities} \\\\")

    # Table footer
    latex_content.extend(
        [
            "\\bottomrule",
            "\\end{tabular*}",
            "\\end{table}",
        ]
    )

    return "\n".join(latex_content)


def generate_bibliography(video_models):
    """Generate BibTeX bibliography from model citations."""

    seen_citations = set()
    bib_entries = []

    for model in video_models:
        if model.citation and model.citation.strip():
            # Clean up citation - remove extra whitespace
            citation = model.citation.strip()
            if not citation.endswith("}"):
                citation += "}"

            # Deduplicate by citation key
            citation_key = extract_citation_key(citation)
            if citation_key not in seen_citations:
                seen_citations.add(citation_key)
                bib_entries.append(citation)

    return "\n\n".join(bib_entries)


def main():
    """Main function."""
    output_file = Path(__file__).parent / "video_models_table.tex"
    bib_file = Path(__file__).parent / "models.bib"

    try:
        # Get video models
        video_models = get_model_metas(modalities=["video"])
        video_models = sorted(video_models, key=lambda m: m.name or "")

        print(f"Found {len(video_models)} models with video support")

        # Generate LaTeX table
        latex_content = generate_latex_table(video_models)

        # Generate bibliography
        bib_content = generate_bibliography(video_models)

        # Write LaTeX table
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(latex_content)

        # Write bibliography
        with open(bib_file, "w", encoding="utf-8") as f:
            f.write(bib_content)

        print(f"\nFiles generated:")
        print(f"LaTeX table: {output_file} ({output_file.stat().st_size} bytes)")
        print(f"Bibliography: {bib_file} ({bib_file.stat().st_size} bytes)")

        # Show preview of table
        lines = latex_content.split("\n")
        table_start = False
        preview_lines = []

        for line in lines:
            if "\\textbf{Model}" in line:
                table_start = True
                preview_lines.append(line)
                continue
            elif table_start and line.strip() and not line.startswith("\\"):
                preview_lines.append(line)
                if len(preview_lines) > 8:  # Show ~5 model rows
                    break
            elif table_start and line.startswith("\\"):
                preview_lines.append(line)

        print(f"\nPreview of LaTeX table:")
        for line in preview_lines:
            print(f"  {line}")

        if len(preview_lines) == 8:
            print("  ... (more rows)")

    except Exception as e:
        print(f"Error generating files: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
