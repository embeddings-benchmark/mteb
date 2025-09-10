"""Updates the available models markdown files."""

from __future__ import annotations

from pathlib import Path

import mteb

START_INSERT = "<!-- START TASK DESCRIPTION -->"
END_INSERT = "<!-- END TASK DESCRIPTION -->"

model_entry = """
####  `{model_name}`

**Revision:** `{revision}` • **License:** {license} • [Learn more →]({reference})


Max Tokens: {max_tokens}
Embedding dimension: {embed_dim}
Parameters: {n_parameters}
Release date: {release_date}
Languages: {languages}

| Max Tokens | Embedding dimension | Parameters | Release date | Languages |
|-------|-------|-------|-------|-------|
| {max_tokens} | {embed_dim} | {n_parameters} | {release_date} | {languages} |
"""

h1_header = """
# {modalities} Model

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of models:** {num_models} 

{models_md}
"""

h2_header = """
## {instruction_based}

{models_md}
"""


def human_readable_number(num: int) -> str:
    """Convert a number to a human-readable format with suffixes (K, M, B, T).

    E.g.
    1500 -> 1.5K
    2000000 -> 2M
    """
    for unit in ["", "K", "M", "B", "T"]:
        if abs(num) < 1000:
            return f"{num:.1f}{unit}" if unit else str(num)
        num /= 1000
    return f"{num:.2f}P"


def pretty_long_list(items: list[str], max_items: int = 5) -> str:
    if len(items) <= max_items:
        return ", ".join(items)
    return ", ".join(items[:max_items]) + f", ... ({len(items)})"


def modality_to_string(modality: tuple[str, ...]) -> str:
    return ("-".join(modality)).capitalize()


def format_model_entry(meta: mteb.ModelMeta) -> str:
    model_name = meta.name
    revision = meta.revision or "not specified"
    license = meta.license or "not specified"
    reference = meta.reference or f"https://huggingface.co/{model_name}"
    max_tokens = (
        human_readable_number(meta.max_tokens)
        if meta.max_tokens is not None
        else "not specified"
    )
    embed_dim = (
        human_readable_number(meta.embed_dim)
        if meta.embed_dim is not None
        else "not specified"
    )
    n_parameters = (
        human_readable_number(meta.n_parameters)
        if meta.n_parameters is not None
        else "not specified"
    )
    release_date = (
        meta.release_date if meta.release_date is not None else "not specified"
    )
    languages = (
        pretty_long_list(sorted(meta.languages)) if meta.languages else "not specified"
    )

    return model_entry.format(
        model_name=model_name,
        revision=revision,
        license=license,
        reference=reference,
        max_tokens=max_tokens,
        embed_dim=embed_dim,
        n_parameters=n_parameters,
        release_date=release_date,
        languages=languages,
    )


def main(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)

    metas = mteb.get_model_metas()
    modality_sets = sorted({tuple(sorted(model.modalities)) for model in metas})

    model_mod2models = {
        model_type: {"Instruction Model": [], "Non-instruction Model": []}
        for model_type in modality_sets
    }
    for meta in metas:
        if meta.use_instructions:
            model_mod2models[tuple(sorted(meta.modalities))][
                "Instruction Model"
            ].append(meta)
        else:
            model_mod2models[tuple(sorted(meta.modalities))][
                "Non-instruction Model"
            ].append(meta)

    for model_modalities, model_dict in model_mod2models.items():
        for instruction, metas in model_dict.items():
            if not metas:
                continue
            _model_entries = ""
            for meta in sorted(metas, key=lambda m: m.name):
                _model_entries += format_model_entry(meta) + "\n"
            md = h2_header.format(
                instruction_based=instruction,
                models_md=_model_entries.strip(),
            )
            modalities_string = modality_to_string(model_modalities)
            doc_task = folder / f"{modalities_string.lower().replace('-', '_')}.md"
            with doc_task.open("w") as f:
                f.write(
                    h1_header.format(
                        modalities=modalities_string,
                        num_models=len(metas),
                        models_md=md.strip(),
                    ).strip()
                )


if __name__ == "__main__":
    root = Path(__file__).parent / ".." / ".."
    main(root / "docs" / "overview" / "available_models")
