from __future__ import annotations

from pathlib import Path

import mteb

START_INSERT = "<!-- START TASK DESCRIPTION -->"
END_INSERT = "<!-- END TASK DESCRIPTION -->"

task_entry = """
#### {task_name}

{description}

**Dataset:** [`{dataset_name}`](https://huggingface.co/datasets/{dataset_name}) • **License:** {license} • [Learn more →]({reference})

| Task category | Score | Languages | Domains | Annotations Creators | Sample Creation |
|-------|-------|-------|-------|-------|-------|
| {task_category_string} ({task_category}) | {main_score} | {languages} | {domains} | {annotation_creators} | {sample_creation} |

"""

task_type_section = """
# {task_type}

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** {num_tasks} 

{tasks_md}
"""


def pretty_long_list(items: list[str], max_items: int = 5) -> str:
    if len(items) <= max_items:
        return ", ".join(items)
    return ", ".join(items[:max_items]) + f", ... ({len(items)})"


def task_category_to_string(category: str) -> str:
    """Convert task category to a more readable string,

    e.g.
    "t2t" -> "text to text",
    "i2t" -> "image to text"
    "it2t" -> "image, text to text"
    "t2c" -> "text to category"
    "a2t" -> "audio to text"
    """
    category = category.lower()
    parts = category.split("2")
    if len(parts) != 2:
        return category  # return as is if not in expected format
    input_modalities = parts[0]
    output_modalities = parts[1]
    modality_map = {
        "t": "text",
        "i": "image",
        "a": "audio",
        "v": "video",
        "c": "category",
    }
    input_strings = [modality_map.get(m, m) for m in input_modalities]
    output_strings = [modality_map.get(m, m) for m in output_modalities]
    return f"{', '.join(input_strings)} to {', '.join(output_strings)}"


def format_task_entry(task: mteb.AbsTask) -> str:
    description = task.metadata.description
    dataset_name = task.metadata.dataset["path"]
    license = task.metadata.license or "not specified"
    reference = (
        task.metadata.reference or f"https://huggingface.co/datasets/{dataset_name}"
    )
    main_score = task.metadata.main_score
    task_category = task.metadata.category
    task_category_string = task_category_to_string(task_category)
    languages = pretty_long_list(task.metadata.languages)
    domains = (
        pretty_long_list(sorted(task.metadata.domains))
        if task.metadata.domains
        else "not specified"
    )
    annotation_creators = task.metadata.annotations_creators or "not specified"
    sample_creation = task.metadata.sample_creation or "not specified"

    return task_entry.format(
        task_name=task.metadata.name,
        description=description,
        dataset_name=dataset_name,
        license=license,
        reference=reference,
        main_score=main_score,
        task_category=task_category,
        task_category_string=task_category_string,
        languages=languages,
        domains=domains,
        annotation_creators=annotation_creators,
        sample_creation=sample_creation,
    )


def insert_between_markers(
    content: str,
    insert: str,
    start_marker: str = START_INSERT,
    end_marker: str = END_INSERT,
) -> str:
    """Insert `insert` between the `start_marker` and `end_marker` in `content`. Delete any content in between.

    Keeps the markers.
    """
    start_idx = content.index(start_marker) + len(start_marker)
    end_idx = content.index(end_marker)
    new_content = content[:start_idx] + "\n" + insert + "\n" + content[end_idx:]
    return new_content


def main(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)

    tasks = mteb.get_tasks(exclude_superseded=False, exclude_aggregate=True)
    task_types = sorted({task.metadata.type for task in tasks})

    task_types2tasks = {task_type: [] for task_type in task_types}
    for task in tasks:
        task_types2tasks[task.metadata.type].append(task)

    for task_type, tasks in task_types2tasks.items():
        _task_entries = ""
        for task in sorted(tasks, key=lambda t: t.metadata.name):
            _task_entries += format_task_entry(task) + "\n"
        md = task_type_section.format(
            task_type=task_type,
            num_tasks=len(tasks),
            tasks_md=_task_entries.strip(),
        )
        doc_task = folder / f"{task_type.lower()}.md"

        with doc_task.open("w") as f:
            f.write(md)


if __name__ == "__main__":
    root = Path(__file__).parent / ".." / ".."
    main(root / "docs" / "overview" / "available_tasks")
