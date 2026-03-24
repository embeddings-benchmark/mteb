"""Updates the available tasks markdown files."""

from pathlib import Path
from typing import cast

from prettify_list import pretty_long_list
from slugify import slugify_anchor

import mteb
from mteb.abstasks.aggregated_task import AbsTaskAggregate
from mteb.abstasks.task_metadata import _TASKTYPE2SIMPLIFIEDTASKTYPE
from mteb.get_tasks import MTEBTasks

task_entry = """
#### `{task_name}` {{ .model-copy }}

{description}

{dataset_line}

| :lucide-tag: Category | :lucide-languages: Languages | :lucide-book-open: Domains | :lucide-users: Annotations | :lucide-plus-circle: Creation | :lucide-gauge: Score |
|:-:|:-:|:-:|:-:|:-:|:-:|
| {task_category_string} ({task_category}) | {languages} | {domains} | {annotation_creators} | {sample_creation} | {main_score} |

"""
aggregated_tasks_section = """
??? info "Tasks"

{task_table}
"""

task_type_section = """

<div class="nowrap-table" markdown>

## {task_type}

<!-- This document is auto-generated. Changes will be overwritten. Please change the generating script. -->

- **Number of tasks:** {num_tasks}

{tasks_md}
"""



citation_admonition = """

??? quote "Citation"

{citation_chunk}

"""

citation_chunk = """
```bibtex
{bibtex_citation}
```
"""


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


def create_aggregate_table(task: AbsTaskAggregate) -> str:
    tasks = cast("MTEBTasks", MTEBTasks(task.metadata.tasks))
    df = tasks.to_dataframe(["name", "type", "modalities", "languages"])
    df["name"] = df.apply(
        lambda row: (
            f"[{row['name']}](./{row['type'].lower()}.md#{slugify_anchor(row['name'])})"
        ),
        axis=1,
    )
    df["modalities"] = df["modalities"].apply(lambda x: pretty_long_list(x))
    df["languages"] = df["languages"].apply(lambda x: pretty_long_list(x))
    return df.to_markdown(index=False)


def format_task_entry(task: mteb.AbsTask) -> str:
    description = task.metadata.description
    if task.metadata.contributed_by:
        description += f" Contributed by {task.metadata.contributed_by}."
    license = task.metadata.license or "not specified"
    reference = task.metadata.reference
    dataset_name = task.metadata.dataset["path"]
    if not reference and not isinstance(task, AbsTaskAggregate):
        reference = f"https://huggingface.co/datasets/{dataset_name}"
    main_score = task.metadata.main_score
    task_category = task.metadata.category or "not specified"
    task_category_string = (
        task_category_to_string(task_category)
        if task.metadata.category
        else "not specified"
    )
    languages = pretty_long_list(task.metadata.languages)
    domains = (
        pretty_long_list(sorted(task.metadata.domains))
        if task.metadata.domains
        else "not specified"
    )
    annotation_creators = task.metadata.annotations_creators or "not specified"
    sample_creation = task.metadata.sample_creation or "not specified"

    if reference:
        learn_more = f" • [Learn more →]({reference})"
    else:
        learn_more = ""

    if not isinstance(task, AbsTaskAggregate):
        dataset_line = (
            f"**Dataset:** [`{dataset_name}`](https://huggingface.co/datasets/{dataset_name}) "
            f"• **License:** {license}{learn_more}"
        )
    else:
        dataset_line = f"**License:** {license}{learn_more}"

    entry = task_entry.format(
        task_name=task.metadata.name,
        description=description,
        dataset_line=dataset_line,
        main_score=main_score,
        task_category=task_category,
        task_category_string=task_category_string,
        languages=languages,
        domains=domains,
        annotation_creators=annotation_creators,
        sample_creation=sample_creation,
    )
    if task.metadata.bibtex_citation:
        citation = citation_chunk.format(bibtex_citation=task.metadata.bibtex_citation)
        citation = "\n".join([f"    {line}" for line in citation.split("\n")])  # indent
        entry += citation_admonition.format(citation_chunk=citation)

    if isinstance(task, AbsTaskAggregate):
        task_table = create_aggregate_table(task)
        task_table = "\n".join([f"    {line}" for line in task_table.split("\n")])
        entry += aggregated_tasks_section.format(task_table=task_table)

    return entry


def insert_content(doc_path: Path, content: str, tag: str) -> None:
    """Insert content into a markdown file between tags.

    If the file does not exist, it will be created with the content between the tags.
    If the file exists, the content between the tags will be replaced with the new content.
    """
    start_tag = f"<!-- START-{tag} -->"
    end_tag = f"<!-- END-{tag} -->"

    if doc_path.exists():
        doc_content = doc_path.read_text()
        if start_tag in doc_content and end_tag in doc_content:
            before = doc_content.split(start_tag)[0]
            after = doc_content.split(end_tag)[1]
            new_content = before + start_tag + "\n" + content + "\n" + end_tag + after
        else:
            # If tags are not found, append them at the end
            new_content = (
                doc_content + "\n\n" + start_tag + "\n" + content + "\n" + end_tag
            )
    else:
        # Create new file with content between tags
        new_content = start_tag + "\n" + content + "\n" + end_tag

    doc_path.write_text(new_content)


def main(folder: Path) -> None:
    folder.mkdir(exist_ok=True)

    tasks = mteb.get_tasks(
        exclude_superseded=False,
        exclude_aggregate=False,
        exclude_private=False,
    )

    task_types = sorted({task.metadata.type for task in tasks})
    task_types2tasks = {task_type: [] for task_type in task_types}
    for task in tasks:
        task_types2tasks[task.metadata.type].append(task)

    assert set(task_types2tasks.keys()) == set(_TASKTYPE2SIMPLIFIEDTASKTYPE.keys()), (
        f"Task types in tasks do not match expected task types. Found: {set(task_types2tasks.keys())}, expected: {set(_TASKTYPE2SIMPLIFIEDTASKTYPE.keys())}, difference: {set(task_types2tasks.keys()).symmetric_difference(set(_TASKTYPE2SIMPLIFIEDTASKTYPE.keys()))}"
    )

    stask_types = {stt: [] for stt in _TASKTYPE2SIMPLIFIEDTASKTYPE.values()}
    for tt, stt in _TASKTYPE2SIMPLIFIEDTASKTYPE.items():
        stask_types[stt].append(tt)

    for stt, tt in stask_types.items():
        # For each simplified task type, combine the markdown entries of the corresponding task types, each consisting of the tasks of that type.
        mds = []

        for tt in sorted(tt):
            tt_tasks = task_types2tasks[tt]
            if not tt_tasks:
                raise ValueError(f"No tasks found for task type {tt}")
            _task_entries = ""
            for task in sorted(tt_tasks, key=lambda t: t.metadata.name):
                _task_entries += format_task_entry(task) + "\n"

            md = task_type_section.format(
                task_type=tt,
                num_tasks=len(tt_tasks),
                tasks_md=_task_entries.strip(),
            )

            mds.append(md)

        doc_path = tasks_path / f"{stt}.md"
        doc_content = "\n\n".join(mds)
        insert_content(doc_path, doc_content, tag="TASKS")


if __name__ == "__main__":
    root = Path(__file__).parent
    tasks_path = root / "available_tasks"
    main(tasks_path)
