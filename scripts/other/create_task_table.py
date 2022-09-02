from mteb import MTEB

HEADER = "| Name | Hub URL | Description | Type | Category | N° Languages |"
SEP = "|:-----|:-----|:-----|:-----|:-----|-----:|"
ONE_LINE = "| {} | {} | {} | {} | {} | {} |"

TABLE_STRING = "\n".join([HEADER, SEP])

# Select all tasks
for task in MTEB().tasks:
    if "hf_hub_name" in task.description:
        hub_name = hub_url = task.description.get("hf_hub_name")
    elif "beir_name" in task.description:
        hub_name = "BeIR/" + task.description.get("beir_name")
        hub_url = "BeIR/cqadupstack-qrels" if "cqadupstack" in hub_name else hub_name
    TABLE_STRING += "\n" + ONE_LINE.format(
        f"[{task.description['name']}]({task.description['reference']})",
        f"[{hub_name}](https://huggingface.co/datasets/{hub_url})",
        task.description["description"],
        task.description["type"],
        task.description["category"],
        len(task.description["eval_langs"]),
    )

with open("./mdtable.md", "w") as f:
    f.write(TABLE_STRING)
