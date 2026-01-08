from collections.abc import Iterable, Sequence

from mteb.abstasks import AbsTask
from mteb.benchmarks import Benchmark
from mteb.get_tasks import MTEBTasks


def _display_benchmarks(benchmarks: Sequence[Benchmark]) -> None:
    """Get all benchmarks available in the MTEB."""
    # get all the MTEB specific benchmarks:
    sorted_mteb_benchmarks = sorted(benchmarks, key=lambda obj: obj.name.lower())

    mteb_b, remaining_b = [], []
    for b in sorted_mteb_benchmarks:
        if "MTEB" in b.name or "MIEB" in b.name:
            mteb_b.append(b)
        else:
            remaining_b.append(b)

    # place mteb first, then remaining
    sorted_mteb_benchmarks = mteb_b + remaining_b

    # task ordering within each benchmark should be alphabetical
    for st in sorted_mteb_benchmarks:
        st.tasks = MTEBTasks(
            sorted(st.tasks, key=lambda obj: obj.metadata.name.lower())
        )

    for benchmark in sorted_mteb_benchmarks:
        name = benchmark.name
        _display_tasks(benchmark.tasks, name=name)


def _display_tasks(task_list: Iterable[AbsTask], name: str | None = None) -> None:
    from rich.console import Console

    console = Console()
    if name:
        console.rule(f"[bold]{name}\n", style="grey15")

    available_task_types = sorted({t.metadata.type for t in task_list})
    for task_type in available_task_types:
        current_type_tasks = list(
            filter(lambda x: x.metadata.type == task_type, task_list)
        )
        if len(current_type_tasks) == 0:
            continue
        else:
            console.print(f"[bold]{task_type}[/]")
            for task in current_type_tasks:  # will be sorted as input to this function
                prefix = "    - "
                name = f"{task.metadata.name}"
                category = f", [italic grey39]{task.metadata.category}[/]"
                multilingual = (
                    f", [italic red]multilingual {len(task.hf_subsets)} / {len(task.metadata.eval_langs)} Subsets[/]"
                    if task.metadata.is_multilingual
                    else ""
                )
                console.print(f"{prefix}{name}{category}{multilingual}")
            console.print("\n")
