from __future__ import annotations

from multiprocessing import Pool, cpu_count

from tqdm import tqdm

import mteb


def process_task(task):
    task.calculate_metadata_metrics()


if __name__ == "__main__":
    tasks = mteb.get_tasks(task_types=["ImageClustering"])

    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap(process_task, tasks), total=len(tasks)))
