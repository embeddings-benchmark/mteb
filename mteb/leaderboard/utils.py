import functools
import itertools
import json

import mteb


def get_languages():
    langs = [task.languages for task in mteb.get_tasks()]
    langs = itertools.chain.from_iterable(langs)
    return list(set(langs))


def get_task_types():
    return list(set(task.metadata.type for task in mteb.get_tasks()))


def get_domains():
    domains = [
        task.metadata.domains
        for task in mteb.get_tasks()
        if task.metadata.domains is not None
    ]
    domains = itertools.chain.from_iterable(domains)
    return list(set(domains))


def equal_criteria(c0, c1):
    keys = set(c0.keys()) | set(c1.keys())
    for key in keys:
        if set(c0.get(key, [])) != set(c1.get(key, [])):
            return False
    return True


def get_model_size_range() -> tuple[int, int]:
    model_metas = mteb.get_model_metas()
    sizes = [meta.n_parameters for meta in model_metas if meta.n_parameters is not None]
    if not len(sizes):
        return None, None
    return min(sizes), max(sizes)


def json_cache(function):
    """Caching decorator that can deal with anything json serializable"""
    cached_results = {}

    def wrapper(*args, **kwargs):
        key = json.dumps({"__args": args, **kwargs})
        if key in cached_results:
            return cached_results[key]
        result = function(*args, **kwargs)
        cached_results[key] = result
        return result

    return wrapper
