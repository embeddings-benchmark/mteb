import re


def slugify_anchor(value: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"[\s]+", "-", slug)
