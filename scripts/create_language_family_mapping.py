from __future__ import annotations

import json
from pathlib import Path

from pyglottolog.api import Glottolog, lls
from tqdm import tqdm

glottolog = Glottolog(
    "/home/ubuntu/isaac/work/glottolog"
)  # Download the Glottolog repository


def get_languages_with_iso_by_languoid(languoid, level=0, prev_fam=None):
    # Recursively gather all descendant languages with ISO codes
    if prev_fam is None:
        prev_fam = {}  # Start with a fresh dictionary for each top-level languoid

    if not isinstance(languoid, lls.Languoid):
        return

    for descendant in languoid.children:
        # Create a copy of `prev_fam` to avoid overwriting
        current_fam = prev_fam.copy()
        current_fam[f"level{level}"] = languoid.name

        if descendant.level.name == "language":  # Direct languages
            if descendant.iso:
                iso_key = descendant.iso
                if len(ISO2FAMILY.get(iso_key, {})) > len(current_fam):
                    continue
                ISO2FAMILY[iso_key] = current_fam
        elif descendant.level.name == "family":  # Subfamilies, recurse
            get_languages_with_iso_by_languoid(descendant, level + 1, current_fam)


all_languoids = list(glottolog.languoids())
with Path("language_family.json").open("r") as f:
    ISO2FAMILY = json.load(f)

for languoid in tqdm(all_languoids, total=len(all_languoids)):
    get_languages_with_iso_by_languoid(languoid)

ISO2FAMILY = dict(sorted(ISO2FAMILY.items()))

with Path("language_family.json").open("w") as f:
    json.dump(ISO2FAMILY, f, indent=3)
