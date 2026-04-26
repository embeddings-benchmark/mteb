"""Generate the model list for the HuggingFace leaderboard space.

Outputs a Python file (to stdout) containing the complete list of model
names registered in MTEB. The leaderboard space imports this to know
which models are available.

Usage:
    python scripts/generate_leaderboard_models.py > models.py
"""

from __future__ import annotations

import mteb


def main():
    model_metas = mteb.get_model_metas()
    model_names = sorted({m.name for m in model_metas})

    print('"""Auto-generated list of models registered in MTEB."""')
    print()
    print("MODEL_NAMES = [")
    for name in model_names:
        print(f'    "{name}",')
    print("]")


if __name__ == "__main__":
    main()
