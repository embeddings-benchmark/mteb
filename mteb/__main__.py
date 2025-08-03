from __future__ import annotations

from mteb.cli import build_cli


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()
    args.func(args)
