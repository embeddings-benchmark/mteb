"""
entry point for the library
example call:
  pip install git+https://github.com/embeddings-benchmark/mteb-draft.git@packaging
  mteb --model average_word_embeddings_komninos \
       --tasks Banking77Classification EmotionClassification \
       --k 5 \
       --device 0 \
       --batch_size 32 \
       --seed 42 \
       --output_folder /tmp/mteb_output \
       --n_splits 5 \
       --samples_per_label 8 \
       --verbosity 3

  mteb create_task
"""

from argparse import ArgumentParser

from .add_new_task import AddNewTaskCommand
from .eval import EvalCommand

def main():
    parser = ArgumentParser("Mteb CLI tool", usage="mteb <command> [<args>]")
    commands_parser = parser.add_subparsers(help="mteb command helpers")

    # Register commands
    AddNewTaskCommand.register_subcommand(commands_parser)
    EvalCommand.register_subcommand(commands_parser)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # Run
    service = args.func(args)
    service.run()


if __name__ == "__main__":
    main()