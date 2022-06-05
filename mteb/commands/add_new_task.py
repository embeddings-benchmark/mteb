import json
import os
import shutil
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List

try:
    from cookiecutter.main import cookiecutter

    _has_cookiecutter = True
except ImportError:
    _has_cookiecutter = False


class AddNewTaskCommand:
    @classmethod
    def register_subcommand(cls, parser: ArgumentParser):
        add_new_task_parser = parser.add_parser("add-new-task")
        add_new_task_parser.set_defaults(func=lambda args: cls(args))

    def __init__(self, args):
        self.args = args

    def run(self):
        if not _has_cookiecutter:
            raise ImportError(
                "Cookiecutter is not installed. Please install it with `pip install cookiecutter`"
            )

        path_to_mteb_root = Path(__file__).parent.parent.parent
        path_to_cookiecutter = path_to_mteb_root / "templates" / "adding_a_new_task"

        cookiecutter(str(path_to_cookiecutter))
        output_dir = [Path(f) for f in os.listdir() if f.startswith("cookie-template-")][0]

        if output_dir is None:
            raise ValueError(
                "Cookiecutter template not found. Please check the name of the template."
            )

        # Retrieve configuration
        with open(output_dir / "configuration.json", "r") as configuration_file:
            config = json.load(configuration_file)

        # move file "cookie-template-" + config['task_classname'] to "tasks"
        new_task_dir = path_to_mteb_root / "mteb" / "tasks" / config["type"]
        # create folders
        new_task_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(
            output_dir / (config["task_classname"] + ".py"),
            new_task_dir / (config["task_classname"] + ".py"),
        )
        print(f"Created new task: {new_task_dir / (config['task_classname'] + '.py')}")
        shutil.rmtree(output_dir)

        # import task class in __init__.py
        init_file = new_task_dir / "__init__.py"
        if not init_file.exists():
            # create file if doesnt exist
            init_file.touch()

        with open(new_task_dir / "__init__.py", "r+") as init_file:
            if config['task_classname'] not in init_file.read():
                init_file.write(f"from .{config['task_classname']} import *\n")

        # import task type in mteb/tasks/__init__.py
        with open(new_task_dir.parent / "__init__.py", "r+") as init_file:
            if config['type'] not in init_file.read():
                init_file.write(f"from .{config['type']} import *\n")
