import importlib
import os
import sys


def validate_usr_dir(usr_dir):
    if (
        not os.path.isdir(os.path.join(usr_dir, "models"))
        or not os.path.isdir(os.path.join(usr_dir, "data"))
        or not os.path.isfile(os.path.join(usr_dir, "train.py"))
    ):
        raise ValueError(f"{usr_dir} is not a valid larq-swarm directory.")


def import_all_files(usr_dir, module):
    with os.scandir(os.path.join(usr_dir, module)) as it:
        for entry in it:
            name = entry.name
            if name.endswith(".py") and not name.startswith("_") and entry.is_file():
                importlib.import_module(module + "." + name[:-3])


def import_usr_dir(usr_dir):
    validate_usr_dir(usr_dir)
    usr_dir = os.path.abspath(os.path.expanduser(usr_dir).rstrip("/"))
    sys.path.insert(0, usr_dir)
    import_all_files(usr_dir, "models")
    import_all_files(usr_dir, "data")
    importlib.import_module("train")
    sys.path.pop(0)
