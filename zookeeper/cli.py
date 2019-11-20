import re
from inspect import isclass

import click

from zookeeper.task import Task
from zookeeper.utils import convert_to_snake_case, parse_value_from_string


@click.group()
def cli():
    pass


class ConfigParam(click.ParamType):
    def convert(self, str_value, param, ctx):
        try:
            key, value = str_value.split("=")
            # Make sure the key is alpha-numeric (possibly with full stops).
            assert re.match("^[\\w.]+$", key)
        except Exception:
            self.fail(
                "configuration parameters must be of the form 'key=value', where "
                "the key contains only alpha-numeric characters, '_', and '.', "
                f"and the value doesn't contain '='. Received '{str_value}'.",
                param,
                ctx,
            )

        try:
            value = parse_value_from_string(value)
        except Exception:
            self.fail(
                f"unable to parse value of configuration parameter {str_value}. The "
                "only supported types are `int`, `float`, `str`, `None`, and "
                "lists/tuples of the above.",
                param,
                ctx,
            )

        return key, value


def add_task_to_cli(task_cls: type):
    """A decorator which adds a CLI command to run the Task."""

    if not isclass(task_cls) or not issubclass(task_cls, Task):
        raise ValueError(
            "The decorator `add_task_to_cli` can only be applied to `zookeeper.Task` "
            "subclasses."
        )

    task_name = convert_to_snake_case(task_cls.__name__)

    @cli.command(task_name)
    @click.argument("config", type=ConfigParam(), nargs=-1)
    @click.option("-i", "--interactive", is_flag=True, default=False)
    def command(config, interactive):
        config = {k: v for k, v in config}
        task_instance = task_cls()
        task_instance.configure(config, interactive=interactive)
        task_instance.run()

    return task_cls
