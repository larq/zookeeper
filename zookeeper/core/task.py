import inspect

import click

from zookeeper.core.cli import ConfigParam, cli
from zookeeper.core.component import component, configure
from zookeeper.core.utils import convert_to_snake_case


def task(cls):
    """A decorator which turns a class into a Zookeeper task, which is a Zookeeper
    method with an argument-less `run` method.

    Tasks are runnable through the CLI. Upon execution, the task is instantiated and all
    component fields are configured using configuration passed as CLI arguments of the
    form `field_name=field_value`, and then the `run` method is called.
    """
    cls = component(cls)

    if not (hasattr(cls, "run") and callable(cls.run)):
        raise TypeError("Classes decorated with @task must define a `run` method.")

    # Enforce argument-less `run`.

    call_args = inspect.signature(cls.run).parameters
    if len(call_args) > 1 or len(call_args) == 1 and "self" not in call_args:
        raise TypeError(
            "A @task class must define a `run` method taking no arguments except "
            f"`self`, which runs the task, but `{cls.__name__}.run` accepts arguments "
            f"{tuple(name for name in call_args)}."
        )

    # Register a CLI command to run the task.

    if convert_to_snake_case(cls.__name__) in (
        convert_to_snake_case(c) for c in cli.commands
    ):
        raise ValueError(
            f"Task naming conflict. Task with name '{cls.__name__}' (or similar) "
            "already registered. Note that the task name is the name of the class that "
            "the @task decorator is applied to."
        )

    @cli.command(cls.__name__, context_settings=dict(ignore_unknown_options=True))
    @click.option(
        "-i",
        "--interactive",
        is_flag=True,
        default=False,
        help="Interactively configure task.",
    )
    @click.argument("config", type=ConfigParam(), nargs=-1)
    def command(config, interactive):
        config = {k: v for k, v in config}
        task_instance = cls()
        configure(task_instance, config, interactive=interactive)
        task_instance.run()

    return cls
