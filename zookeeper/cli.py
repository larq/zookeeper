from inspect import isclass

import re
import click
from zookeeper.job import Job
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
        except:
            self.fail(
                "configuration parameters must be of the form 'key=value', where "
                "the key contains only alpha-numeric characters, underscores, and "
                f"full-stops, but received '{str_value}'.",
                param,
                ctx,
            )

        try:
            value = parse_value_from_string(value)
        except:
            self.fail(
                f"unable to parse value of configuration parameter {str_value}. The "
                "only supported types are `int`, `float`, `str`, `None`, and "
                "lists/tuples of the above.",
                param,
                ctx,
            )

        return key, value


def add_job_to_cli(job_cls: type):
    """A decorator which adds a CLI command to run the Job."""

    if not isclass(job_cls) or not issubclass(job_cls, Job):
        raise ValueError(
            "The decorator `add_job_to_cli` can only be applied to `zookeeper.Job` "
            "subclasses."
        )

    job_name = convert_to_snake_case(job_cls.__name__)

    @cli.command(job_name)
    @click.argument("config", type=ConfigParam(), nargs=-1)
    @click.option("-i", "--interactive", is_flag=True, default=False)
    def command(config, interactive):
        config = {k: v for k, v in config}
        job_instance = job_cls()
        job_instance.configure(config, interactive=interactive)
        job_instance.run()

    return job_cls
