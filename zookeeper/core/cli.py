import re

import click

from zookeeper.core.utils import parse_value_from_string


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
