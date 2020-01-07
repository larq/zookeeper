import re

import click

from zookeeper.core.utils import convert_to_snake_case, parse_value_from_string


class ConfigParam(click.ParamType):
    def convert(self, str_value, param, ctx):
        # Allow the syntax `--name_of_key` to set the boolean flag to `True`,
        # and likewise `--no-name_of_key` to set the boolean flag to `False`.
        if re.match("^--[\\w.]+$", str_value):
            return str_value[2:], True
        if re.match("^--no-[\\w.]+$", str_value):
            return str_value[5:], False

        # Otherwise, expect the syntax `name_of_key=value`.
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


class CamelCaseGroup(click.Group):
    """Invoke commands with string that resolves to the same camel-case."""

    def get_command(self, ctx, cmd_name):
        cmd_name = convert_to_snake_case(cmd_name)
        for c in self.list_commands(ctx):
            if convert_to_snake_case(c) == cmd_name:
                return click.Group.get_command(self, ctx, c)
        return None


@click.group(cls=CamelCaseGroup)
def cli():
    pass
