import re
from ast import literal_eval
from typing import Sequence, Type

from prompt_toolkit import print_formatted_text, prompt


def type_name_str(type) -> str:
    try:
        if hasattr(type, "__qualname__"):
            return str(type.__qualname__)
        if hasattr(type, "__name__"):
            return str(type.__name__)
        return str(type)
    except Exception:
        return "<unknown type>"


def convert_to_snake_case(name):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def parse_value_from_string(string: str):
    try:
        value = literal_eval(string)
    except (ValueError, SyntaxError):
        # Parse as string if above raises a ValueError or SyntaxError.
        value = str(string)
    except Exception:
        raise ValueError(f"Could not parse '{string}'.")
    return value


def prompt_for_value(field_name: str, field_type):
    """Promt the user to input a value for the parameter `field_name`."""

    print_formatted_text(
        f"No value found for field '{field_name}' of type '{field_type}'. "
        "Please enter a value for this parameter:"
    )
    response = prompt("> ")
    while response == "":
        print_formatted_text(f"No input received, please enter a value:")
        response = prompt("> ")
    return parse_value_from_string(response)


def prompt_for_component_subclass(component_name: str, classes: Sequence[Type]) -> Type:
    """Prompt the user to choose a compnent subclass from `classes`."""

    print_formatted_text(f"No instance found for nested component '{component_name}'.")
    choices = {cls.__qualname__: cls for cls in classes}
    names = sorted(list(choices.keys()))
    print_formatted_text(
        f"Please choose from one of the following component subclasses to instantiate:\n"
        + "\n".join([f"{i + 1})    {o}" for i, o in enumerate(names)])
    )
    response = prompt("> ")
    while True:
        try:
            response = int(response) - 1
        except ValueError:
            response = -1
        if 0 <= response < len(names):
            break
        print_formatted_text(
            f"Invalid input. Please enter a number between 1 and {len(names)}:"
        )
        response = prompt("> ")
    return choices[names[response]]
