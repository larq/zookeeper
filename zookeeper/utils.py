import re
from ast import literal_eval
from inspect import isabstract
from typing import Set

from prompt_toolkit import print_formatted_text, prompt


def convert_to_snake_case(name):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def get_concrete_subclasses(cls) -> Set[type]:
    """Return a set of all non-abstract classes which inherit from `cls`."""
    subclasses = set([cls] if not isabstract(cls) else [])
    for s in cls.__subclasses__():
        if not isabstract(s):
            subclasses.add(s)
        subclasses.update(get_concrete_subclasses(s))
    return subclasses


def parse_value_from_string(string: str):
    try:
        value = literal_eval(string)
    except (ValueError, SyntaxError):
        # Parse as string if above raises a ValueError or SyntaxError.
        value = str(string)
    except Exception:
        raise ValueError(f"Could not parse '{string}'.")
    return value


def promt_for_param_value(param_name: str, param_type):
    """Promt the user to input a value for the parameter `param_name`."""
    print_formatted_text(
        f"No value found for parameter '{param_name}' of type '{param_type}'. "
        "Please enter a value for this parameter:"
    )
    response = prompt("> ")
    while response == "":
        print_formatted_text(f"No input received, please enter a value:")
        response = prompt("> ")
    return parse_value_from_string(response)


def prompt_for_component(component_name: str, component_cls: type) -> type:
    print_formatted_text(
        f"No instance found for nested component '{component_name}' of type "
        f"'{component_cls.__qualname__}'."
    )

    component_options = {
        cls.__qualname__: cls for cls in get_concrete_subclasses(component_cls)
    }

    if len(component_options) == 0:
        raise ValueError(
            f"'{component_cls}' has no defined concrete subclass implementation."
        )

    component_names = sorted(list(component_options.keys()))

    print_formatted_text(
        f"Please choose from one of the following concrete subclasses of "
        f"'{component_cls.__qualname__}' to instantiate:\n"
        + "\n".join([f"{i + 1})    {o}" for i, o in enumerate(component_names)])
    )
    response = prompt("> ")
    while True:
        try:
            response = int(response) - 1
        except ValueError:
            response = -1
        if 0 <= response < len(component_names):
            break
        print_formatted_text(
            f"Invalid input. Please enter a number between 1 and {len(component_names)}:"
        )
        response = prompt("> ")

    return component_options[component_names[response]]
