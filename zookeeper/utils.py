from ast import literal_eval
from inspect import isabstract
from typing import Set

from prompt_toolkit import print_formatted_text, prompt


def get_concrete_subclasses(cls) -> Set[type]:
    """Return a set of all non-abstract classes which inherit from `cls`."""
    subclasses = set([cls] if not isabstract(cls) else [])
    for s in cls.__subclasses__():
        if not isabstract(s):
            subclasses.add(s)
        subclasses.update(get_concrete_subclasses(s))
    return subclasses


def promt_for_param_value(param_name: str, param_type):
    """Promt the user to input a value for the parameter `param_name`."""
    print_formatted_text(
        f"No value found for parameter '{param_name}' of type '{param_type}'. Please enter a value for this parameter:"
    )
    response = prompt("> ")
    while response == "":
        print_formatted_text(f"No input received, please enter a value:")
        response = prompt("> ")
    try:
        value = literal_eval(response)
    except ValueError:
        # Parse as string if above raises ValueError. Note that
        # syntax errors will still raise an error.
        value = str(value)
    except:
        raise ValueError(f"Could not parse '{value}'")
    return value


def prompt_for_component(component_name: str, component_cls: type) -> type:
    print_formatted_text(
        f"No instance found for nested component '{component_name}' of type '{component_cls}'."
    )

    component_options = {
        cls.__name__: cls for cls in get_concrete_subclasses(component_cls)
    }

    if len(component_options) == 0:
        raise ValueError(
            f"'{component_cls}' has no defined concrete subclass implementation."
        )

    component_names = sorted(list(component_options.keys()))

    print_formatted_text(
        f"Please choose from one of the following concrete subclasses of '{component_cls}' to instantiate:"
        + "\n"
        + "\n".join([f"{i + 1})\t{o}" for i, o in enumerate(component_names)])
    )
    response = prompt("> ")
    while True:
        try:
            response = int(response) - 1
        except:
            response = -1
        if 1 <= response < len(component_names):
            break
        print_formatted_text(
            f"Invalid input. Please enter a number between 1 and {len(component_names)}:"
        )
        response = prompt("> ")

    return component_options[component_names[response]]
