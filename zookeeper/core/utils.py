import inspect
import re
from ast import literal_eval
from typing import Any, Callable, Iterator, Sequence, Type, TypeVar

from prompt_toolkit import print_formatted_text, prompt


def is_component_class(cls: Type) -> bool:
    try:
        return inspect.isclass(cls) and "__component_name__" in cls.__dict__
    except AttributeError:
        return False


def is_component_instance(instance: Any) -> bool:
    return is_component_class(instance.__class__)


def generate_subclasses(cls: Type) -> Iterator[Type]:
    """Recursively find subclasses of `cls`."""

    if not inspect.isclass(cls):
        return
    yield cls
    for s in cls.__subclasses__():
        yield from generate_subclasses(s)


def generate_component_subclasses(cls: Type) -> Iterator[Type]:
    """Find component subclasses of `cls`."""

    for subclass in generate_subclasses(cls):
        if is_component_class(subclass) and not inspect.isabstract(subclass):
            yield subclass


def generate_component_ancestors_with_field(
    instance: Any, field_name, include_instance: bool = False
) -> Iterator[Any]:
    """
    A utility method to generate from closest to furthest each ancestor
    component instance with a field called `field_name`.
    """
    if include_instance:
        parent = instance
    else:
        parent = instance.__component_parent__
    while parent is not None:
        if field_name in parent.__component_fields__:
            yield parent
        parent = parent.__component_parent__


T = TypeVar("T")


def wrap_in_callable(value: T) -> Callable[[], T]:
    def wrapper():
        return value

    return wrapper


def is_immutable(value: Any) -> bool:
    """
    Decide the immutability of `value`. Recurses a single level if `value` is a
    set or a tuple, but does not recurse infinitely.
    """
    return (
        value is None
        or isinstance(value, (int, float, bool, str, frozenset))
        or (
            isinstance(value, (set, tuple))
            and all(
                isinstance(inner_value, (int, float, bool, str, frozenset))
                for inner_value in value
            )
        )
    )


def type_name_str(type) -> str:
    try:
        if hasattr(type, "__qualname__"):
            return str(type.__qualname__)
        if hasattr(type, "__name__"):
            return str(type.__name__)
        return str(type)
    except Exception:
        return "<unknown type>"


def convert_to_snake_case(name: str) -> str:
    s = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    s = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s)
    return re.sub(r"__+", "_", s).lower()


def parse_value_from_string(string: str) -> Any:
    try:
        value = literal_eval(string)
    except (ValueError, SyntaxError):
        # Parse as string if above raises a ValueError or SyntaxError.
        value = str(string)
    except Exception:
        raise ValueError(f"Could not parse '{string}'.")
    return value


def prompt_for_value(field_name: str, field_type) -> Any:
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
