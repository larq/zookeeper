import inspect
import re
from ast import literal_eval
from typing import Any, Callable, Iterator, Sequence, Type, TypeVar

import click
import typeguard


# A sentinel class/object for missing default values.
class Missing:
    def __repr__(self):
        return "<missing>"


missing = Missing()


class ConfigurationError(Exception):
    pass


def warn(message: str) -> None:
    click.secho(f"WARNING: {message}", fg="yellow", err=True)


def is_component_class(cls: Type) -> bool:
    try:
        return inspect.isclass(cls) and "__component_name__" in cls.__dict__
    except AttributeError:
        return False


def is_component_instance(instance: Any) -> bool:
    return is_component_class(instance.__class__)


def is_factory_class(cls: Type) -> bool:
    return is_component_class(cls) and hasattr(cls, "__component_factory_return_type__")


def is_factory_instance(instance: Any) -> bool:
    return is_factory_class(instance.__class__)


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
    """A utility method to generate from closest to furthest each ancestor component
    instance with a field called `field_name`."""
    if include_instance:
        parent = instance
    else:
        parent = instance.__component_parent__
    while parent is not None:
        if field_name in parent.__component_fields__:
            yield parent
        parent = parent.__component_parent__


def type_check(value, expected_type) -> bool:
    """Check that the `value` satisfies type `expected_type`."""
    if is_factory_instance(value):
        # If `value` is a @factory instance, what's relevant is the return type
        # of the `build()` method: we want to check if the return type is a
        # sub-type of the expected type.
        try:
            # If they are both classes, this is easy...
            return issubclass(value.__component_factory_return_type__, expected_type)
        except TypeError:
            # ...but in general this might fail (`issubclass` can't be used with
            # subscripted generics in Python 3.7, or with any type from the
            # `typing` module in Python 3.6). If the check fails we should print
            # a warning and conservatively return `True`.
            warn(
                f"Unable to check that {value.__component_factory_return_type__} is a "
                f"sub-type of {expected_type}."
            )
            return True
    try:
        # typeguard.check_type requires a name as the first argument for their
        # error message, but we want to catch their error so we can pass an
        # empty string.
        typeguard.check_type("", value, expected_type)
    except TypeError:
        return False
    return True


T = TypeVar("T")


def wrap_in_callable(value: T) -> Callable[[], T]:
    def wrapper():
        return value

    return wrapper


def is_immutable(value: Any) -> bool:
    """Decide the immutability of `value`.

    Recurses a single level if `value` is a tuple, but does not recurse infinitely.
    """
    return (
        value is None
        or isinstance(value, (int, float, bool, str, frozenset))
        or (
            isinstance(value, tuple)
            and all(
                inner_value is None
                or isinstance(inner_value, (int, float, bool, str, frozenset))
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

    return click.prompt(
        f"\nNo value found for field '{field_name}' of type '{field_type}'. ",
        prompt_suffix="Please enter a value for this parameter:\n> ",
        value_proc=parse_value_from_string,
    )


def prompt_for_component_subclass(component_name: str, classes: Sequence[T]) -> T:
    """Prompt the user to choose a component subclass from `classes`."""

    choices = {cls.__qualname__: cls for cls in classes}
    names = sorted(list(choices.keys()))

    def process_index(string: str) -> int:
        try:
            index = int(string) - 1
        except ValueError:
            index = -1
        if 0 <= index < len(names):
            return index
        raise click.UsageError(f"Please enter a number between 1 and {len(names)}.")

    index = click.prompt(
        f"\nNo instance found for nested component '{component_name}'. Please choose "
        "from one of the following component subclasses to instantiate:\n"
        + "\n".join([f"{i + 1})  {o}" for i, o in enumerate(names)]),
        prompt_suffix="\n> ",
        value_proc=process_index,
    )
    return choices[names[index]]
