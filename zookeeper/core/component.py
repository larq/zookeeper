import inspect
from typing import Any, Dict, Optional

from prompt_toolkit import print_formatted_text
from typeguard import check_type

from zookeeper.core.utils import (
    convert_to_snake_case,
    prompt_for_component_subclass,
    prompt_for_value,
    type_name_str,
)

try:  # pragma: no cover
    from colorama import Fore

    BLUE, YELLOW, RESET = Fore.BLUE, Fore.YELLOW, Fore.RESET
except ImportError:  # pragma: no cover
    BLUE = YELLOW = RESET = ""

# Indent for nesting in the string representation
INDENT = " " * 4


def str_key_val(key, value, color=True, single_line=False):
    if is_component_class(value.__class__):
        if single_line:
            value = repr(value)
        else:
            value = f"\n{INDENT}".join(str(value).split("\n"))
    elif callable(value):
        value = "<callable>"
    elif type(value) == str:
        value = f'"{value}"'
    space = "" if single_line else " "
    return (
        f"{BLUE}{key}{RESET}{space}={space}{YELLOW}{value}{RESET}"
        if color
        else f"{key}{space}={space}{value}"
    )


def is_component_class(cls):
    try:
        return "__component_name__" in cls.__dict__
    except AttributeError:
        return False


def generate_subclasses(cls):
    """Recursively find subclasses of `cls`."""

    if not inspect.isclass(cls):
        return
    yield cls
    for s in cls.__subclasses__():
        yield from generate_subclasses(s)


def generate_component_subclasses(cls):
    """Find component subclasses of `cls`."""

    for subclass in generate_subclasses(cls):
        if is_component_class(subclass) and not inspect.isabstract(subclass):
            yield subclass


def __component_repr__(instance):
    fields = ", ".join(
        [
            str_key_val(
                field_name, getattr(instance, field_name), color=False, single_line=True
            )
            for field_name in sorted(instance.__component_fields__)
        ]
    )
    return f"{instance.__class__.__name__}({fields})"


def __component_str__(instance):
    fields = f",\n{INDENT}".join(
        [
            str_key_val(field_name, getattr(instance, field_name))
            for field_name in sorted(instance.__component_fields__)
        ]
    )
    return f"{instance.__class__.__name__}(\n{INDENT}{fields}\n)"


def __component_init__(instance, **kwargs):
    for name, value in kwargs.items():
        if name in instance.__component_fields__:
            setattr(instance, name, value)
        else:
            raise ValueError(
                f"Argument '{name}' does not correspond to any annotated field "
                f"of '{type_name_str(instance.__class__)}'."
            )


class EmptyFieldError(AttributeError):
    def __init__(self, component, field_name):
        message = (
            f"The component `{component.__component_name__}` has no default or "
            f"configured value for field `{field_name}`. Please configure the "
            "component to provide a value."
        )
        super().__init__(message)


class InheritedFieldValue:
    def __init__(self, ancestor, is_overriden):
        self.ancestor = ancestor
        self.is_overriden = is_overriden


def getattribute_wrapper(getattribute_fn):
    """
    Get a value for the field `name` on an instance or a component ancestor,
    if the value is inherited.
    """

    def wrapped_fn(instance, name):
        if name in object.__getattribute__(instance, "__component_fields__"):
            # We can't use `hasattr`, because that would call `__getattribute__`
            # and cause infinite recursion.
            if name not in dir(instance):
                raise EmptyFieldError(instance, name)
            value = object.__getattribute__(instance, name)
            if isinstance(value, InheritedFieldValue):
                value = getattr(value.ancestor, name)
            return value
        else:
            return getattribute_fn(instance, name)

    return wrapped_fn


def component(cls):
    """
    A decorater which turns a class into a Zookeeper component. Components are
    generic, modular classes designed to be easily configurable.

    Components can have configurable fields, which can contain either generic
    Python objects or nested sub-components. These are declared with class-level
    Python type annotations, in the same way that fields of
    [dataclasses](https://docs.python.org/3/library/dataclasses.html) are
    declared. After instantiation, components are 'configured' with a
    configuration dictionary; this process automatically injects the correct
    field values into the component and all subcomponents. Component fields can
    have defaults set, either in the class definition or passed via `__init__`,
    but configuration values passed to `configure` will always take precedence
    over these values.

    If a nested sub-component child declares a field with the same name as a
    field in one of its ancestors, it will receive the same configured field
    value as the parent does. Howevever, configuration is scoped: if the field
    on the child, or on a _closer anscestor_, is configured with a different
    value, then that value will override the one from the original parent.

    Configuration can be interactive. In this case, the method will prompt for
    missing fields via the CLI.

    The following example illustrates the configuration mechanism with scoped
    configuration:

    ```
    @component
    class A:
        x: int
        z: float

    @component
    class B:
        a: A
        y: str = "foo"

    @component
    class C:
        b: B
        x: int
        z: float = 3.14

    c = C()
    configure(
        c,
        {
            "x": 5,                     # (1)
            "b.x": 10,                  # (2)
            "b.a.x": 15,                # (3)

            "b.y": "foo",               # (4)

            "b.z": 2.71                 # (5)
        }
    )
    print(c)

    >>  C(
            b = B(
                a = A(
                    x = 15,             # (3) overrides (2) overrides (1)
                    z = 2.71            # Inherits from parent: (5)
                ),
                y = "foo"               # (4) overrides the default
            ),
            x = 5,                      # Only (1) applies
            z = 3.14                    # The default is taken
        )
    ```
    """

    if not inspect.isclass(cls):
        raise ValueError("Only classes can be decorated with @component.")

    if inspect.isabstract(cls):
        raise ValueError("Abstract classes cannot be decorated with @component.")

    if is_component_class(cls):
        raise ValueError(
            f"The class {cls} is already a component; the @component decorator cannot "
            "be applied again."
        )

    cls.__component_name__ = convert_to_snake_case(cls.__name__)
    cls.__component_configured__ = False

    # Populate `__component_fields__` with all annotations set on this class and
    # all superclasses. We have to go through the MRO chain and collect them in
    # reverse order so that they are correctly overriden.
    cls.__component_fields__ = {}
    for base_class in reversed(inspect.getmro(cls)):
        cls.__component_fields__.update(getattr(base_class, "__annotations__", {}))
    cls.__component_fields__.update(getattr(cls, "__annotations__", {}))

    # Override `__getattribute__` to allow components to get field values from
    # ancestors.
    cls.__getattribute__ = getattribute_wrapper(cls.__getattribute__)  # type: ignore

    if cls.__init__ != object.__init__:
        # If the class overrides `__init__`, we check that `__init__` does not
        # accept any positional arguments.
        for i, (name, param) in enumerate(
            inspect.signature(cls.__init__).parameters.items()
        ):
            if (
                i > 0
                and param.default == inspect.Parameter.empty
                and param.kind
                not in [inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD]
            ):
                raise ValueError(
                    "The `__init__` method of a component must not accept any "
                    "positional arguments, as the component configuration process "
                    "requires component classes to be instantiable without arguments. "
                    f"Please set a default value for the parameter '{name}' of "
                    f"`{type_name_str(cls)}.__init__`."
                )
    else:
        # Otherwise, we set an `__init__` which will set any field values passed
        # in as keyword args.
        cls.__init__ = __component_init__

    # Components should have nice `__str__` and `__repr__` methods.
    cls.__str__ = __component_str__
    cls.__repr__ = __component_repr__

    return cls


def configure(
    instance,
    conf: Dict[str, Any],
    name: Optional[str] = None,
    interactive: bool = False,
):
    """
    Configure the component instance with parameters from the `conf` dict.

    Configuration passed through `conf` takes precedence over and will
    overwrite any values already set on the instance - either class defaults
    or those set in `__init__`.
    """

    # Configuration can only happen once.
    if instance.__component_configured__:
        raise ValueError(
            f"Component '{instance.__component_name__}' has already been configured."
        )

    if name is not None:
        instance.__component_name__ = name

    # Set the correct value for each field.
    for field_name, field_type in instance.__component_fields__.items():
        full_name = f"{instance.__component_name__}.{field_name}"
        field_type_name = (
            field_type.__name__ if inspect.isclass(field_type) else str(field_type)
        )
        component_subclasses = list(generate_component_subclasses(field_type))

        if field_name in conf:
            field_value = conf[field_name]
            # The configuration value could be a string specifying a component
            # class to instantiate.
            if len(component_subclasses) > 0 and isinstance(field_value, str):
                for subclass in component_subclasses:
                    if (
                        field_value == subclass.__name__
                        or field_value == subclass.__qualname__
                        or convert_to_snake_case(field_value)
                        == convert_to_snake_case(subclass.__name__)
                    ):
                        field_value = subclass()
                        break

            # The only scenario in which we don't set `field_value` on the
            # instance is if a value for `field_name` already exists and
            # `field_value` is a non-overriden `InheritedFieldValue`.
            if not (
                field_name in dir(instance)
                and isinstance(field_value, InheritedFieldValue)
                and not field_value.is_overriden
            ):
                setattr(instance, field_name, field_value)

            # We set a placeholder in `conf` which points any subcomponents to
            # `instance` if they inherit `field_name`.
            conf[field_name] = InheritedFieldValue(
                instance,
                is_overriden=(
                    # Either the field value is _not_ inherited (i.e. must
                    # be a configuration value being passed in directly)...
                    not isinstance(field_value, InheritedFieldValue)
                    # ...or it is inherited from an overriden value.
                    or field_value.is_overriden
                ),
            )

        # If there's no config value but a value is already set on the instance,
        # we don't need to do anything directly. `hasattr` isn't safe so we have
        # to check directly.
        elif field_name in dir(instance):
            # Add a placeholder to the `conf` dict to so this value can be
            # accessed by sub-components.
            conf[field_name] = InheritedFieldValue(instance, is_overriden=False)

        # If there is only one concrete component subclass of the annotated
        # type, we assume the user must intend to use that subclass, and so
        # instantiate and use an instance automatically.
        elif len(component_subclasses) == 1:
            component_cls = list(component_subclasses)[0]
            print_formatted_text(
                f"'{type_name_str(component_cls)}' is the only concrete component "
                f"class that satisfies the type of the annotated field '{full_name}'. "
                "Using an instance of this class by default."
            )
            # This is safe because we don't allow `__init__` to have any
            # positional arguments without defaults.
            field_value = component_cls()

            setattr(instance, field_name, field_value)

            # Add a placeholder to the `conf` dict to so this value can be
            # accessed by sub-components.
            conf[field_name] = InheritedFieldValue(instance, is_overriden=False)

        # If we are running interactively, prompt for a value.
        elif interactive:
            if len(component_subclasses) > 0:
                component_cls = prompt_for_component_subclass(
                    full_name, component_subclasses
                )
                # This is safe because we don't allow `__init__` to have any
                # positional arguments without defaults.
                field_value = component_cls()
            else:
                field_value = prompt_for_value(full_name, field_type)

            setattr(instance, field_name, field_value)

            # Add a placeholder to the `conf` dict to so this value can be
            # accessed by sub-components.
            conf[field_name] = InheritedFieldValue(instance, is_overriden=False)

        # Otherwise, raise an appropriate error.
        else:
            if len(component_subclasses) > 0:
                raise ValueError(
                    f"Annotated field '{full_name}' of type '{field_type_name}' "
                    f"has no configured value. Please configure '{full_name}' with "
                    f"one of the following component subclasses of '{field_type_name}':"
                    + "\n    ".join(
                        [""] + list(type_name_str(c) for c in component_subclasses)
                    )
                )
            raise ValueError(
                "No configuration value found for annotated field "
                f"'{full_name}' of type '{field_type_name}'."
            )

    # Recursively configure any sub-components.
    for field_name, field_type in instance.__component_fields__.items():
        field_value = getattr(instance, field_name)
        full_name = f"{instance.__component_name__}.{field_name}"
        if (
            is_component_class(field_value.__class__)
            and not field_value.__component_configured__
        ):
            # Configure the nested sub-component. The configuration we use
            # consists of all non-scoped keys and any keys scoped to
            # `field_name`, where the keys scoped to `field_name` override the
            # non-scoped keys.
            non_scoped_conf = {a: b for a, b in conf.items() if "." not in a}
            field_name_scoped_conf = {
                a[len(f"{field_name}.") :]: b
                for a, b in conf.items()
                if a.startswith(f"{field_name}.")
            }
            nested_conf = {**non_scoped_conf, **field_name_scoped_conf}
            configure(field_value, nested_conf, name=full_name, interactive=interactive)

    # Type check all fields.
    for field_name, field_type in instance.__component_fields__.items():
        assert field_name in instance.__component_fields__
        field_value = getattr(instance, field_name)
        try:
            check_type(field_name, field_value, field_type)
            # Because boolean `True` and `False` are coercible to ints and
            # floats, `typeguard.check_type` doesn't throw if we e.g. pass
            # `True` to a value expecting a float. This would, however,
            # likely be a user error, so explicitly check for this.
            if field_type in [float, int] and isinstance(field_value, bool):
                raise TypeError
        except TypeError:
            raise TypeError(
                f"Attempting to set field '{instance.__component_name__}.{field_name}' "
                f"which has annotated type '{type_name_str(field_type)}' with value "
                f"'{field_value}'."
            ) from None

    instance.__component_configured__ = True
