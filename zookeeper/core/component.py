"""
Components are generic, modular classes designed to be easily configurable.

Components have configurable fields, which can contain either generic Python
objects or nested sub-components. These are declared with class-level Python
type annotations, in the same way that fields of
[dataclasses](https://docs.python.org/3/library/dataclasses.html) are declared.
After instantiation, components are 'configured' with a configuration
dictionary; this process automatically injects the correct field values into the
component and all sub-components. Component fields can have defaults set, either
in the class definition or passed via `__init__`, but field values passed to
`configure` will always take precedence over these default values.

If a nested sub-component declares a field with the same name as a field in one
of its ancestors, it will receive the same configured field value as the parent
does. Howevever, configuration is scoped: if the field on the child, or on a
_closer anscestor_, is configured with a different value, then that value will
override the one from the original parent.

Configuration can be interactive. In this case, the method will prompt for
missing fields via the CLI.

The following example illustrates the configuration mechanism with and
configuration scoping:

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

    YELLOW, GREEN, RED, RESET = Fore.YELLOW, Fore.GREEN, Fore.RED, Fore.RESET
except ImportError:  # pragma: no cover
    YELLOW = GREEN = RED = RESET = ""


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


#####################
# Component fields. #
#####################


class Field:
    def __init__(self, annotated_type):
        self.annotated_type = annotated_type


class EmptyField(Field):
    pass


class InheritedField(Field):
    def __init__(self, annotated_type, is_overriden=False):
        super().__init__(annotated_type)
        self.is_overriden = is_overriden


class NonEmptyField(Field):
    def __init__(self, annotated_type, is_overriden):
        super().__init__(annotated_type)
        self.is_overriden = is_overriden


class EmptyFieldError(AttributeError):
    def __init__(self, component, field_name):
        message = (
            f"The component `{component.__component_name__}` has no default or "
            f"configured value for field `{field_name}`. Please configure the "
            "component to provide a value."
        )
        super().__init__(message)


# Constants which are used internally during component configuration. They are
# used as placeholders to indicate to a nested sub-component that an ancestor
# component has a value for a given field name.
OVERRIDEN_CONF_VALUE = object()
NON_OVERRIDEN_CONF_VALUE = object()


def set_field_value(instance, name, value):
    assert not instance.__component_configured__
    assert name in instance.__component_fields__
    field = instance.__component_fields__[name]

    if value == OVERRIDEN_CONF_VALUE:
        instance.__component_fields__[name] = InheritedField(
            annotated_type=field.annotated_type, is_overriden=True
        )
    elif value == NON_OVERRIDEN_CONF_VALUE:
        if isinstance(field, EmptyField):
            instance.__component_fields__[name] = InheritedField(
                annotated_type=field.annotated_type, is_overriden=False
            )
    else:
        object.__setattr__(instance, name, value)
        instance.__component_fields__[name] = NonEmptyField(
            annotated_type=field.annotated_type, is_overriden=False
        )


####################################
# Component class method wrappers. #
####################################


def init_wrapper(init_fn):
    # Components need to be instantiable without arguments, so check that
    # `init_fn` does not accept any positional arguments without default values.
    for i, (name, param) in enumerate(inspect.signature(init_fn).parameters.items()):
        if (
            i > 0
            and param.default == inspect.Parameter.empty
            and param.kind
            not in [inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD]
        ):
            raise ValueError(
                "The `__init__` method of a component must not accept any "
                "positional arguments, as the component configuration process "
                "requires component classes to be instantiable without arguments."
            )

    def __component_init__(instance, **kwargs):
        # Fake 'super' call.
        if init_fn != object.__init__:
            init_fn(instance, **kwargs)

        instance.__component_fields__ = {}

        # Populate `__component_fields__` with all annotations set on this class
        # and all superclasses. We have to go through the MRO chain and collect
        # them in reverse order so that they are correctly overriden.
        annotations = {}
        for base_class in reversed(inspect.getmro(instance.__class__)):
            annotations.update(getattr(base_class, "__annotations__", {}))
        instance.__component_fields__ = {}
        for name, annotated_type in annotations.items():
            if name in object.__dir__(instance):  # type: ignore
                instance.__component_fields__[name] = NonEmptyField(
                    annotated_type, is_overriden=False
                )
            else:
                instance.__component_fields__[name] = EmptyField(annotated_type)

        if init_fn == object.__init__:
            for name, value in kwargs.items():
                if name in instance.__component_fields__:
                    set_field_value(instance, name, value)
                else:
                    raise ValueError(
                        f"Argument '{name}' does not correspond to any annotated field "
                        f"of '{type_name_str(instance.__class__)}'."
                    )

    return __component_init__


def dir_wrapper(dir_fn):
    def __component_dir__(instance):
        return set(dir_fn(instance)) | set(instance.__component_fields__.keys())

    return __component_dir__


def getattribute_wrapper(getattr_fn):
    def __component_getattr__(instance, name):
        component_fields = object.__getattribute__(instance, "__component_fields__")
        if name in component_fields:
            field = component_fields[name]
            if isinstance(field, EmptyField):
                raise EmptyFieldError(instance, name)
            if isinstance(field, NonEmptyField):
                return object.__getattribute__(instance, name)
            if isinstance(field, InheritedField):
                return getattr(instance.__component_parent__, name)
        else:
            return getattr_fn(instance, name)
        raise AttributeError

    return __component_getattr__


def setattr_wrapper(setattr_fn):
    def __component_setattr__(instance, name, value):
        if name in instance.__component_fields__:
            raise ValueError(
                "Setting component field values directly is prohibited. Use Zookeeper "
                "component configuration to set field values."
            )
        else:
            return setattr_fn(instance, name, value)

    return __component_setattr__


def delattr_wrapper(delattr_fn):
    def __component_delattr__(instance, name):
        if name in instance.__component_fields__:
            raise ValueError("Deleting component fields is prohibited.")
        return delattr_fn(instance, name)

    return __component_delattr__


##################################
# Pretty string representations. #
##################################


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
        f"{YELLOW}{key}{RESET}{space}={space}{YELLOW}{value}{RESET}"
        if color
        else f"{key}{space}={space}{value}"
    )


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


#######################
# Exported functions. #
#######################


def component(cls):
    """A decorater which turns a class into a Zookeeper component."""

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
    cls.__component_parent__ = None
    cls.__component_configured__ = False
    cls.__component_fields__ = {}

    # Override `__getattribute__`, `__setattr__`, and `__delattr__` to correctly
    # manage getting, setting, and deleting component fields.
    cls.__getattribute__ = getattribute_wrapper(cls.__getattribute__)  # type: ignore
    cls.__setattr__ = setattr_wrapper(cls.__setattr__)
    cls.__delattr__ = delattr_wrapper(cls.__delattr__)

    # Override `__dir__` so that field names are included.
    cls.__dir__ = dir_wrapper(cls.__dir__)

    # Override `__init__` to perform component initialisation and (potentially)
    # set key-word args as field values.
    cls.__init__ = init_wrapper(cls.__init__)

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
    for field_name, field in instance.__component_fields__.items():
        full_name = f"{instance.__component_name__}.{field_name}"
        field_type_name = (
            field.annotated_type.__name__
            if inspect.isclass(field.annotated_type)
            else str(field.annotated_type)
        )
        component_subclasses = list(generate_component_subclasses(field.annotated_type))

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

            set_field_value(instance, field_name, field_value)

            # If this is a 'raw' value, add a placeholder to `conf` so that it
            # gets picked up correctly in sub-components.
            if (
                field_value != OVERRIDEN_CONF_VALUE
                and field_value != NON_OVERRIDEN_CONF_VALUE
            ):
                conf[field_name] = OVERRIDEN_CONF_VALUE

        # If there's no config value but a value is already set on the instance,
        # we only need to add a placeholder to `conf` to make sure that the
        # value will be accessible from sub-components. `hasattr` isn't safe so
        # we have to check membership directly.
        elif field_name in object.__dir__(instance):  # type: ignore
            conf[field_name] = NON_OVERRIDEN_CONF_VALUE

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

            set_field_value(instance, field_name, field_value)

            # Add a placeholder to `conf` to so that this value can be accessed
            # by sub-components.
            conf[field_name] = NON_OVERRIDEN_CONF_VALUE

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
                field_value = prompt_for_value(full_name, field.annotated_type)

            set_field_value(instance, field_name, field_value)

            # Add a placeholder to `conf` so that this value can be accessed by
            # sub-components.
            conf[field_name] = OVERRIDEN_CONF_VALUE

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
            # Set the component parent so that inherited fields function
            # correctly.
            field_value.__component_parent__ = instance

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
    for field_name, field in instance.__component_fields__.items():
        assert field_name in instance.__component_fields__
        field_value = getattr(instance, field_name)
        try:
            check_type(field_name, field_value, field.annotated_type)
            # Because boolean `True` and `False` are coercible to ints and
            # floats, `typeguard.check_type` doesn't throw if we e.g. pass
            # `True` to a value expecting a float. This would, however, likely
            # be a user error, so explicitly check for this.
            if field.annotated_type in [float, int] and isinstance(field_value, bool):
                raise TypeError
        except TypeError:
            raise TypeError(
                f"Attempting to set field '{instance.__component_name__}.{field_name}' "
                f"which has annotated type '{type_name_str(field.annotated_type)}' "
                f"with value '{field_value}'."
            ) from None

    instance.__component_configured__ = True
