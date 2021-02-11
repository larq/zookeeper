import inspect
from typing import Callable, Generic, Type, TypeVar, Union

from zookeeper.core import utils
from zookeeper.core.partial_component import PartialComponent
from zookeeper.core.utils import ConfigurationError

# Type-variables to parameterise fields. `C` is the type of the component the
# field is attached to, and `F` is the type the field is annotated with.
C = TypeVar("C")
F = TypeVar("F")


class Field(Generic[C, F]):
    """A configurable field for Zookeeper components. Fields must be typed, may take
    default values, and are configurable through the CLI.

    This class is not appropriate for nesting child sub-components; for this,
    use `ComponentField` instead.

    Internally, each component will cache the default value once it has been
    generated: `get_default` will be called at most once per instance.
    """

    def __init__(
        self,
        default: Union[
            utils.Missing, F, Callable[[], F], Callable[[C], F]
        ] = utils.missing,
        *,  # `allow_missing` must be a keyword argument.
        allow_missing: bool = False,
    ):
        # Define here once to avoid having to define twice below.
        default_error = TypeError(
            "If `default` is passed to `Field`, it must be either:\n"
            "- An immutable value (int, float, bool, string, or None).\n"
            "- A function or lambda accepting no arguments or a single\n"
            "  argument (`self`), and returning the default value.\n"
            f"Received: {default}."
        )

        self.name = None
        self.allow_missing = allow_missing
        self.host_component_class = None
        self.type = None
        self._registered = False
        self._return_annotation = inspect.Signature.empty

        if allow_missing and default is not utils.missing:
            raise ValueError(
                "If a `Field` has `allow_missing=True`, no default can be provided."
            )

        if default is utils.missing or utils.is_immutable(default):
            self._default = default
            return

        if inspect.isfunction(default):
            signature = inspect.signature(default)
            if len(signature.parameters) > 1 or (
                len(signature.parameters) == 1
                and list(signature.parameters.values())[0].kind
                in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            ):
                raise default_error

            self._default = default
            self._return_annotation = signature.return_annotation
            return

        raise default_error

    # This follows the PEP 487 `__set_name__` protocol; this method is called
    # automatically on every object defined within a class body, passing in the
    # class object and name of the descriptor. We use it here to obtain the name
    # and type of the field.
    def __set_name__(self, host_component_class: Type[C], name: str):
        if self._registered:
            raise ValueError("This field has already been registered to a component.")
        if name.startswith("_"):
            raise ValueError("Field names cannot start with underscores.")

        type_annotation = utils.missing
        for super_class in inspect.getmro(host_component_class):
            if name in getattr(super_class, "__annotations__", {}):
                type_annotation = super_class.__annotations__[name]
                break

        if (
            self._return_annotation is not inspect.Signature.empty
            and type_annotation != utils.missing
            and type_annotation != self._return_annotation
        ):
            raise TypeError(
                f"Two non-equal type annotations found for field '{name}': "
                f"{type_annotation} and {self._return_annotation}."
            )

        if type_annotation is utils.missing:
            if self._return_annotation is inspect.Signature.empty:
                raise TypeError(
                    "Fields must be defined inside the component class definition, "
                    "with a type annotation in one of the following ways:\n\n"
                    "```\n"
                    "@component\n"
                    "class ComponentClass:\n"
                    "    ...\n"
                    "    # Like this\n"
                    "    name_1: type_1 = Field(default_1)\n"
                    "    ...\n"
                    "    # Or like this\n"
                    "    @Field\n"
                    "    def name_2(self) -> type_2:\n"
                    "        ...\n"
                    "        return default_2\n"
                    "```\n\n"
                    f"Unable to find a type annotation for field '{name}' on class "
                    f"'{host_component_class.__name__}'."
                )
            type_annotation = self._return_annotation

        self.name = name
        self.host_component_class = host_component_class
        self.type = type_annotation
        self._registered = True

    def __repr__(self) -> str:
        if not self._registered:
            return "<Unregistered Field>"
        return (
            f"<Field '{self.name}' of {self.host_component_class.__name__} with type "
            f"{self.type}>"
        )

    @property
    def has_default(self) -> bool:
        if not self._registered:
            raise ValueError("This field has not been registered to a component.")
        return self._default is not utils.missing

    def get_default(self, instance: C) -> F:
        if not self._registered:
            raise ValueError("This field has not been registered to a component.")
        if not self.has_default:
            msg = f"Field '{self.name}' has no default or configured value."
            # If this field is allow_missing, we raise an AttributeError, since it
            # doesn't really exist.
            if self.allow_missing:
                raise AttributeError(msg)
            # If it isn't allow_missing, this is a configuration error, because the
            # attribute does exist, but has no value.
            raise ConfigurationError(msg)
        if not isinstance(instance, self.host_component_class):
            raise TypeError(
                f"Field '{self.name}' belongs to component "
                f"'{self.host_component_class.__name__}'; `get_default` must be called "
                f"with an instance of '{self.host_component_class.__name__}'. Received: "
                f"{repr(instance)}."
            )

        if not inspect.isfunction(self._default):
            return self._default

        params = inspect.signature(self._default).parameters
        if len(params) == 0:
            value = self._default()  # type: ignore
        else:
            value = self._default(instance)  # type: ignore

        if utils.is_component_instance(value):
            raise TypeError(
                f"Field '{self.name}' of component '{instance.__component_name__}' "
                "is returning a component instance as its default value. To use "
                "components in fields, use `ComponentField` rather than `Field`."
            )

        return value


# TODO: Maybe add lower-case alias `field`?


class ComponentField(Field, Generic[C, F]):
    """A Zookeeper field for nesting child sub-components.

    `ComponentField`s must be defined with a type annotation: a Python class
    from which all possible component instances for the field must inherit.

    A default component class may also be provided, which will be used to
    instantiate a value if there is no configured value for this field. If a
    default component class is provided, keyword arguments corresponding to the
    field names of the default component may optionally also be provided to be
    used to initialise the default component.
    """

    def __init__(
        self,
        default: Union[utils.Missing, F, PartialComponent[F]] = utils.missing,
        *,  # `allow_missing` must be a keyword argument.
        allow_missing: bool = False,
        **kwargs,
    ):
        if allow_missing and default is not utils.missing:
            raise ValueError(
                "If a `Field` has `allow_missing=True`, no default can be provided."
            )

        if default is utils.missing:
            if len(kwargs) > 0:
                raise TypeError(
                    "Keyword arguments can only be passed to `ComponentField` if "
                    "a default component class is also passed."
                )
        elif isinstance(default, PartialComponent) or utils.is_component_class(default):
            if len(kwargs) > 0:
                default = PartialComponent(default, **kwargs)
        elif utils.is_component_instance(default):
            raise TypeError(
                "The `default` passed to `ComponentField` must be a component class, "
                f"not a component instance. Received: {repr(default)}."
            )
        else:
            raise TypeError(
                "The `default` passed to `ComponentField` must be either a component "
                "class or a `PartialComponent`."
            )

        self.name = utils.missing
        self.allow_missing = allow_missing
        self.host_component_class = utils.missing
        self.type = utils.missing
        self._registered = False
        self._default = default

    def __set_name__(self, host_component_class: Type[C], name: str):
        if self._registered:
            raise ValueError("This field has already been registered to a component.")
        if name.startswith("_"):
            raise ValueError("Field names cannot start with underscores.")
        type_annotation = utils.missing
        for super_class in inspect.getmro(host_component_class):
            if name in getattr(super_class, "__annotations__", {}):
                type_annotation = super_class.__annotations__[name]
                break
        if type_annotation is utils.missing:
            raise TypeError(
                "ComponentFields must be defined inside the component class definition, "
                "with a type-annotation as follows:\n\n"
                "```\n"
                "@component\n"
                "class ParentComponentClass:\n"
                "    field_name: SomeChildComponentType = ComponentField(\n"
                "        SomeDefaultChildComponentClass\n"
                "    )\n"
                "```\n"
                "\nUnlike `Field`, `ComponentField` cannot be used as a decorator.\n\n"
                f"Unable to find a type annotation for field '{name}' on class "
                f"'{host_component_class.__name__}'."
            )

        self.name = name
        self.host_component_class = host_component_class
        self.type = type_annotation
        self._registered = True

    def get_default(self, component_instance: C) -> F:
        if not self._registered:
            raise ValueError("This field has not been registered to a component.")
        if not self.has_default:
            msg = (
                f"ComponentField '{self.name}' has no default or configured component "
                "class."
            )
            # If this field is allow_missing, we raise an AttributeError, since it
            # doesn't really exist.
            if self.allow_missing:
                raise AttributeError(msg)
            # If it isn't allow_missing, this is a configuration error, because the
            # attribute does exist, but has no value.
            raise ConfigurationError(msg)

        if not isinstance(component_instance, self.host_component_class):
            raise TypeError(
                f"ComponentField '{self.name}' belongs to component "
                f"'{self.host_component_class.__name__}'; `get_default` must be called "
                f"with an instance of '{self.host_component_class.__name__}'. Received: "
                f"{repr(component_instance)}."
            )

        # We build the instance without passing in any additional arguments. The
        # child instance will be able to pick up any missing field values from
        # the parent in the usual way, as it will be a nested sub-component.
        return self._default()
