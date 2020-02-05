import inspect
from typing import Callable, Generic, Type, TypeVar, Union

from zookeeper.core import utils
from zookeeper.core.partial_component import PartialComponent


# A sentinel class/object for missing default values.
class _MISSING:
    def __repr__(self):
        return f"<missing>"


_missing = _MISSING()


# Type-variables to parameterise fields.
_ComponentType = TypeVar("_ComponentType")
_FieldType = TypeVar("_FieldType")


class Field(Generic[_ComponentType, _FieldType]):
    """
    A configurable field for Zookeeper components. Fields must be typed, may
    take default values, and are configurable through the CLI.

    This class is not appropriate for nesting child sub-components; for this,
    use `ComponentField` instead.

    Internally, it is expected that each component will cache the result of
    `get_default`.
    """

    def __init__(
        self,
        default_value: Union[
            _MISSING,
            _FieldType,
            Callable[[], _FieldType],
            Callable[[_ComponentType], _FieldType],
        ] = _missing,
    ):
        # Define here once to avoid having to define twice below.
        default_value_error = TypeError(
            "If `default_value` is passed to `Field`, it must be either:\n"
            "- An immutable value (int, float, bool, string, or None).\n"
            "- A function or lambda accepting no arguments or a single\n"
            "  argument (`self`), and returning the default value.\n"
            f"Received: {default_value}."
        )

        return_annotation = inspect.Signature.empty

        if default_value is _missing:
            default_factory = None
        elif isinstance(default_value, (int, float, bool, str, type(None))):
            default_factory = lambda instance: default_value  # noqa: E731
        elif inspect.isfunction(default_value):
            signature = inspect.signature(default_value)
            return_annotation = signature.return_annotation
            if len(signature.parameters) == 0:
                default_factory = lambda instance: default_value()  # noqa: E731
            elif len(signature.parameters) == 1 and all(
                signature.parameters[name].kind
                not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
                for name in signature.parameters
            ):
                default_factory = default_value
            else:
                raise default_value_error
        else:
            raise default_value_error

        self._registered = False
        self._return_annotation = return_annotation
        self._default_factory = default_factory

    # This follows the PEP 487 `__set_name__` protocol; this method is called
    # automatically on every object defined within a class body, passing in the
    # class object and name of the descriptor. We use it here to obtain the name
    # and type of the field.
    def __set_name__(self, host_component_class: Type[_ComponentType], name: str):
        if self._registered:
            raise ValueError("This field has already been registered to a component.")
        if name.startswith("_"):
            raise ValueError("Field names cannot start with underscores.")
        try:
            type_annotation = host_component_class.__annotations__[name]
            if (
                self._return_annotation is not inspect.Signature.empty
                and type_annotation != self._return_annotation
            ):
                raise TypeError(
                    f"Two non-equal type annotations found for field '{name}': "
                    f"{type_annotation} and {self._return_annotation}."
                )
        except (AttributeError, KeyError):
            if self._return_annotation is inspect.Signature.empty:
                raise TypeError(
                    "Fields must be defined inside the component class definition, "
                    "with a type-annotation in one of the following ways:\n\n"
                    "```\n"
                    "@component\n"
                    "class ComponentClass:\n"
                    "    ...\n"
                    "    # Like this\n"
                    "    name_1: type_1 = Field(default_value_1)\n"
                    "    ...\n"
                    "    # Or like this\n"
                    "    @Field\n"
                    "    def name_2(self) -> type_2:\n"
                    "        ...\n"
                    "        return default_value_2\n"
                    "```\n\n"
                    f"Unable to find a type annotation for field '{name}' on class "
                    f"'{host_component_class.__name__}'."
                )
            type_annotation = self._return_annotation

        self.name = name
        self.host_component_class = host_component_class
        self.type = type_annotation
        self._registered = True

    @property
    def has_default(self) -> bool:
        if not self._registered:
            raise ValueError("This field has not been registered to a component.")
        return self._default_factory is not None

    def get_default(self, component_instance: _ComponentType) -> _FieldType:
        if not self._registered:
            raise ValueError("This field has not been registered to a component.")
        if not self.has_default:
            raise AttributeError(
                f"Field '{self.name}' has no default or configured value."
            )
        if not isinstance(component_instance, self.host_component_class):
            raise TypeError(
                f"Field '{self.name}' belongs to component "
                f"'{self.host_component_class.__name__}'; `get_default` must be called "
                f"with an instance of '{self.host_component_class.__name__}'. Received: "
                f"{repr(component_instance)}."
            )

        return self._default_factory(component_instance)


# TODO: Maybe add lower-case alias `field`?


class ComponentField(Field, Generic[_ComponentType, _FieldType]):
    def __init__(
        self,
        default_component_class: Union[
            _MISSING, _FieldType, PartialComponent[_FieldType]
        ] = _missing,
    ):
        if default_component_class is _missing:
            default_factory = None
        elif isinstance(
            default_component_class, PartialComponent
        ) or utils.is_component_class(default_component_class):
            default_factory = default_component_class
        elif utils.is_component_instance:
            raise TypeError(
                "The `default_component_class` passed to `ComponentField` must be "
                "a component class, not a component instance. Received: "
                f"{repr(default_component_class)}."
            )
        else:
            raise TypeError(
                "The `default_component_class` passed to `ComponentField` must be "
                "either a component class or a `PartialComponent`."
            )

        self._registered = False
        self._default_factory = default_factory

    def __set_name__(self, host_component_class: Type[_ComponentType], name: str):
        if self._registered:
            raise ValueError("This field has already been registered to a component.")
        if name.startswith("_"):
            raise ValueError("Field names cannot start with underscores.")
        try:
            type_annotation = host_component_class.__annotations__[name]
        except (AttributeError, KeyError):
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

    def get_default(self, component_instance: _ComponentType) -> _FieldType:
        if not self._registered:
            raise ValueError("This field has not been registered to a component.")
        if not self.has_default:
            raise AttributeError(
                f"ComponentField '{self.name}' has no default or configured component "
                "class."
            )
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
        return self._default_factory()
