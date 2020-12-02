import inspect
from typing import Generic, Type, TypeVar

from zookeeper.core import utils

# A type-variable used to parameterise `PartialComponent`. This is for static
# type-checking.
_ComponentType = TypeVar("_ComponentType")


# Defined once here to avoid having to define it twice in the `__init__` body.
_kwargs_error = TypeError(
    "Keyword arguments passed to `PartialComponent` must be either:\n"
    "- An immutable value (int, float, bool, string, or None).\n"
    "- A function or lambda accepting no arguments and returning the value that "
    "should be passed to the component upon instantiation.\n"
    "- An @component class that will be used to instantiate a component instance "
    "for the corresponding field value.\n"
    "- Another `PartialComponent`.\n"
    "Wrapping non-immutable values in a function / lambda allows the values "
    "to be lazily evaluated; they won't be created at all if the partial "
    "component is never instantiated."
)


class PartialComponent(Generic[_ComponentType]):
    """A wrapper around a component class that represents the component with some
    default field values modified, similar in principle to `functools.partial`.

    `PartialComponent(SomeComponentClass, a=3)(b=4)` is equivalent to
    `SomeComponentClass(a=3, b=4)`.
    """

    def __init__(self, component_class: Type[_ComponentType], **kwargs):
        if utils.is_component_instance(component_class):
            raise TypeError(
                "`PartialComponent` must be passed component classes, not component "
                f"instances. Received: {repr(component_class)}"
            )
        if not utils.is_component_class(component_class):
            raise TypeError(
                "The class passed to `PartialComponent` must be a component class. "
                f"Received: {component_class}."
            )
        if len(kwargs) == 0:
            raise TypeError(
                "`PartialComponent` must receive at least one keyword argument."
            )

        lazy_kwargs = {}
        for name, value in kwargs.items():
            if name not in component_class.__component_fields__:
                raise TypeError(
                    f"Keyword argument '{name}' passed to `PartialComponent` does "
                    "not correspond to any field of component class "
                    f"'{component_class.__name__}'."
                )
            if utils.is_immutable(value):
                lazy_kwargs[name] = utils.wrap_in_callable(value)
            elif utils.is_component_class(value) or isinstance(value, PartialComponent):
                lazy_kwargs[name] = value
            else:
                if not inspect.isfunction(value):
                    raise _kwargs_error
                if len(inspect.signature(value).parameters) == 0:
                    lazy_kwargs[name] = value
                else:
                    raise _kwargs_error

        self._component_class = component_class
        self._lazy_kwargs = lazy_kwargs

    # This follows the PEP 487 `__set_name__` protocol; this method is called
    # automatically on every object defined within a class body. We use it here
    # to disallow setting instances directly on classes.
    def __set_name__(self, cls, name):
        raise ValueError(
            "`PartialComponent` instances should not be directly assigned to class "
            "bodies. You should instead use `PartialComponent` inside a "
            "`ComponentField`, like so:\n"
            "```\n"
            "@component\n"
            "class ParentComponentClass:\n"
            "    child_component: SomeChildComponentType = ComponentField(\n"
            "        PartialComponent(\n"
            "            SomeDefaultChildComponentClass,\n"
            "            some_arg=some_default_value,\n"
            "            some_other_arg=some_other_default_value,\n"
            "            ...\n"
            "        )\n"
            "    )\n"
            "```"
        )

    def __call__(self, **kwargs) -> _ComponentType:
        for name, value in kwargs.items():
            if name not in self._component_class.__component_fields__:
                raise TypeError(
                    f"Keyword argument '{name}' passed to `PartialComponent` does "
                    "not correspond to any field of component class "
                    f"'{self._component_class.__name__}'."
                )

        # TODO: perhaps consider passing the argument-factories rather than
        #       evaluating the values here, as it's still possible that these
        #       values won't end up being used.
        evaluted_saved_kwargs = {
            name: value()
            for name, value in self._lazy_kwargs.items()
            if name not in kwargs
        }

        combined_kwargs = {**evaluted_saved_kwargs, **kwargs}

        return self._component_class(**combined_kwargs)
