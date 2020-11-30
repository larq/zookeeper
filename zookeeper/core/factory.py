import functools
import inspect
from typing import Type

from zookeeper.core import utils
from zookeeper.core.component import component
from zookeeper.core.factory_registry import FACTORY_REGISTRY


def _wrap_build(factory_cls: Type) -> None:
    """Every @factory has a `build()` method, which we wrap so that `build()` is only
    called once (lazily) and the value is cached."""
    fn = factory_cls.build

    @functools.wraps(fn)
    def wrapped_fn(factory_instance):
        if factory_instance.__component_factory_value__ is utils.missing:
            result = fn(factory_instance)
            if not utils.type_check(
                result, factory_cls.__component_factory_return_type__
            ):
                raise TypeError(
                    f"@factory '{factory_cls}' has a `build()` method annotated with "
                    f"return type {factory_cls.__component_factory_return_type__}, "
                    f"which is not satisfied by the return value {result}."
                )
            factory_instance.__component_factory_value__ = result
        return factory_instance.__component_factory_value__

    factory_cls.build = wrapped_fn


def _wrap_str_repr(factory_cls: Type) -> None:
    str_fn = factory_cls.__str__
    repr_fn = factory_cls.__repr__

    @functools.wraps(str_fn)
    def wrapped_str(factory_instance):
        result = str_fn(factory_instance)
        return result.replace("<Unconfigured component ", "<Unconfigured factory ")

    @functools.wraps(repr_fn)
    def wrapped_repr(factory_instance):
        result = repr_fn(factory_instance)
        return result.replace("<Unconfigured component ", "<Unconfigured factory ")

    factory_cls.__str__ = wrapped_str
    factory_cls.__repr__ = wrapped_repr


def factory(cls: Type):
    """A decorator which turns a class into a Zookeeper factory.

    Factories are in particular Zookeeper components, so can have `Field`s and
    `ComponentFields`. Factories must define an argument-less `build()` method,
    with a return type annotation.

    When a factory component is used as a sub-component (i.e., configured as the
    value of a `ComponentField` in some parent component instance), the
    `build()` method is implicitly called upon the first access of the field
    value, and the result of `build()` is used as the value in the parent.

    Here is an example:

    ```
    @factory
    class F:
        a: int = Field()
        def build(self):
            return self.a + 4

    @component
    class C:
        a: int = Field(3)
        f: int = ComponentField(F)

    c = C()
    configure(c, {})
    print(c.f)

    >> # Output
    >> 7
    ```
    """
    cls = component(cls)

    try:
        signature = inspect.signature(cls.build)
        params = signature.parameters
        if (
            len(params) != 1
            or "self" not in params
            or list(params.values())[0].kind
            in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ):
            raise TypeError()
    except (AttributeError, TypeError):
        raise TypeError(
            "Classes decorated with @factory must implement a `build()` method taking "
            "precisely one positional argument, `self`."
        ) from None

    if signature.return_annotation is signature.empty:
        raise TypeError(
            "The `build()` method of a @factory class must have an annotated return "
            "type annotation, e.g.:\n\n"
            "```\n"
            "@factory\n"
            "class MyFactory:\n"
            "    ...\n"
            "    def build(self) -> SomeReturnType:\n"
            "        ...\n"
            "        return some_value\n"
            "```"
        )

    cls.__component_factory_return_type__ = signature.return_annotation
    cls.__component_factory_value__ = utils.missing

    _wrap_build(cls)
    _wrap_str_repr(cls)

    if signature.return_annotation not in FACTORY_REGISTRY:
        FACTORY_REGISTRY[signature.return_annotation] = set([cls])
    else:
        FACTORY_REGISTRY[signature.return_annotation].add(cls)

    return cls
