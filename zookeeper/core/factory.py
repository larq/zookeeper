import functools
import inspect
from typing import Dict, Set, Type

from zookeeper.core import utils
from zookeeper.core.component import component


# A sentinel class/object for missing default values.
class Missing:
    def __repr__(self):
        return f"<missing>"


missing = Missing()


_FACTORY_REGISTRY: Dict[Type, Set] = {}


def _wrap_build(factory_cls: Type) -> None:
    """
    Every @factory has a `build()` method, which we wrap so that `build()` is
    only called once (lazily) and the value is cached.
    """
    fn = factory_cls.build

    @functools.wraps(fn)
    def wrapped_fn(factory_instance):
        if factory_instance.__component_factory_value__ is missing:
            result = fn(factory_instance)
            if not utils.type_check(
                result, factory_cls.__component_factory_return_type__
            ):
                raise TypeError(
                    f"@factory '{factory_cls}' has a `build()` method is annotated with "
                    f"return type {factory_cls.__component_factory_return_type__}, "
                    f"which is not satisfied by the return value {result}."
                )
            factory_instance.__component_factory_value__ = result
        return factory_instance.__component_factory_value__

    factory_cls.build = wrapped_fn


# TODO: pretty `str` and `repr`


def factory(cls: Type):
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
            "Classes decorated with @factory must implement a build method taking "
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
    cls.__component_factory_value__ = missing

    _wrap_build(cls)

    if signature.return_annotation not in _FACTORY_REGISTRY:
        _FACTORY_REGISTRY[signature.return_annotation] = set([cls])
    else:
        _FACTORY_REGISTRY[signature.return_annotation].add(cls)

    return cls
