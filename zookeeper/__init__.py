from zookeeper.core import (
    ComponentField,
    Field,
    PartialComponent,
    cli,
    component,
    configure,
    factory,
    task,
)

try:
    from importlib import metadata  # type: ignore
except ImportError:
    # Running on pre-3.8 Python; use importlib-metadata package
    import importlib_metadata as metadata  # type: ignore

__version__ = metadata.version("zookeeper")

__all__ = [
    "cli",
    "ComponentField",
    "component",
    "configure",
    "factory",
    "Field",
    "PartialComponent",
    "task",
]
