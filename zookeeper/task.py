from abc import ABC, abstractmethod
from zookeeper import Component
from inspect import getfullargspec, ismethod


class Task(Component, ABC):
    """
    A 'Task' component that performs a task on `run`.
    """

    def __init_subclass__(cls: type, *args, **kwargs):
        # Enforce argument-less `run`
        if hasattr(cls, "run"):
            call_args = getfullargspec(cls.run).args
            is_cls_method = ismethod(cls.run) and cls.run.__self__ == cls

            if (
                len(call_args) == 0
                or (is_cls_method and call_args == ["cls"])
                or call_args == ["self"]
            ):
                return

        raise ValueError(
            "A `Task` subclass must define a `run` method taking no positional "
            f"arguments which runs the task, but {cls.__name__}.run accepts "
            f"positional arguments {call_args}."
        )

    @abstractmethod
    def run(self):
        pass
