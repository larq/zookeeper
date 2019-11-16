from abc import ABC, abstractmethod
from inspect import signature

from zookeeper.component import Component


class Task(Component, ABC):
    """
    A 'Task' component that performs a task on `run`.
    """

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        # Enforce argument-less `run`
        if hasattr(cls, "run"):
            call_args = signature(cls.run).parameters
            if len(call_args) == 0 or len(call_args) == 1 and "self" in call_args:
                return
            raise ValueError(
                "A `Task` subclass must define a `run` method taking no positional "
                f"arguments which runs the task, but {cls.__name__}.run accepts "
                f"positional arguments {call_args}."
            )

    @abstractmethod
    def run(self):
        pass
