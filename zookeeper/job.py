from zookeeper import Component
from inspect import getfullargspec, ismethod


class Job(Component):
    """

    """

    def __init_subclass__(cls: type, *args, **kwargs):
        # Enforce argument-less `__call__`
        if hasattr(cls, "__call__"):
            call_args = getfullargspec(cls.__call__).args
            is_cls_method = ismethod(cls.__call__) and cls.__call__.__self__ == cls

            if (
                len(call_args) == 0
                or (is_cls_method and call_args == ["cls"])
                or call_args == ["self"]
            ):
                return

        raise ValueError(
            "A `Job` subclass must define a `__call__` method taking no positional "
            f"arguments which runs the job, but {cls.__name__}.__call__ accepts "
            f"positional arguments {call_args}."
        )
