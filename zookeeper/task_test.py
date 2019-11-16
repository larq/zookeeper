import pytest

from zookeeper.task import Task


def test_override_call_with_args_error():

    # Defining a subclass which overrides `run` with positional arguments
    # should raise a ValueError.
    with pytest.raises(
        ValueError,
        match=r"^A `Task` subclass must define a `run` method taking no positional",
    ):

        class J(Task):
            def run(self, a, b):
                pass

    # Overriding `run` without positional arguments should not raise an
    # error.

    class J1(Task):
        def run(self):
            pass

    # The same should be true if `run` is a static method or class method.

    class J2(Task):
        @classmethod
        def run(cls):
            pass

    class J3(Task):
        @staticmethod
        def run():
            pass
