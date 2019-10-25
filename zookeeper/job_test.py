import pytest

from zookeeper import Job


def test_override_call_with_args_error():

    # Defining a subclass which overrides `__call__` with positional arguments
    # should raise a ValueError.
    with pytest.raises(
        ValueError,
        match=r"^A `Job` subclass must define a `__call__` method taking no positional",
    ):

        class J(Job):
            def __call__(self, a, b):
                pass

    # Overriding `__call__` without positional arguments should not raise an
    # error.

    class J1(Job):
        def __call__(self):
            pass

    # The same should be true if `__call__` is a static method or class method.

    class J2(Job):
        @classmethod
        def __call__(cls):
            pass

    class J3(Job):
        @staticmethod
        def __call__():
            pass
