import pytest

from zookeeper.core.task import task


def test_with_argumentless_run():
    """Tasks with argument-less `run` should not cause errors."""

    @task
    class T1:
        def run(self):
            pass

    @task
    class T2:
        @classmethod
        def run(cls):
            pass

    @task
    class T3:
        @staticmethod
        def run():
            pass


def test_no_run_error():
    """Tasks without `run` should cause an error."""

    with pytest.raises(
        TypeError, match="Classes decorated with @task must define a `run` method."
    ):

        @task
        class T:
            pass


def test_run_with_args_error():
    """Defining a subclass which has a `run` that takes any arguments should raise a
    ValueError."""

    with pytest.raises(
        TypeError,
        match=r"^A @task class must define a `run` method taking no arguments except `self`",
    ):

        @task
        class T:
            def run(a, b):
                pass
