import pytest

from zookeeper.core.factory import FACTORY_REGISTRY, factory


def test_no_build_method():
    with pytest.raises(
        TypeError,
        match=r"Classes decorated with @factory must implement a `build\(\)` method taking precisely one positional argument",
    ):

        @factory
        class A:
            pass


def test_too_many_build_args():
    with pytest.raises(
        TypeError,
        match=r"Classes decorated with @factory must implement a `build\(\)` method taking precisely one positional argument",
    ):

        @factory
        class A:
            def build(self, a, b, c):
                pass


def test_no_return_annotation():
    with pytest.raises(
        TypeError,
        match=r"The `build\(\)` method of a @factory class must have an annotated return type annotation",
    ):

        @factory
        class A:
            def build(self):
                pass


def test_class_added_to_registry():
    class AbstractType:
        pass

    @factory
    class A:
        def build(self) -> AbstractType:
            return AbstractType()

    assert A in FACTORY_REGISTRY.get(AbstractType, set())


# TODO: test wrapped `build()` function.
