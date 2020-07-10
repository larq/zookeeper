from zookeeper.core import utils
from zookeeper.core.component import component
from zookeeper.core.factory import factory


def test_is_component_class():
    assert not utils.is_component_class(5)
    assert not utils.is_component_class(lambda: "foo")

    class A:
        pass

    assert not utils.is_component_class(A)

    @component
    class B:
        pass

    assert utils.is_component_class(B)
    assert not utils.is_component_class(B())


def test_is_component_instance():
    assert not utils.is_component_instance(5)
    assert not utils.is_component_instance(lambda: "foo")

    class A:
        pass

    assert not utils.is_component_instance(A())

    @component
    class B:
        pass

    assert not utils.is_component_instance(B)
    assert utils.is_component_instance(B())


def test_is_factory_class():
    class A:
        def build(self) -> object:
            pass

    assert not utils.is_factory_class(A)

    @factory
    class B:
        def build(self) -> object:
            pass

    assert utils.is_factory_class(B)
    assert not utils.is_factory_class(B())


def test_is_factory_instance():
    class A:
        def build(self) -> object:
            pass

    assert not utils.is_factory_instance(A())

    @factory
    class B:
        def build(self) -> object:
            pass

    assert not utils.is_factory_instance(B)
    assert utils.is_factory_instance(B())
