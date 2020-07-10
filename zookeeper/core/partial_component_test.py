import pytest

from zookeeper.core.component import component, configure
from zookeeper.core.field import ComponentField, Field
from zookeeper.core.partial_component import PartialComponent


@pytest.fixture
def ExampleComponentClasses():
    class AbstractChild:
        pass

    @component
    class Child1(AbstractChild):
        a: int = Field()
        b: str = Field("foo")

    @component
    class Child2(AbstractChild):
        c: float = Field(3.14)

    @component
    class Parent:
        a: int = Field(10)
        child: AbstractChild = ComponentField()

    return Parent, Child1, Child2


def test_init_error_on_non_component():
    with pytest.raises(
        TypeError,
        match="The class passed to `PartialComponent` must be a component class.",
    ):
        PartialComponent(2.71, a=3)

    with pytest.raises(
        TypeError,
        match="The class passed to `PartialComponent` must be a component class.",
    ):
        PartialComponent(lambda x: x * 2, a=3)

    class Test:
        a: int

    with pytest.raises(
        TypeError,
        match="The class passed to `PartialComponent` must be a component class.",
    ):
        PartialComponent(Test, a=3)

    @component
    class Test2:
        a: int = Field(5)

    with pytest.raises(
        TypeError,
        match="`PartialComponent` must be passed component classes, not component instances.",
    ):
        PartialComponent(Test2(), a=3)


def test_init_error_no_kwargs(ExampleComponentClasses):
    _, _, Child2 = ExampleComponentClasses

    with pytest.raises(
        TypeError,
        match="`PartialComponent` must receive at least one keyword argument.",
    ):
        PartialComponent(Child2)


def test_kwargs_accept_component_class(ExampleComponentClasses):
    Parent, Child1, _ = ExampleComponentClasses

    # This should succeed without error.
    partial = PartialComponent(Parent, child=Child1)

    # Generate a component instance from the partial, and configure it.
    p = partial()
    configure(p, {})

    assert isinstance(p.child, Child1)
    assert p.child.b == "foo"
    assert p.child.a == 10  # This tests that field value inheritence still works.


def test_kwargs_accept_nested_partial_component(ExampleComponentClasses):
    Parent, Child1, _ = ExampleComponentClasses

    # This should succeed without error.
    partial = PartialComponent(Parent, child=PartialComponent(Child1, a=5))

    # Generate a component instance from the partial, and configure it.
    p = partial()
    configure(p, {})

    assert isinstance(p.child, Child1)
    assert p.child.a == 5
