from typing import List, Optional

import pytest

from zookeeper.core.component import component
from zookeeper.core.field import ComponentField, Field
from zookeeper.core.partial_component import PartialComponent
from zookeeper.core.utils import ConfigurationError


def test_init_non_immutable_non_callable():
    with pytest.raises(
        TypeError, match="If `default` is passed to `Field`, it must be either:"
    ):
        Field([1, 2, 3])


def test_init_callable_many_arg():
    with pytest.raises(
        TypeError, match="If `default` is passed to `Field`, it must be either:"
    ):
        Field(lambda x, y, z: x + y + z)


def test_no_default():
    class A:
        foo: int = Field()

    assert not A.foo.has_default

    with pytest.raises(
        ConfigurationError, match="Field 'foo' has no default or configured value"
    ):
        A.foo.get_default(A())


def test_immutable_default():
    class A:
        foo: int = Field(5)
        bar: float = Field(1.2)
        baz: bool = Field(False)
        foobar: Optional[int] = Field(None)

    for field_name, expected_default in zip(
        ["foo", "bar", "baz", "foobar"], [5, 1.2, False, None]
    ):
        instance = A()
        field = getattr(A, field_name)
        assert field.has_default
        assert field.get_default(instance) == expected_default


def test_non_immutable_non_factory_default():
    with pytest.raises(TypeError):
        Field([1, 2, 3])


def test_factory_default():
    class A:
        foo: int = Field(lambda: 7)
        bar: List[int] = Field(lambda instance: [instance.baz])
        baz = 2

    instance = A()

    assert A.foo.has_default
    assert A.foo.get_default(instance) == 7

    assert A.bar.has_default
    bar_default = A.bar.get_default(instance)
    assert isinstance(bar_default, list)
    assert bar_default[0] == 2


def test_decorated_field():
    class A:
        @Field
        def foo() -> float:
            return 3.14

        # Class `A` would usually be decorated with @component, which would
        # modify `__getattribute__` so that calling `self.foo` correctly fetches
        # the default value of the field. However, as we want to test `Field`
        # independently of `component`, use this workaround here (otherwise the
        # `bar` definition below would fail).
        @property
        def foo_value(self):
            return A.foo.get_default(self)

        @Field
        def bar(self) -> int:
            return int(self.foo_value ** self.foo_value)

    instance = A()

    assert A.foo.has_default
    assert A.foo.get_default(instance) == 3.14

    assert A.bar.has_default
    assert A.bar.get_default(instance) == 36


def test_unregistered_field():
    field = Field(5)

    with pytest.raises(
        ValueError, match="This field has not been registered to a component"
    ):
        field.has_default

    with pytest.raises(
        ValueError, match="This field has not been registered to a component"
    ):
        field.get_default(object())


def test_prohibit_underscore_field_name():
    # This is ugly because the actual exception raised is a `ValueError` with
    # appropriate error message, but the the standard library catches that
    # exception and chains it with a RuntimeError. There's no (easy) way to get
    # the original exception and error message.
    with pytest.raises(RuntimeError):

        class A:
            _field = Field()


def test_allow_missing():
    # This should succeed because we don't set a default value...
    f = Field(allow_missing=True)
    assert f.allow_missing

    # ...but this should fail because there's a default provided
    with pytest.raises(ValueError):
        Field(3.14, allow_missing=True)


# Used in the `ComponentField` tests below.
class AbstractClass:
    a: int = Field()


@component
class ConcreteComponent:
    a: int = Field(2)


def test_component_field_component_instance_default():
    with pytest.raises(
        TypeError,
        match="The `default` passed to `ComponentField` must be a component class, not a component instance.",
    ):
        ComponentField(ConcreteComponent())


def test_component_field_no_default():
    class A:
        foo: AbstractClass = ComponentField()

    assert not A.foo.has_default

    with pytest.raises(
        ConfigurationError,
        match="ComponentField 'foo' has no default or configured component class.",
    ):
        A.foo.get_default(A())


def test_component_field_component_class_default():
    class A:
        foo: AbstractClass = ComponentField(ConcreteComponent)

    assert A.foo.has_default
    assert isinstance(A.foo.get_default(A()), ConcreteComponent)


def test_component_field_partial_component_default():
    class A:
        foo: AbstractClass = ComponentField(PartialComponent(ConcreteComponent, a=5))

    assert A.foo.has_default
    default_value = A.foo.get_default(A())
    assert isinstance(default_value, ConcreteComponent)
    assert default_value.a == 5


def test_component_field_kwargs():
    with pytest.raises(TypeError, match="Keyword arguments can only be passed"):
        ComponentField(a=1, b=2, c=3)

    class A:
        foo: AbstractClass = ComponentField(ConcreteComponent, a=5)

    assert A.foo.has_default
    default_value = A.foo.get_default(A())
    assert isinstance(default_value, ConcreteComponent)
    assert default_value.a == 5


def test_component_field_allow_missing():
    # This should succeed because we don't set a default value...
    f = ComponentField(allow_missing=True)
    assert f.allow_missing

    # ...but this should fail because there's a default provided
    with pytest.raises(ValueError):
        ComponentField(3.14, allow_missing=True)
