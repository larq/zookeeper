import abc
from typing import List, Tuple
from unittest.mock import patch

import click
import pytest

from zookeeper.core.component import component, configure
from zookeeper.core.factory import factory
from zookeeper.core.field import ComponentField, Field


@pytest.fixture
def ExampleComponentClass():
    @component
    class A:
        a: int = Field()
        b: str = Field("foo")

    return A


def test_non_class_decorate_error():
    """
    An error should be raised when attempting to decorate a non-class object.
    """
    with pytest.raises(
        TypeError, match="Only classes can be decorated with @component."
    ):

        @component  # type: ignore
        def fn():
            pass


def test_abstract_class_decorate_error():
    """
    An error should be raised when attempting to decorate an abstract class.
    """
    with pytest.raises(
        TypeError, match="Abstract classes cannot be decorated with @component."
    ):

        @component
        class A(abc.ABC):
            @abc.abstractmethod
            def foo(self):
                pass


def test_init_decorate_error():
    """
    An error should be raised when attempting to decorate a class with an
    `__init__` method.
    """
    with pytest.raises(
        TypeError,
        match="Component classes must not define a custom `__init__` method.",
    ):

        @component
        class A:
            def __init__(self, a, b=5):
                self.a = a
                self.b = b


def test_no_init(ExampleComponentClass):
    """
    If the decorated class does not have an `__init__` method, the decorated
    class should define an `__init__` which accepts kwargs to set field values,
    and raises appropriate arguments when other values are passed.
    """

    x = ExampleComponentClass(a=2)
    assert x.a == 2
    assert x.b == "foo"

    x = ExampleComponentClass(a=0, b="bar")
    assert x.a == 0
    assert x.b == "bar"

    # Verify that arguments are disallowed (the 1 positional argument the error
    # message refers to is `self`).
    with pytest.raises(
        TypeError, match=r"takes 1 positional argument but 2 were given",
    ):
        ExampleComponentClass("foobar")

    with pytest.raises(
        TypeError,
        match=(
            "Keyword arguments passed to component `__init__` must correspond to "
            "component fields. Received non-matching argument 'some_other_field_name'"
        ),
    ):
        ExampleComponentClass(some_other_field_name=0)


def test_configure_override_field_values(ExampleComponentClass):
    """Component fields should be overriden correctly."""

    x = ExampleComponentClass()
    configure(x, {"a": 0, "b": "bar"})
    assert x.a == 0
    assert x.b == "bar"


def test_configure_scoped_override_field_values():
    """Field overriding should respect component scope."""

    @component
    class Child:
        a: int = Field()
        b: str = Field()
        c: List[float] = Field()

    @component
    class Parent:
        b: str = Field("bar")
        child: Child = ComponentField(Child)

    @component
    class GrandParent:
        a: int = Field()
        b: str = Field()
        parent: Parent = ComponentField(Parent)

    grand_parent = GrandParent()

    configure(
        grand_parent,
        {
            "a": 10,
            "parent.child.a": 15,
            "b": "foo",
            "parent.child.b": "baz",
            "parent.child.c": [0, 4.2],
        },
    )

    # The grand-parent `grand_parent` should have the value `a` = 10, but the
    # child `grand_parent.parent.child` should get the value `a` = 15.
    assert grand_parent.a == 10
    assert grand_parent.parent.child.a == 15

    # `b` is declared as a field at all three levels. The 'baz' value should be
    # scoped only to the child, so 'foo' will apply to both the parent and
    # grand-parent.
    assert grand_parent.b == "foo"
    assert grand_parent.parent.b == "foo"
    assert grand_parent.parent.child.b == "baz"

    # `c` is declared as a field only in the child.
    assert grand_parent.parent.child.c == [0, 4.2]


def test_configure_automatically_instantiate_subcomponent():
    """
    If there is only a single component subclass of a field type, an instance of
    the class should be automatically instantiated during configuration.
    """

    class AbstractChild:
        pass

    @component
    class Child1(AbstractChild):
        pass

    @component
    class Parent:
        child: AbstractChild = ComponentField()

    # There is only a single defined component subclass of `AbstractChild`,
    # `Child1`, so we should be able to configure an instance of `Parent` and
    # have an instance automatically instantiated in the process.

    p = Parent()
    configure(p, {})
    assert isinstance(p.child, Child1)

    @component
    class Child2(AbstractChild):
        pass

    # Now there is another defined component subclass of `AbstractChild`,
    # so configuration will now fail (as we cannot choose between the two).

    p = Parent()
    with pytest.raises(
        ValueError,
        match=r"^Component field 'Parent.child' of type 'AbstractChild' has no default or configured class.",
    ):
        configure(p, {})


def test_configure_non_interactive_missing_field_value(ExampleComponentClass):
    """
    When not configuring interactively, an error should be raised if a field has
    neither a default nor a configured value.
    """

    with pytest.raises(
        ValueError,
        match=r"^No configuration value found for annotated field 'FAKE_NAME.a' of type 'int'.",
    ):
        configure(ExampleComponentClass(), {"b": "bar"}, name="FAKE_NAME")


def test_configure_interactive_prompt_missing_field_value(ExampleComponentClass):
    """
    When configuring interactively, fields without default or configured values
    should prompt for value input through the CLI.
    """

    x = ExampleComponentClass()
    a_value = 42

    with patch("zookeeper.core.utils.prompt", return_value=str(a_value)) as prompt:
        configure(x, {"b": "bar"}, name="FAKE_NAME", interactive=True)

    assert x.a == a_value
    assert x.b == "bar"
    prompt.assert_called_once()


def test_configure_interactive_prompt_for_subcomponent_choice():
    """
    When configuring interactively, sub-component fields without default or
    configured values should prompt for a choice of subcomponents to instantiate
    through the CLI.
    """

    class AbstractChild:
        pass

    @component
    class Child1(AbstractChild):
        pass

    @component
    class Child2(AbstractChild):
        pass

    class Child3_Abstract(AbstractChild):
        pass

    @component
    class Child3A(Child3_Abstract):
        pass

    @component
    class Child3B(Child3_Abstract):
        pass

    @component
    class Parent:
        child: AbstractChild = ComponentField()

    # The prompt lists the concrete component subclasses (alphabetically) and
    # asks for an an integer input corresponding to an index in this list.

    # We expect the list to therefore be as follows (`AbstractChild` and
    # `Child3_Abstract` are excluded because although they live in the subclass
    # hierarchy, neither is a component):
    expected_class_choices = [Child1, Child2, Child3A, Child3B]

    for i, expected_choice in enumerate(expected_class_choices):
        p = Parent()

        with patch("zookeeper.core.utils.prompt", return_value=str(i + 1)) as prompt:
            configure(p, {}, interactive=True)

        assert isinstance(p.child, expected_choice)
        prompt.assert_called_once()


def test_str_and_repr():
    """
    `__str__` and `__repr__` should give formatted strings that represent nested
    components nicely.
    """

    @component
    class Child:
        a: int = Field()
        b: str = Field()
        c: List[float] = Field()
        d: int = Field(allow_missing=True)

    @component
    class Parent:
        b: str = Field("bar")
        child: Child = ComponentField(Child)

    p = Parent()

    configure(p, {"child.a": 10, "b": "foo", "child.c": [1.5, -1.2]}, name="parent")

    assert (
        click.unstyle(repr(p))
        == """Parent(b="foo", child=Child(a=10, b="foo", c=[1.5, -1.2], d=<missing>))"""
    )
    assert (
        click.unstyle(str(p))
        == """Parent(
    b="foo",
    child=Child(
        a=10,
        b="foo",
        c=[1.5, -1.2],
        d=<missing>
    )
)"""
    )


def test_type_check(ExampleComponentClass):
    """During configuration we should type-check all field values."""

    instance = ExampleComponentClass()

    configure(instance, {"a": 4.5}, name="x")

    # Attempting to access the field should now raise a type error.
    with pytest.raises(
        TypeError,
        match="Field 'a' of component 'x' is annotated with type '<class 'int'>', which is not satisfied by value 4.5.",
    ):
        instance.a


def test_error_if_field_overwritten_in_subclass():
    @component
    class SuperClass:
        foo: str = Field("bar")

    with pytest.raises(ValueError, match="Field 'foo' is defined on super-class"):

        @component
        class SubClass(SuperClass):
            foo = 1


def test_component_field_factory_type_check(capsys):
    class Base:
        pass

    class Concrete(Base):
        pass

    @factory
    class F1:
        def build(self) -> Base:
            return Concrete()

    @factory
    class F2:
        def build(self) -> Concrete:
            return Concrete()

    @factory
    class F3:
        def build(self) -> Tuple[int, int, int]:
            return (1, 2, 3)

    @component
    class A1:
        base: Base = ComponentField(F1)

    @component
    class A2:
        base: Base = ComponentField(F2)

    @component
    class A3:
        base: Tuple[float, float, float] = ComponentField(F3)

    # These should succeed.
    A1().base
    A2().base

    # Do this here to drop any already captured output.
    capsys.readouterr()

    # This should succeed, but without a type-check (should print a warning)
    A3().base
    captured = capsys.readouterr()
    assert (
        captured.err
        == "WARNING: Unable to check that typing.Tuple[int, int, int] is a sub-type of typing.Tuple[float, float, float].\n"
    )


def test_component_getattr_value_via_factory_parent():
    """See https://github.com/larq/zookeeper/issues/121."""

    @component
    class Child:
        x: int = Field()

    @factory
    class Factory:
        child: Child = ComponentField(Child)

        x: int = Field(5)

        def build(self) -> int:
            return self.child.x

    f = Factory()

    configure(f, {})

    assert f.child.x == 5
    assert f.build() == 5


def test_component_inherited_factory_value():
    """https://github.com/larq/zookeeper/issues/123."""

    @factory
    class IntFactory:
        def build(self) -> int:
            return 5

    @component
    class Child:
        x: int = ComponentField()

    @component
    class Parent:
        child: Child = ComponentField(Child)
        x: int = ComponentField(IntFactory)

    p = Parent()
    configure(p, {})
    assert p.x == 5
    assert p.child.x == 5

    p = Parent()
    configure(p, {"child.x": 7})
    assert p.x == 5
    assert p.child.x == 7


def test_component_post_configure():
    with pytest.raises(
        TypeError,
        match="The `__post_configure__` attribute of a @component class must be a method.",
    ):

        @component
        class A:
            __post_configure__ = 3.14

    with pytest.raises(
        TypeError,
        match="The `__post_configure__` method of a @component class must take no arguments except `self`",
    ):

        @component
        class B:
            def __post_configure__(self, x):
                pass

    # This definition should succeed.
    @component
    class C:
        a: int = Field(0)
        b: float = Field(3.14)

        def __post_configure__(self):
            self.c = self.a + self.b

    c = C()

    configure(c, {"a": 1, "b": -3.14})

    assert c.c == 1 - 3.14


def test_component_configure_error_non_existant_key():
    @component
    class Child:
        a: int = Field(1)

    @component
    class Parent:
        b: str = Field("foo")
        child: Child = ComponentField(Child)

    @component
    class GrandParent:
        c: float = Field(3.14)
        parent: Parent = ComponentField(Parent)

    # A missing key should cause an error.
    with pytest.raises(
        ValueError,
        match="Key 'd' does not correspond to any field of component 'GrandParent'.",
    ):
        configure(GrandParent(), {"d": "bar"}, name="GrandParent")

    # 'b' exists on the parent but not the grand-parent, so should be set via
    # `parent.b='bar'`. This should raise an error:
    with pytest.raises(
        ValueError,
        match="Key 'b' does not correspond to any field of component 'GrandParent'.",
    ):
        configure(GrandParent(), {"b": "bar"})
    # But this should not:
    g = GrandParent()
    configure(g, {"parent.b": "bar"})
    assert g.parent.b == "bar"

    # Test that an error is correctly raised recursively.
    with pytest.raises(
        ValueError,
        match="Key 'non_existent_field' does not correspond to any field of component 'GrandParent.parent'.",
    ):
        configure(GrandParent(), {"parent.non_existent_field": "bar"})


def test_component_configure_error_non_component_instance():
    class A:
        a: int = Field()

    with pytest.raises(
        TypeError,
        match="Only @component, @factory, and @task instances can be configured.",
    ):
        configure(A(), conf={"a": 5})

    @component
    class B:
        b: int = Field()

    with pytest.raises(
        TypeError,
        match="Only @component, @factory, and @task instances can be configured.",
    ):
        # The following we expect to fail because it is a component class, not
        # an instance.
        configure(B, conf={"b": 3})

    class C(B):
        c: int = Field()

    with pytest.raises(
        TypeError,
        match="Only @component, @factory, and @task instances can be configured.",
    ):
        # Even the an instance of a class that subclasses a component class
        # should fail.
        configure(C(), conf={"b": 3, "c": 42})


def test_component_configure_field_allow_missing():
    @component
    class A:
        a: int = Field()
        b: float = Field(allow_missing=True)

        @Field
        def c(self) -> float:
            if hasattr(self, "b"):
                return self.b
            return self.a

    # Missing field 'a' should cause an error.
    with pytest.raises(
        ValueError,
        match="No configuration value found for annotated field 'A.a' of type 'int'.",
    ):
        configure(A(), {"b": 3.14})

    # But missing field 'b' should not cause an error.
    instance = A()
    configure(instance, {"a": 0})
    assert instance.c == 0
    instance = A()
    configure(instance, {"a": 0, "b": 3.14})
    assert instance.c == 3.14


def test_component_configure_component_field_allow_missing():
    class Base:
        a: int = Field()

    @component
    class Child1(Base):
        a = Field(5)

    @component
    class Child2(Base):
        a = Field(5)

    @component
    class Parent:
        child: Base = ComponentField()
        child_allow_missing: Base = ComponentField(allow_missing=True)

    # Missing out "child" should cause an error.
    with pytest.raises(
        ValueError,
        match="Component field 'Parent.child' of type 'Base' has no default or configured class.",
    ):
        configure(Parent(), {"child_allow_missing": "Child2"})

    # But missing out "child_allow_missing" should succeed.
    instance = Parent()
    configure(instance, {"child": "Child1"})
    assert not hasattr(instance, "child_allow_missing")


def test_component_allow_missing_field_inherits_defaults():
    @component
    class Child:
        a: int = Field(allow_missing=True)

    @component
    class Parent:
        a: int = Field(5)
        child: Child = ComponentField(Child)

    # This should succeed without error.
    instance = Parent()
    configure(instance, {})
    assert instance.child.a == 5
