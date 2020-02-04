import abc
from typing import List
from unittest.mock import patch

import click
import pytest

from zookeeper.core.component import component, configure
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

        @component
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
            "parent.a": 15,
            "b": "foo",
            "parent.child.b": "baz",
            "c": [1.5, -1.2],
            "parent.c": [-17.2],
            "parent.child.c": [0, 4.2],
        },
    )

    # The grand-parent `grand_parent` should have the value `a` = 10. Even
    # though a config value is declared for its scope, `grand_parent.child`
    # should have no `a` value set, as it doesn't declare `a` as a field.
    # Despite this, `grand_parent.parent.child` should get the value `a` = 15,
    # as it lives inside the configuration scope of its parent,
    # `grand_parent.parent`.
    assert grand_parent.a == 10
    assert not hasattr(grand_parent.parent, "a")
    assert grand_parent.parent.child.a == 15

    # `b` is declared as a field at all three levels. The 'baz' value should be
    # scoped only to the child, so 'foo' will apply to both the parent and
    # grand-parent.
    assert grand_parent.b == "foo"
    assert grand_parent.parent.b == "foo"
    assert grand_parent.parent.child.b == "baz"

    # `c` is declared as a field only in the child. The more specific scopes
    # override the more general.
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
        match=r"^Annotated field 'Parent.child' of type 'AbstractChild' has no configured value. Please configure 'Parent.child' with one of the following component subclasses",
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

    @component
    class Parent:
        b: str = Field("bar")
        child: Child = ComponentField(Child)

    p = Parent()

    configure(p, {"a": 10, "b": "foo", "c": [1.5, -1.2]}, name="parent")

    assert (
        click.unstyle(repr(p))
        == """Parent(b="foo", child=Child(a=10, b="foo", c=[1.5, -1.2]))"""
    )
    assert (
        click.unstyle(str(p))
        == """Parent(
    b="foo",
    child=Child(
        a=10,
        b="foo",
        c=[1.5, -1.2]
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
        match="Field 'a' of component 'x' is annotated with type '<class 'int'>', which is not satisfied by default value 4.5.",
    ):
        instance.a
