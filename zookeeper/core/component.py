"""Components are generic, modular classes designed to be easily configurable.

Components have configurable fields, which can contain either generic Python
objects or nested sub-components. These are declared with class-level Python
type annotations, in the same way that fields of
[dataclasses](https://docs.python.org/3/library/dataclasses.html) are declared.
After instantiation, components are 'configured' with a configuration
dictionary; this process automatically injects the correct field values into the
component and all sub-components. Component fields can have defaults set, either
in the class definition or passed via `__init__`, but field values passed to
`configure` will always take precedence over these default values.

If a nested sub-component declares a field with the same name as a field in one
of its ancestors, it will receive the same configured field value as the parent
does. Howevever, configuration is scoped: if the field on the child, or on a
_closer anscestor_, is configured with a different value, then that value will
override the one from the original parent.

Configuration can be interactive. In this case, the method will prompt for
missing fields via the CLI.

The following example illustrates the configuration mechanism with and
configuration scoping:

```
@component
class A:
    w: int = Field(3)
    x: int = Field()
    y: str = Field("foo")
    z: float = Field()

@component
class B:
    a: A = ComponentField()
    w: int = Field(5)
    x: int = Field()
    y: str = Field("bar")

@component
class C:
    b: B = ComponentField()
    x: int = Field()
    z: float = Field(3.14)

c = C()
configure(
    c,
    {
        "x": 5,                 # (1)
        "b.x": 10,              # (2)
        "b.a.x": 15,            # (3)

        "b.y": "baz",           # (4)

        "b.a.z": 2.71           # (5)
    }
)
print(c)

>>  C(
        b = B(
            a = A(
                w = 3,          # Default value (not overriden by the parent default value)
                x = 15,         # Configured value (3) overrides (2) overrides (1)
                y = "baz",      # Parent configured value (4) overrides child default value
                z = 2.71        # Configured value (5)
            ),
            w = 5,              # Default value
            x = 10,             # Configured value (2) overrides (1)
            y = "baz"           # Configured value (4) overrides the default value
        ),
        x = 5,                  # Configured value (1)
        z = 3.14                # Default value
    )
```
"""

import functools
import inspect
from typing import AbstractSet, Any, Dict, Iterator, List, Optional, Tuple, Type

from zookeeper.core import utils
from zookeeper.core.factory_registry import FACTORY_REGISTRY
from zookeeper.core.field import ComponentField, Field
from zookeeper.core.utils import ConfigurationError

###################################
# Component class method wrappers #
###################################


def _type_check_and_maybe_cache(instance, field: Field, result: Any) -> None:
    """A utility method for `_wrap_getattribute`."""

    if not utils.type_check(result, field.type):
        raise TypeError(
            f"Field '{field.name}' of component '{instance.__component_name__}' is "
            f"annotated with type '{field.type}', which is not satisfied by "
            f"value {repr(result)}."
        )

    if instance.__component_configured__:
        object.__setattr__(instance, field.name, result)


def _wrap_getattribute(component_cls: Type) -> None:
    """The logic for this overriden `__getattribute__` is as follows:

    During component instantiation, any values passed to `__init__` are stored
    in a dict on the instance `__component_instantiated_field_values__`. This
    means that a priori the `__dict__` of a component instance is empty (of
    non-dunder attributes).

    Field values can come from four sources, in descending order of priority:
      1) A value that was passed into `configure` (e.g. via the CLI), which is
         stored in the `__component_configured_field_values__` dict on the
         component instance or some parent component instance.
      2) A value that was passed in at instantiation, which is stored in the
         `__component_instantiated_field_values__` dict on the current component
         instance (but not any parent instance).
      3) A default value obtained from the `get_default` factory method of a
         field defined on the component class of the current instance if it has
         one. Values obtained in this way are cached in the dict
         `__component_default_field_values__` after generation, to ensure that
         the default value for each field is generated at most once per
         instance.
      4) A value obtained from the nearest parent component instance with a
         field of the same name.

    Once we find a field value from one of these four sources, we perform
    type-checking. If the component has already been configured, we set the
    value on the instance `__dict__` (i.e. we 'cache' it). This is only done
    after configuration because prior to that point we allow using `setattr` to
    override field values. This means that, after configuration, if we find a
    value in the instance `__dict__` we can immediately return it without
    worrying about checking the four cases above in order, or doing
    type-checking. Thus, each subsequent look-up will incur no substantial time
    penalty.
    """
    fn = component_cls.__getattribute__  # type: ignore

    @functools.wraps(fn)
    def base_wrapped_fn(instance, name):
        # If this is not an access to a field, return via the wrapped function.
        if name not in fn(instance, "__component_fields__"):
            return fn(instance, name)

        # If there is a cached value, return it immediately.
        if name in instance.__dict__:
            return instance.__dict__[name]

        field = instance.__component_fields__[name]

        # Next, try Source 1)
        for ancestor_with_field in utils.generate_component_ancestors_with_field(
            instance, field_name=name, include_instance=True
        ):
            if name in ancestor_with_field.__component_configured_field_values__:
                result = ancestor_with_field.__component_configured_field_values__[name]
                # Type-check, cache the result, delete it if possible from the
                # instantiated values, and return it.
                _type_check_and_maybe_cache(instance, field, result)
                if name in instance.__component_instantiated_field_values__:
                    del instance.__component_instantiated_field_values__[name]
                return result

        # Next, try Source 2)
        if name in instance.__component_instantiated_field_values__:
            # Type-check, cache, and return the result.
            result = instance.__component_instantiated_field_values__[name]
            _type_check_and_maybe_cache(instance, field, result)
            if instance.__component_configured__:
                del instance.__component_instantiated_field_values__[name]
            return result

        # Next, try Sources 3)
        if name in instance.__component_default_field_values__:
            return instance.__component_default_field_values__[name]
        try:
            result = field.get_default(instance)
            # Set the correct component parent if `result` is a sub-component.
            if isinstance(field, ComponentField):
                result.__component_parent__ = instance
            instance.__component_default_field_values__[name] = result
        except (ConfigurationError, AttributeError) as e:
            # And if necessary fall back to Source 4)
            #     Find the closest parent with a field of the same name, and
            # recurse.
            parent_instance = next(
                utils.generate_component_ancestors_with_field(instance, name), None
            )
            try:
                result = parent_instance.__base_getattribute__(name)  # type: ignore
            except AttributeError:
                # From here we raise the original exception instead because it
                # will correctly refer to this component rather than some parent
                # component.
                raise e from None

        # Type-check, cache, and return the result.
        _type_check_and_maybe_cache(instance, field, result)
        return result

    component_cls.__base_getattribute__ = base_wrapped_fn

    # `__base_getattribute__`, defined above, is correct for all values unless
    # they are @factory instances. If the result is an @factory instance, we
    # would like `build()` to be implicitly called and the result returned
    # instead of the actual @factory instance.
    #
    # However, internally we would like to still be able to access the @factory
    # instance, e.g. during configuration of sub-components.
    #
    # The solution is for `__base_getattribute__` to *not* call `build()`
    # implicitly - this function can be used internally. Then we create a
    # general-purpose `__getattribute__` by wrapping `__base_getattribute__`
    # once more and inspecting the returned value.

    @functools.wraps(base_wrapped_fn)
    def wrapped_fn(instance, name):
        result = base_wrapped_fn(instance, name)
        if name in base_wrapped_fn(
            instance, "__component_fields__"
        ) and utils.is_factory_instance(result):
            return result.build()
        return result

    component_cls.__getattribute__ = wrapped_fn


# Syntatic sugar around `__base_getattribute__`, similar to `getattr` for
# calling `__getattribute__` in the standard library.
def base_getattr(instance, name: str):
    if utils.is_component_instance(instance):
        return instance.__class__.__base_getattribute__(instance, name)  # type: ignore
    return getattr(instance, name)


def base_hasattr(instance, name: str):
    """Like the default `hasattr`, except that this does not throw a
    `ConfigurationError` when the accessed attribute is not yet configured, and will
    return `True` for fields that have `allow_missing=True` and have not been provided
    any value."""
    return name in dir(instance)


def _wrap_setattr(component_cls: Type) -> None:
    fn = component_cls.__setattr__  # type: ignore

    @functools.wraps(fn)
    def wrapped_fn(instance, name, value):
        try:
            value_defined_on_class = getattr(component_cls, name)
            if isinstance(value_defined_on_class, Field):
                if instance.__component_configured__:
                    raise ValueError(
                        "Setting already configured component field values directly is "
                        "prohibited. Use Zookeeper component configuration to set field "
                        "values."
                    )
                if utils.is_component_instance(value):
                    if not isinstance(value_defined_on_class, ComponentField):
                        raise ValueError(
                            "Component instances can only be set as values for "
                            "`ComponentField`s, but "
                            f"{instance.__component_name__}.{name} is a `Field`."
                        )
                    if value.__component_configured__:
                        raise ValueError(
                            "Component instances can only be set as values if they are "
                            "not yet configured."
                        )
                instance.__component_fields_with_values_in_scope__.add(name)
                instance.__component_instantiated_field_values__[name] = value
                return
        except AttributeError:
            pass
        return fn(instance, name, value)

    component_cls.__setattr__ = wrapped_fn


def _wrap_delattr(component_cls: Type) -> None:
    fn = component_cls.__delattr__  # type: ignore

    @functools.wraps(fn)
    def wrapped_fn(instance, name):
        if name in instance.__component_fields__:
            raise ValueError("Deleting component field values is prohibited.")
        return fn(instance, name)

    component_cls.__delattr__ = wrapped_fn


def _wrap_dir(component_cls: Type) -> None:
    fn = component_cls.__dir__  # type: ignore

    @functools.wraps(fn)
    def wrapped_fn(instance) -> List[str]:
        return list(set(fn(instance)) | set(instance.__component_fields__.keys()))

    component_cls.__dir__ = wrapped_fn


######################################
# Implement the `ItemsView` protocol #
######################################

# Implementing these three methods makes `dict(component_instance)`,
# `list(component_instance)`, et cetera be something sensible.
#
# In particular, `dict(component_instance)` has field names as keys and field
# values as values, giving a flattened view of the entire component tree
# including all sub-components. This dictionary is precisely the configuration
# that is required to completely re-create the component, including all
# sub-components, i.e. `configure(new_instance, dict(old_instance))`.


def __component_len__(instance) -> int:
    return len(list(iter(instance)))


def __component_contains__(instance, key: str) -> bool:
    if not isinstance(key, str):
        return False

    if "." not in key:
        # The base case is whether or not this is a component field defined
        # on this class, that also has a value in scope (which excludes the
        # `allow_missing` fields without a configured value).
        return (
            key in instance.__component_fields__
            and key in instance.__component_fields_with_values_in_scope__
        )

    # The recusive case is to split the key and check the sub-component.
    key_prefix = key.split(".")[0]
    if isinstance(instance.__component_fields__.get(key_prefix, None), ComponentField):
        try:
            sub_component = base_getattr(instance, key_prefix)
            if utils.is_component_instance(sub_component):
                return key[len(key_prefix) + 1 :] in sub_component
        except AttributeError:
            pass

    return False


def __component_iter__(instance) -> Iterator[Tuple[str, Any]]:
    for field_name, field in instance.__component_fields__.items():
        # Get the field value. If there's an attribute error (possible
        # because of `allow_missing`), just skip.
        try:
            field_value = base_getattr(instance, field_name)
        except AttributeError:
            continue

        # Check if the value is inherited and if so skip.
        parent_instance = next(
            utils.generate_component_ancestors_with_field(instance, field_name),
            None,
        )
        if (
            parent_instance is not None
            and base_getattr(parent_instance, field_name) is field_value
        ):
            continue

        # If this is *not* a sub-component, yield the value directly.
        if not isinstance(field, ComponentField) or not utils.is_component_instance(
            field_value
        ):
            yield field_name, field_value

        # Otherwise, yield the name of the sub-component class and
        # recursively yield from the sub-component.
        else:
            yield field_name, field_value.__class__.__qualname__
            for sub_field_name, sub_field_value in iter(field_value):
                yield f"{field_name}.{sub_field_name}", sub_field_value


#################################
# Pretty string representations #
#################################


# Indent for nesting in the string representation
INDENT = " " * 4


def _list_field_strings(instance, color: bool, single_line: bool) -> Iterator[str]:
    for field_name, field in instance.__component_fields__.items():
        try:
            value = base_getattr(instance, field_name)
        except (ConfigurationError, AttributeError) as e:
            if isinstance(e, AttributeError) and field.allow_missing:
                value = utils.missing
            else:
                raise e from None

        parent_instance = next(
            utils.generate_component_ancestors_with_field(instance, field_name), None
        )
        if value is not utils.missing and parent_instance is not None:
            is_inherited = base_getattr(parent_instance, field_name) is value  # type: ignore
        else:
            is_inherited = False

        if utils.is_component_instance(value):
            if is_inherited:
                value = "<inherited component instance>"
            elif single_line:
                value = repr(value)
            else:
                value = f"\n{INDENT}".join(str(value).split("\n"))
        elif is_inherited:
            value = "<inherited value>"
        elif callable(value):
            value = "<callable>"
        elif isinstance(value, str):
            value = f'"{value}"'

        yield f"{field_name}={value}" if color else f"{field_name}={value}"


def __component_repr__(instance):
    if not instance.__component_configured__:
        return f"<Unconfigured component '{instance.__component_name__}' instance>"
    joined_str = ", ".join(_list_field_strings(instance, color=False, single_line=True))
    return f"{instance.__class__.__name__}({joined_str})"


def __component_str__(instance):
    if not instance.__component_configured__:
        return f"<Unconfigured component '{instance.__component_name__}' instance>"
    joined_str = f",\n{INDENT}".join(
        _list_field_strings(instance, color=True, single_line=False)
    )
    return f"{instance.__class__.__name__}(\n{INDENT}{joined_str}\n)"


##########################
# A component `__init__` #
##########################


def __component_init__(instance, **kwargs):
    """Accepts keyword-arguments corresponding to fields defined on the component."""

    # Use the `kwargs` to set field values.
    for name, value in kwargs.items():
        if name not in instance.__component_fields__:
            raise TypeError(
                "Keyword arguments passed to component `__init__` must correspond to "
                f"component fields. Received non-matching argument '{name}'."
            )
        if utils.is_component_instance(value):
            if value.__component_configured__:
                raise ValueError(
                    "Sub-component instances passed to the `__init__` method of a "
                    "component must not already be configured. Received configured "
                    f"component argument '{name}={repr(value)}'."
                )
            # Set the component parent correctly if the value being passed in
            # does not already have a parent.
            if value.__component_parent__ is None:
                value.__component_parent__ = instance

    # This will contain default field values from the fields defined on this
    # instance.
    instance.__component_default_field_values__ = {}

    # Save a shallow-clone of the arguments.
    instance.__component_instantiated_field_values__ = {**kwargs}

    # This will contain configured field values that apply to this instance, if
    # any, which override everything else.
    instance.__component_configured_field_values__ = {}

    # This will contain the names of every field of this instance and all
    # component ancestors for which a value is defined. More names will be added
    # during configuration.
    instance.__component_fields_with_values_in_scope__ = set(
        field.name
        for field in instance.__component_fields__.values()
        if field.has_default
    ) | set(kwargs)


#####################################
# Recursive component configuration #
#####################################


def configure_component_instance(
    instance,
    conf: Dict[str, Any],
    name: str,
    fields_in_scope: AbstractSet[str],
    interactive: bool,
):
    """Configure the component instance with parameters from the `conf` dict.

    This method is recursively called for each component instance in the component tree
    by the exported `configure` function.
    """
    if name is not None:
        instance.__component_name__ = name

    if hasattr(instance.__class__, "__pre_configure__"):
        conf = instance.__pre_configure__({**conf})
        if not isinstance(conf, dict):
            raise ValueError(
                "Expected the `__pre_configure__` method of component "
                f"'{instance.__component_name__}' to return a dict of configuration, "
                f"but received: {conf}"
            )

    # Extend the field names in scope.
    instance.__component_fields_with_values_in_scope__ |= fields_in_scope

    # Set the correct value for each field.
    for field in instance.__component_fields__.values():
        full_name = f"{instance.__component_name__}.{field.name}"
        field_type_name = (
            field.type.__name__ if inspect.isclass(field.type) else str(field.type)
        )

        if isinstance(field, ComponentField):
            # Create a list of all component subclasses of the field type, and
            # add to the list all factory classes which can build the type (or
            # any subclass of the type).
            component_subclasses = list(utils.generate_component_subclasses(field.type))
            for type_subclass in utils.generate_subclasses(field.type):
                component_subclasses.extend(FACTORY_REGISTRY.get(type_subclass, []))

        if field.name in conf:
            conf_field_value = conf[field.name]

            if isinstance(field, ComponentField):
                # The configuration value could be a string specifying a component
                # or factory class to instantiate.
                if len(component_subclasses) > 0 and isinstance(conf_field_value, str):
                    for subclass in component_subclasses:
                        if (
                            conf_field_value == subclass.__name__
                            or conf_field_value == subclass.__qualname__
                            or utils.convert_to_snake_case(conf_field_value)
                            == utils.convert_to_snake_case(subclass.__name__)
                        ):
                            conf_field_value = subclass()
                            break

            # If this isn't the case, then it's a user type error, but we don't
            # throw here and instead let the run-time type-checking take care of
            # it (which will provide a better error message).
            if utils.is_component_instance(conf_field_value):
                # Set the component parent so that field value inheritence will
                # work correctly.
                conf_field_value.__component_parent__ = instance

            # Set the value on the instance.
            instance.__component_configured_field_values__[
                field.name
            ] = conf_field_value

        # If there's a value in scope, we don't need to do anything.
        elif field.name in instance.__component_fields_with_values_in_scope__:
            pass

        # If the field explicitly allows values to be missing, there's no need
        # to do anything.
        elif field.allow_missing:
            continue

        # If there is only one concrete component subclass of the annotated
        # type, we assume the user must intend to use that subclass, and so
        # instantiate and use an instance automatically.
        elif isinstance(field, ComponentField) and len(component_subclasses) == 1:
            component_cls = list(component_subclasses)[0]
            utils.warn(
                f"'{utils.type_name_str(component_cls)}' is the only concrete component "
                f"class that satisfies the type of the annotated field '{full_name}'. "
                "Using an instance of this class by default.",
            )
            # This is safe because we don't allow custom `__init__` methods.
            conf_field_value = component_cls()
            # Set the component parent so that field value inheritence will work
            # correctly.
            conf_field_value.__component_parent__ = instance

            # Set the value on the instance.
            instance.__component_configured_field_values__[
                field.name
            ] = conf_field_value

        # If we are running interactively, prompt for a value.
        elif interactive:
            if isinstance(field, ComponentField):
                if len(component_subclasses) > 0:
                    component_cls = utils.prompt_for_component_subclass(
                        full_name, component_subclasses
                    )
                    # This is safe because we don't allow custom `__init__` methods.
                    conf_field_value = component_cls()
                else:
                    raise ValueError(
                        "No component or factory class is defined which satisfies the "
                        f"type {field_type_name} of field {full_name}. If such a class "
                        "has been defined, it must be imported before calling "
                        "`configure`."
                    )
            else:
                conf_field_value = utils.prompt_for_value(full_name, field.type)

            # Set the value on the instance.
            instance.__component_configured_field_values__[
                field.name
            ] = conf_field_value

        # Otherwise, raise an appropriate error.
        else:
            if isinstance(field, ComponentField):
                if len(component_subclasses) > 0:
                    raise ValueError(
                        f"Component field '{full_name}' of type '{field_type_name}' "
                        f"has no default or configured class. Please configure "
                        f"'{full_name}' with one of the following @component or "
                        "@factory classes:"
                        + "\n    ".join(
                            [""]
                            + list(utils.type_name_str(c) for c in component_subclasses)
                        )
                    )
                else:
                    raise ValueError(
                        f"Component field '{full_name}' of type '{field_type_name}' "
                        f"has no default or configured class. No defined @component "
                        "or @factory class satisfies this type. Please define an "
                        f"@component class subclassing '{field_type_name}', or an "
                        "@factory class with a `build()` method returning a "
                        f"'{field_type_name}' instance. This class must be imported "
                        "before invoking `configure()`."
                    )
            raise ValueError(
                "No configuration value found for annotated field "
                f"'{full_name}' of type '{field_type_name}'."
            )

        # At this point we are certain that this field has has a value, so keep
        # track of that fact.
        instance.__component_fields_with_values_in_scope__.add(field.name)

    # Check that all `conf` values are being used, and throw if we've been
    # passed an un-used option.
    for key in conf:
        error = ValueError(
            f"Key '{key}' does not correspond to any field of component "
            f"'{instance.__component_name__}'."
            "\n\n"
            "If you have nested components as follows:\n\n"
            "```\n"
            "@component\n"
            "class ChildComponent:\n"
            "    a: int = Field(0)\n"
            "\n"
            "@task\n"
            "class SomeTask:\n"
            "    child: ChildComponent = ComponentField(ChildComponent)\n"
            "    def run(self):\n"
            "        print(self.child.a)\n"
            "```\n\n"
            "then trying to configure `a=<SOME_VALUE>` will fail. You instead need to "
            "fully qualify the key name, and configure the value with "
            "`child.a=<SOME_VALUE>`."
        )

        if "." in key:
            scoped_component_name = key.split(".")[0]
            if not (
                scoped_component_name in instance.__component_fields__
                and isinstance(
                    instance.__component_fields__[scoped_component_name], ComponentField
                )
            ):
                raise error
        elif key not in instance.__component_fields__:
            raise error

    instance.__component_configured__ = True

    if hasattr(instance.__class__, "__post_configure__"):
        instance.__post_configure__()

    return conf


######################
# Exported functions #
######################


def component(cls: Type):
    """A decorator which turns a class into a Zookeeper component."""

    if not inspect.isclass(cls):
        raise TypeError("Only classes can be decorated with @component.")

    if inspect.isabstract(cls):
        raise TypeError("Abstract classes cannot be decorated with @component.")

    if utils.is_component_class(cls):
        raise TypeError(
            f"The class {cls.__name__} is already a component; the @component decorator "
            "cannot be applied again."
        )

    if cls.__init__ not in (object.__init__, __component_init__):
        # A component class could have `__component_init__` as its init method
        # if it inherits from a component.
        raise TypeError("Component classes must not define a custom `__init__` method.")
    cls.__init__ = __component_init__

    if hasattr(cls, "__pre_configure__"):
        if not callable(cls.__pre_configure__):
            raise TypeError(
                "The `__pre_configure__` attribute of a @component class must be a "
                "method."
            )
        call_args = inspect.signature(cls.__pre_configure__).parameters
        if len(call_args) > 2 or len(call_args) > 1 and "self" not in call_args:
            raise TypeError(
                "The `__pre_configure__` method of a @component class must take no "
                f"arguments except `self` and `conf`, but "
                f"`{cls.__name__}.__pre_configure__` accepts arguments "
                f"{tuple(name for name in call_args)}."
            )

    if hasattr(cls, "__post_configure__"):
        if not callable(cls.__post_configure__):
            raise TypeError(
                "The `__post_configure__` attribute of a @component class must be a "
                "method."
            )
        call_args = inspect.signature(cls.__post_configure__).parameters
        if len(call_args) > 1 or len(call_args) == 1 and "self" not in call_args:
            raise TypeError(
                "The `__post_configure__` method of a @component class must take no "
                f"arguments except `self`, but `{cls.__name__}.__post_configure__` "
                f"accepts arguments {tuple(name for name in call_args)}."
            )

    # Populate `__component_fields__` with all fields defined on this class and
    # all superclasses. We have to go through the MRO chain and collect them in
    # reverse order so that they are correctly overriden.
    fields = {}
    for base_class in reversed(inspect.getmro(cls)):
        for name, value in base_class.__dict__.items():
            if isinstance(value, Field):
                fields[name] = value

    if len(fields) == 0:
        utils.warn(f"Component {cls.__name__} has no defined fields.")

    # Throw an error if there is a field defined on a superclass that has been
    # overriden with a non-Field value.
    for name in dir(cls):
        if name in fields and not isinstance(getattr(cls, name), Field):
            super_class = fields[name].host_component_class
            raise ValueError(
                f"Field '{name}' is defined on super-class {super_class.__name__}. "
                f"In subclass {cls.__name__}, '{name}' has been overriden with value: "
                f"{getattr(cls, name)}.\n\n"
                f"If you wish to change the default value of field '{name}' in a "
                f"subclass of {super_class.__name__}, please wrap the new default "
                "value in a new `Field` instance."
            )

    cls.__component_fields__ = fields

    # Override class methods to correctly interact with component fields.
    _wrap_getattribute(cls)
    _wrap_setattr(cls)
    _wrap_delattr(cls)
    _wrap_dir(cls)

    # Implement the `ItemsView` protocol
    if hasattr(cls, "__len__") and cls.__len__ != __component_len__:
        raise TypeError("Component classes must not define a custom `__len__` method.")
    cls.__len__ = __component_len__
    if hasattr(cls, "__contains__") and cls.__contains__ != __component_contains__:
        raise TypeError(
            "Component classes must not define a custom `__contains__` method."
        )
    cls.__contains__ = __component_contains__
    if hasattr(cls, "__iter__") and cls.__iter__ != __component_iter__:
        raise TypeError("Component classes must not define a custom `__iter__` method.")
    cls.__iter__ = __component_iter__

    # Components should have nice `__str__` and `__repr__` methods.
    cls.__str__ = __component_str__
    cls.__repr__ = __component_repr__

    # These will be overriden during configuration.
    cls.__component_name__ = cls.__name__
    cls.__component_parent__ = None
    cls.__component_configured__ = False

    return cls


def configure(
    instance,
    conf: Dict[str, Any],
    name: Optional[str] = None,
    interactive: bool = False,
):
    """Configure the component instance with parameters from the `conf` dict.

    Configuration passed through `conf` takes precedence over and will
    overwrite any values already set on the instance - either class defaults
    or those set in `__init__`.
    """
    # Only component instances can be configured.
    if not utils.is_component_instance(instance):
        raise TypeError(
            "Only @component, @factory, and @task instances can be configured. "
            f"Received: {instance}."
        )

    # Configuration can only happen once.
    if instance.__component_configured__:
        raise ValueError(
            f"Component '{instance.__component_name__}' has already been configured."
        )

    # Maintain a FIFO queue of component instances that need to be configured,
    # along with the config dict, name that should be passed, and a set of field
    # names that are in-scope for component field inheritence.
    #     This queue allows us to recursively configure component instances in
    # the component tree in a top-down, breadth-first order.
    fifo_component_queue = [(instance, conf, name, frozenset(conf.keys()))]

    while len(fifo_component_queue) > 0:
        (
            current_instance,
            current_conf,
            current_name,
            current_fields_in_scope,
        ) = fifo_component_queue.pop(0)

        if current_instance.__component_configured__:
            continue

        current_conf = configure_component_instance(
            current_instance,
            conf=current_conf,
            name=current_name,
            fields_in_scope=current_fields_in_scope,
            interactive=interactive,
        )

        # Collect the sub-component instances that need to be recursively
        # configured, and add them to the queue.
        for field in current_instance.__component_fields__.values():
            if not isinstance(field, ComponentField):
                continue

            try:
                sub_component_instance = base_getattr(current_instance, field.name)
            except (AttributeError, ConfigurationError) as e:
                if field.allow_missing:
                    continue
                raise e from None

            if (
                not utils.is_component_instance(sub_component_instance)
                or sub_component_instance.__component_configured__
            ):
                continue

            # Generate the configuration dict that will be used with the nested
            # sub-component. This consists of all keys scoped to `field.name`.
            sub_component_conf = {
                a[len(f"{field.name}.") :]: b
                for a, b in current_conf.items()
                if a.startswith(f"{field.name}.")
            }

            # The name of the sub-component is full-stop-delimited.
            sub_component_name = f"{current_instance.__component_name__}.{field.name}"

            # At this point the current instance has already been configured so
            # we know that every one of its fields is in scope.
            sub_component_fields_in_scope = current_fields_in_scope | frozenset(
                current_instance.__component_fields__.keys()
            )

            # Add the sub-component to the end of the queue.
            fifo_component_queue.append(
                (
                    sub_component_instance,
                    sub_component_conf,
                    sub_component_name,
                    sub_component_fields_in_scope,
                )
            )
