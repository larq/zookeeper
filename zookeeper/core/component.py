"""
Components are generic, modular classes designed to be easily configurable.

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
from typing import Any, Dict, List, Optional, Type

from zookeeper.core import utils
from zookeeper.core.factory_registry import FACTORY_REGISTRY
from zookeeper.core.field import ComponentField, Field

try:  # pragma: no cover
    from colorama import Fore, Style

    RED, YELLOW = Fore.RED, Fore.YELLOW
    BRIGHT, RESET_ALL = Style.BRIGHT, Style.RESET_ALL
except ImportError:  # pragma: no cover
    BRIGHT = RED = RESET_ALL = YELLOW = ""


###################################
# Component class method wrappers #
###################################


def _type_check_and_cache(instance, field: Field, result: Any) -> None:
    """A utility method for `_wrap_getattribute`."""

    if not utils.type_check(result, field.type):
        raise TypeError(
            f"Field '{field.name}' of component '{instance.__component_name__}' is "
            f"annotated with type '{field.type}', which is not satisfied by "
            f"value {repr(result)}."
        )

    object.__setattr__(instance, field.name, result)


def _wrap_getattribute(component_cls: Type) -> None:
    """
    The logic for this overriden `__getattribute__` is as follows:

    During component instantiation, any values passed to `__init__` are stored
    in a dict on the instance `__component_instantiated_field_values__`. This
    means that a priori the `__dict__` of a component instance is empty (of
    non-dunder attributes).

    Field values can come from three sources, in descending order of priority:
      1) A value that was passed into `configure` (e.g. via the CLI), which is
         stored in the `__component_configured_field_values__` dict on the
         component instance or some parent component instance.
      2) A value that was passed in at instantiation, which is stored in the
         `__component_instantiated_field_values__` dict on the current component
         instance (but not any parent instance).
      3) A default value obtained from the `get_default` factory method of a
         field defined on the component class of the current instance if it has
         one, or otherwise from the factory of the field on the component class
         of the nearest parent component instance with a field of the same name,
         et cetera.

    Once we find a field value from one of these three sources, we set the value
    on the instance `__dict__` (i.e. we 'cache' it).

    This means that if we find a value in the instance `__dict__` we can
    immediately return it without worrying about checking the three cases above
    in order. It also means that each look-up other than the first will incur no
    substantial time penalty.
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
                _type_check_and_cache(instance, field, result)
                if name in instance.__component_instantiated_field_values__:
                    del instance.__component_instantiated_field_values__[name]
                return result

        # Next, try Source 2)
        if name in instance.__component_instantiated_field_values__:
            # Type-check, cache, and return the result.
            result = instance.__component_instantiated_field_values__[name]
            _type_check_and_cache(instance, field, result)
            return result

        # Finally, try Source 3)
        try:
            result = field.get_default(instance)
        except AttributeError as e:
            # Find the closest parent with a field of the same name, and
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
        _type_check_and_cache(instance, field, result)
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
def base_getattr(instance, name):
    if utils.is_component_instance:
        return instance.__class__.__base_getattribute__(instance, name)
    return getattr(instance, name)


def _wrap_setattr(component_cls: Type) -> None:
    fn = component_cls.__setattr__  # type: ignore

    @functools.wraps(fn)
    def wrapped_fn(instance, name, value):
        try:
            value_defined_on_class = getattr(component_cls, name)
            if isinstance(value_defined_on_class, Field):
                raise ValueError(
                    "Setting component field values directly is prohibited. Use "
                    "Zookeeper component configuration to set field values."
                )
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


#################################
# Pretty string representations #
#################################


# Indent for nesting in the string representation
INDENT = " " * 4


def _field_key_val_str(key: str, value: Any, color: bool, single_line: bool) -> str:
    if utils.is_component_instance(value):
        if single_line:
            value = repr(value)
        else:
            value = f"\n{INDENT}".join(str(value).split("\n"))
    elif callable(value):
        value = "<callable>"
    elif isinstance(value, str):
        value = f'"{value}"'

    return f"{key}={value}" if color else f"{key}={value}"


def __component_repr__(instance):
    if not instance.__component_configured__:
        return f"<Unconfigured component '{instance.__component_name__}' instance>"

    field_strings = []
    for field_name, field in instance.__component_fields__.items():
        try:
            value = base_getattr(instance, field_name)
        except AttributeError as e:
            if field.allow_missing:
                value = utils.missing
            else:
                raise e from None
        field_strings.append(
            _field_key_val_str(field_name, value, color=False, single_line=True)
        )

    joined_str = ", ".join(field_strings)

    return f"{instance.__class__.__name__}({joined_str})"


def __component_str__(instance):
    if not instance.__component_configured__:
        return f"<Unconfigured component '{instance.__component_name__}' instance>"

    field_strings = []
    for field_name, field in instance.__component_fields__.items():
        try:
            value = base_getattr(instance, field_name)
        except AttributeError as e:
            if field.allow_missing:
                value = utils.missing
            else:
                raise e from None
        field_strings.append(
            _field_key_val_str(field_name, value, color=True, single_line=False)
        )

    joined_str = f",\n{INDENT}".join(field_strings)

    return f"{instance.__class__.__name__}(\n{INDENT}{joined_str}\n)"


##########################
# A component `__init__` #
##########################


def __component_init__(instance, **kwargs):
    """
    Accepts keyword-arguments corresponding to fields defined on the component.
    """

    # Use the `kwargs` to set field values.
    for name, value in kwargs.items():
        if name not in instance.__component_fields__:
            raise TypeError(
                "Keyword arguments passed to component `__init__` must correspond to "
                f"component fields. Received non-matching argument '{name}'."
            )

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
    """
    Configure the component instance with parameters from the `conf` dict.

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

    if name is not None:
        instance.__component_name__ = name

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

            # Set the value on the instance.
            instance.__component_configured_field_values__[
                field.name
            ] = conf_field_value

            # This value has now been 'consumed', so delete it from `conf`.
            del conf[field.name]

        # If there's a value in scope, we don't need to do anything.
        elif field.name in instance.__component_fields_with_values_in_scope__:
            pass

        # If the field explicitly allows values to be missing, there's no need
        # to do anything.
        elif field.allow_missing:
            pass

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

    # Recursively configure any sub-components.
    for field in instance.__component_fields__.values():
        if not isinstance(field, ComponentField):
            continue

        try:
            sub_component_instance = base_getattr(instance, field.name)
        except AttributeError as e:
            if field.allow_missing:
                continue
            raise e from None

        if not utils.is_component_instance(sub_component_instance):
            continue

        full_name = f"{instance.__component_name__}.{field.name}"

        if not sub_component_instance.__component_configured__:
            # Set the component parent so that inherited fields function
            # correctly.
            sub_component_instance.__component_parent__ = instance

            # Extend the field names in scope. All fields with values defined in
            # the scope of the parent are also accessible in the child.
            sub_component_instance.__component_fields_with_values_in_scope__ |= (
                instance.__component_fields_with_values_in_scope__
            )

            # Configure the nested sub-component. The configuration we use
            # consists of all any keys scoped to `field.name`.
            field_name_scoped_conf = {
                a[len(f"{field.name}.") :]: b
                for a, b in conf.items()
                if a.startswith(f"{field.name}.")
            }
            configure(
                sub_component_instance,
                field_name_scoped_conf,
                name=full_name,
                interactive=interactive,
            )

    instance.__component_configured__ = True

    if hasattr(instance.__class__, "__post_configure__"):
        instance.__post_configure__()
