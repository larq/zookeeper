from inspect import getmro, isclass

from prompt_toolkit import print_formatted_text
from typeguard import check_type

from zookeeper.utils import (
    convert_to_snake_case,
    get_concrete_subclasses,
    prompt_for_component,
    promt_for_param_value,
)

try:  # pragma: no cover
    from colorama import Fore

    BLUE, YELLOW, RESET = Fore.BLUE, Fore.YELLOW, Fore.RESET
except ImportError:  # pragma: no cover
    BLUE = YELLOW = RESET = ""

# Indent for nesting in the string representation
INDENT = " " * 4


def defined_on_self_or_ancestor(self, name):
    """
    Test if the annotation `name` exists on `self` or a component ancestor of
    `self` with a defined value. If so, return the instance on which `name` is
    defined. Otherwise, return `None`.
    """

    # Using `hasattr` is not safe, as it is implemented with `getattr` wrapped
    # in a try-catch (Python is terrible), so we need to check `dir(self)`.
    if name in self.__component_annotations__ and name in dir(self):
        return self
    if self.__component_parent__:
        return defined_on_self_or_ancestor(self.__component_parent__, name)
    return None


def str_key_val(key, value, color=True, single_line=False):
    if isinstance(value, Component):
        if single_line:
            value = repr(value)
        else:
            value = f"\n{INDENT}".join(str(value).split("\n"))
    elif callable(value):
        value = "<callable>"
    elif type(value) == str:
        value = f'"{value}"'
    space = "" if single_line else " "
    return (
        f"{BLUE}{key}{RESET}{space}={space}{YELLOW}{value}{RESET}"
        if color
        else f"{key}{space}={space}{value}"
    )


class Component:
    """
    A generic, modular component class designed to be easily configurable.

    Components can have configurable parameters, which can be either generic
    Python objects or nested sub-components. These are declared with class-level
    Python type annotations, in the same way that elements of
    [dataclasses](https://docs.python.org/3/library/dataclasses.html) are
    declared. After instantiation, components are 'configured' with a
    configuration dictionary; this process automatically injects the correct
    parameters into the component and all subcomponents. Component parameters
    can have defaults set, either in the class definition or passed via
    `__init__`, but configuration values passed to `configure` will always take
    precedence over these values.

    If a nested sub-component child declares a parameter with the same name as a
    parameter in one of its ancestors, it will receive the same configured value
    as the parent does. Howevever, configuration is scoped: if the parameter on
    the child, or on a _closer anscestor_, is configured with a different value,
    then that value will override the one from the original parent.

    Configuration can be interactive. In this case, the method will prompt for
    missing parameters via the CLI.

    The following example illustrates the configuration mechanism with scoped
    configuration:

    ```
    class A(Component):
        x: int
        z: float

        def __call__(self):
            return str(self.x) + "_" + str(self.z)

    class B(Component):
        a: A
        y: str = "foo"

        def __call__(self):
            return self.y + " / " + self.a()

    class C(Component):
        b: B
        x: int
        z: float = 3.14

        def __call__(self):
            return str(self.x) + "_" + str(self.z) + " / " + self.b()


    c = C()
    c.configure({
        "x": 5,                     # (1)
        "b.x": 10,                  # (2)
        "b.a.x": 15,                # (3)

        "b.y": "foo",               # (4)

        "b.z": 2.71                 # (5)
    })
    print(c)

    >>  C(
            b = B(
                a = A(
                    x = 15,         # (3) overrides (2) overrides (1)
                    z = 2.71        # Inherits from parent: (5)
                ),
                y = "foo"           # (4) overrides the default
            ),
            x = 5,                  # Only (1) applies
            z = 3.14                # The default is taken
        )
    ```
    """

    # The name of the component.
    __component_name__ = None

    # If this instance is nested in another component, a reference to that
    # parent instance.
    __component_parent__ = None

    # All annotations which apply to the class, including those inherited from
    # superclasses. This is populated in `__init__`.
    __component_annotations__ = {}

    def __init__(self, **kwargs):
        """
        `kwargs` may only contain argument names corresponding to component
        annotations. The passed values will be set on the instance.
        """

        # Populate `self.__component_annotations__` with all annotations set on
        # this class and all superclasses. We have to go through the MRO chain
        # and collect them in reverse order so that they are correctly
        # overriden.
        annotations = {}
        for base_class in reversed(getmro(self.__class__)):
            annotations.update(getattr(base_class, "__annotations__", {}))
        annotations.update(getattr(self, "__annotations__", {}))
        self.__component_annotations__ = annotations

        for k, v in kwargs.items():
            if k in self.__component_annotations__:
                setattr(self, k, v)
            else:
                raise ValueError(
                    f"Argument '{k}' passed to `__init__` does not correspond to "
                    f"any annotation of {self.__class__.__name__}."
                )

    def __init_subclass__(cls, *args, **kwargs):
        # Prohibit overriding `__init__` in subclasses.
        if cls.__init__ != Component.__init__:
            raise ValueError(
                f"Overriding `__init__` in component {cls}. `Component.__init__` "
                "must not be overriden, as doing so breaks the built-in "
                "hydration mechanism. It should be unnecessary to override "
                "`__init__`: the default `Component.__init__` implementation "
                "accepts keyword-arguments matching defined class attributes "
                "and sets the corresponding attribute values on the instance."
            )

    def __getattr__(self, name):
        # This is only called if the attribute doesn't exist on the instance
        # (i.e. on `self`, on the class, or on any superclass). When this
        # happens, if `name` is a declared annotation which is also declared on
        # some ancestor with a defined value for `name`, return that value.
        if name in self.__component_annotations__:
            ancestor = defined_on_self_or_ancestor(self, name)
            if ancestor is not None:
                return getattr(ancestor, name)
        raise AttributeError(
            f"Component {self.__component_name__} does not have any attribute {name}."
        )

    def __setattr__(self, name, value):
        # Type-check annotated values.
        if name in self.__component_annotations__:
            annotation = self.__component_annotations__[name]
            check_type(name, value, annotation)
        super().__setattr__(name, value)

    def __str__(self):
        params = f",\n{INDENT}".join(
            [str_key_val(k, getattr(self, k)) for k in self.__component_annotations__]
        )
        return f"{self.__class__.__name__}(\n{INDENT}{params}\n)"

    def __repr__(self):
        params = ", ".join(
            [
                str_key_val(k, getattr(self, k), color=False, single_line=True)
                for k in self.__component_annotations__
            ]
        )
        return f"{self.__class__.__name__}({params})"

    def configure(self, conf, name=None, parent=None, interactive=False):
        """
        Configure the component instance with parameters from the `conf` dict.

        Configuration passed through `conf` takes precedence over and will
        overwrite any values already set on the instance - either class defaults
        or those passed via `__init__`.
        """

        self.__component_name__ = name or self.__class__.__name__
        self.__component_parent__ = parent

        # Divide the annotations into those which are and those which are not
        # nested components. We will process the non-component parameters first,
        # because nested components may depend on parameter (non-component)
        # values set in the parent.
        non_component_annotations = []
        component_annotations = []

        for k, v in self.__component_annotations__.items():
            # We have to be careful because `v` can be a `typing.Type` subclass
            # e.g. `typing.List[float]`.
            #
            # In Python 3.7+, this will cause `issubclass(v, Component)` to
            # throw, but `isclass(v)` will be `False`.
            #
            # In Python 3.6, `isclass(v)` will be `True`, but fortunately
            #  `issubclass(v, Component)` won't throw.
            if isclass(v) and issubclass(v, Component):
                component_annotations.append((k, v))
            else:
                non_component_annotations.append((k, v))

        # Process non-component annotations
        for k, v in non_component_annotations:
            param_name = f"{self.__component_name__}.{k}"
            param_type_name = v.__name__ if isclass(v) else v

            # The value from the `conf` dict takes priority.
            if k in conf:
                param_value = conf[k]
                setattr(self, k, param_value)

            # If there's no config value but a value is already set on the
            # instance (or a parent), no action needs to be taken.
            elif defined_on_self_or_ancestor(self, k) is not None:
                pass

            # If we are running interactively, prompt for the missing value. Add
            # it to the configuration so that it gets passed to any children.
            elif interactive:
                param_value = promt_for_param_value(param_name, v)
                setattr(self, k, param_value)
                conf[k] = param_value

            # If we're not running interactively and there's no value anywhere,
            # raise an error.
            else:
                raise ValueError(
                    "No configuration value found for annotated parameter "
                    f"'{param_name}' of type '{param_type_name}'."
                )

        # Process nested component annotations
        for k, v in component_annotations:
            param_name = f"{self.__component_name__}.{k}"
            param_type_name = v.__qualname__
            concrete_subclasses = get_concrete_subclasses(v)

            # The value from the `conf` dict takes priority.
            if k in conf:
                instance = conf[k]
                # The value might have been parsed from command-line arguments,
                # in which case we expect a string naming the class.
                if isinstance(instance, str):
                    for component_cls in concrete_subclasses:
                        if (
                            instance == component_cls.__name__
                            or instance == component_cls.__qualname__
                            or instance == convert_to_snake_case(component_cls.__name__)
                        ):
                            instance = component_cls()
                            conf[k] = instance
                            break
                setattr(self, k, instance)

            # If there's no config value but a value is already set on the
            # instance (or a parent), no action needs to be taken.
            elif defined_on_self_or_ancestor(self, k) is not None:
                pass

            # If there is no concrete subclass of `v`, raise an error.
            elif len(concrete_subclasses) == 0:
                raise ValueError(
                    "There is no defined, non-abstract class that can be instantiated "
                    f"to satisfy the annotated parameter '{param_name}' of type "
                    f"'{param_type_name}'."
                )

            # If there is only one concrete subclass of `v`, instantiate an
            # instance of that class.
            elif len(concrete_subclasses) == 1:
                component_cls = list(concrete_subclasses)[0]
                print_formatted_text(
                    f"'{component_cls.__qualname__}' is the only concrete component "
                    "class that satisfies the type of the annotated parameter "
                    f"'{param_name}'. Using an instance of this class by default."
                )
                # This is safe because we ban overriding `__init__`.
                instance = component_cls()
                setattr(self, k, instance)

            # If we are running interactively and there is more than one
            # concrete subclass of `v`, prompt for the concrete subclass to
            # instantiate. Add the instance to the configuation so that is can
            # get passed to any children.
            elif interactive:
                component_cls = prompt_for_component(param_name, v)
                # The is safe because we ban overriding `__init__`.
                instance = component_cls()
                setattr(self, k, instance)

            # If we're not running interactively and there is more than one
            # concrete subclass of `v`, raise an error.
            else:
                raise ValueError(
                    f"Annotated parameter '{param_name}' of type '{param_type_name}' "
                    f"has no configured value. Please configure '{param_name}' with one "
                    f"of the following concrete subclasses of '{param_type_name}':\n    "
                    + "\n    ".join(list(c.__qualname__ for c in concrete_subclasses))
                )

            # Configure the sub-component. The configuration we use consists of
            # all non-scoped keys and any keys scoped to `k`, where the keys
            # scoped to `k` override the non-scoped keys.
            non_scoped_conf = {a: b for a, b in conf.items() if "." not in a}
            k_scoped_conf = {
                a[len(f"{k}.") :]: b for a, b in conf.items() if a.startswith(f"{k}.")
            }
            nested_conf = {**non_scoped_conf, **k_scoped_conf}
            getattr(self, k).configure(
                nested_conf, name=param_name, parent=self, interactive=interactive
            )

        # Validate all parameters.
        self.validate_configuration()

    def validate_configuration(self):
        """
        Called automatically at the end of `configure`. Subclasses should
        override this method to provide fine-grained parameter validation.
        Invalid configuration should be flagged by raising an error with a
        descriptive error message.
        """

        # Checking for missing values is done in `configure`. Type-checking is
        # done in `__setattr__`.
        pass
