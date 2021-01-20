from zookeeper import Field, component, configure


@component
class A:
    c: float = Field()


# Setting default values on fields before configuration is fine
instance = A()
instance.c = 7.8
configure(instance, {})
assert instance.c == 7.8
