from larq_swarm import register_model, register_hparams, HParams


@register_model
def foo(hparams, dataset):
    return "foo-model"


@register_hparams(foo)
def bar():
    return HParams(baz=3)
