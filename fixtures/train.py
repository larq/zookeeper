from larq_swarm import register_train_function, HParams, Dataset


@register_train_function
def train(
    build_model,
    dataset,
    hparams,
    output_dir,
    epochs,
    initial_epoch,
    pretrain_dir,
    tensorboard,
):
    assert isinstance(hparams, HParams)
    assert isinstance(dataset, Dataset)
    assert isinstance(output_dir, str)
    assert isinstance(tensorboard, bool)
    assert isinstance(epochs, int)
    assert isinstance(initial_epoch, int)

    model = build_model(hparams, dataset)
    assert model == "foo-model"
    assert dataset.dataset_name == "mnist"
    assert hparams.baz == 3
    assert hparams.baz_overwrite == 42
    print("TESTS PASSED")
