import tensorflow_datasets as tfds
from zookeeper.data import Dataset

MODEL_REGISTRY = {}
HPARAMS_REGISTRY = {}
DATA_REGISTRY = {dataset: {} for dataset in tfds.list_builders()}


class DatasetNotFoundError(ValueError):
    def __init__(self, name):
        available_sets = "\n\t- ".join([""] + list(DATA_REGISTRY.keys()))
        err = f"No dataset named {name} available.\nAvailable datasets:{available_sets}"
        ValueError.__init__(self, err)


class PreprocessNotFoundError(ValueError):
    def __init__(self, dataset_name, name):
        available_prepro_fns = "\n\t- ".join(
            [""] + list(DATA_REGISTRY[dataset_name].keys())
        )
        err = (
            f"No preprocessing named {name} registered for dataset {dataset_name}.\nAvailable preprocessing functions:{available_prepro_fns}"
            if available_prepro_fns
            else f"No preprocessing functions registered for dataset {dataset_name}."
        )
        ValueError.__init__(self, err)


class ModelNotFoundError(ValueError):
    def __init__(self, name):
        available_models = "\n\t- ".join([""] + list(MODEL_REGISTRY.keys()))
        err = f"No model named {name} registered.\nAvailable models:{available_models}"
        ValueError.__init__(self, err)


class HParamsNotFoundError(ValueError):
    def __init__(self, model_name, name):
        err = f"No hyperparameters for model {model_name} registered."
        if model_name in HPARAMS_REGISTRY:
            available_hparam_sets = "\n\t- ".join(
                [""] + list(HPARAMS_REGISTRY[model_name].keys())
            )
            err = f"No hyperparameter set named {name} registered for model {model_name}.\nAvailable hyperparameter sets:{available_hparam_sets}"
        ValueError.__init__(self, err)


def register_preprocess(dataset_name, input_shape=None):
    def register_preprocess_fn(fn):
        if not callable(fn):
            raise ValueError("Preprocess function must be callable")
        name = fn.__name__
        if dataset_name not in DATA_REGISTRY:
            raise DatasetNotFoundError(dataset_name)
        data_preprocess_fns = DATA_REGISTRY[dataset_name]
        if name in data_preprocess_fns:
            raise ValueError(
                f"Cannot register duplicate preprocessing ({name}) for dataset ({dataset_name})"
            )
        if input_shape:
            setattr(fn, "input_shape", input_shape)
        data_preprocess_fns[name] = fn
        return fn

    return register_preprocess_fn


def register_model(model):
    if not callable(model):
        raise ValueError("Model function must be callable")
    name = model.__name__
    if name in MODEL_REGISTRY:
        raise ValueError(f"Cannot register duplicate model ({name})")
    MODEL_REGISTRY[name] = model
    return model


def register_hparams(model):
    def register_hparams_fn(fn):
        if not callable(fn):
            raise ValueError("HParams function must be callable")
        name, model_name = fn.__name__, model.__name__
        if model_name not in HPARAMS_REGISTRY:
            HPARAMS_REGISTRY[model_name] = {}
        model_hparams = HPARAMS_REGISTRY[model_name]
        if name in model_hparams:
            raise ValueError(
                f"Cannot register duplicate hyperparameters ({name}) for model ({model_name})"
            )
        model_hparams[name] = fn
        return fn

    return register_hparams_fn


def get_dataset(
    dataset_name,
    preprocess_name,
    use_val_split=False,
    cache_dir=None,
    version=None,
    data_dir=None,
):
    if dataset_name not in DATA_REGISTRY:
        raise DatasetNotFoundError(dataset_name)

    preprocess_registry = DATA_REGISTRY[dataset_name]
    if preprocess_name not in preprocess_registry:
        raise PreprocessNotFoundError(dataset_name, preprocess_name)
    return Dataset(
        dataset_name,
        preprocess_registry[preprocess_name],
        use_val_split=use_val_split,
        cache_dir=cache_dir,
        data_dir=data_dir,
        version=version,
    )


def get_model_function(name):
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    raise ModelNotFoundError(name)


def get_hparams(model_name, name):
    if model_name in HPARAMS_REGISTRY:
        hparams = HPARAMS_REGISTRY[model_name]
        if name in hparams:
            return hparams[name]()
    raise HParamsNotFoundError(model_name, name)
