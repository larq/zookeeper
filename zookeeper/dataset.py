from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds

from zookeeper.component import Component


class Dataset(Component, ABC):
    """
    An abstract base class to encapsulate a dataset. Concrete sub-classes must
    implement the `train` method, and optionally the `validation` method.
    """

    @abstractmethod
    def train(self, decoders=None) -> Tuple[tf.data.Dataset, int]:
        """
        Return a tuple of the training dataset and the number of training
        examples in the dataset.
        """

        raise NotImplementedError

    def validation(self, decoders=None) -> Tuple[tf.data.Dataset, int]:
        """
        Return a tuple of the validation dataset and the number of validation
        examples in the dataset. By default, raises an error that no validation
        data is provided.
        """

        raise ValueError(
            f"Dataset '{self.__component_name__}' is not configured with validation "
            "data."
        )


def base_splits(split):
    """
    Splits can be merged, e.g. `tfds.Split.TRAIN + tfds.Split.Validation` or
    `"train+validation"`. For such composite splits, find and return a list of
    'base splits'.
    """

    if "+" in split:
        return split.split("+")
    elif isinstance(split, tfds.core.splits._SplitMerged):
        return base_splits(split._split1) + base_splits(split._split2)  # type: ignore
    return [split]


class TFDSDataset(Dataset):
    """
    A wrapper around a TensorFlowDatasets dataset.
    """

    # The TensorFlowDatasets name, which may specify a builder config and/or
    # version, e.g. "imagenet2012:4.0.0"
    name: str

    # The directory that the dataset is stored in.
    data_dir: Optional[str] = None

    # Train and validation splits. A validation split is not required.
    train_split: str
    validation_split: Optional[str] = None

    def validate_configuration(self):
        super().validate_configuration()

        # Check that the name corresponds to a valid TensorFlow dataset.
        builder_names = tfds.list_builders()
        if self.name.split(":")[0].split("/")[0] not in builder_names:
            raise ValueError(
                f"'{self.__component_name__}.name' has invalid value '{self.name}'. "
                "Valid dataset names:\n    " + ",\n    ".join(builder_names)
            )

        # Check that the `train_split` is valid.
        if self.train_split is None or any(
            s not in self.splits for s in base_splits(self.train_split)
        ):
            raise ValueError(
                f"'{self.__component_name__}.train_split' has invalid value "
                f"'{self.train_split}'. Valid values:\n    "
                + ",\n    ".join(self.splits.keys())
            )

        # Check that `validation_split` is valid (`None` is allowed).
        if self.validation_split is not None and any(
            s not in self.splits for s in base_splits(self.train_split)
        ):
            raise ValueError(
                f"'{self.__component_name__}.train_split' has invalid value "
                f"'{self.train_split}'. Valid values:\n    "
                + ",\n    ".join([None] + self.splits.keys())
            )

    @property
    def info(self):
        if not hasattr(self, "_info"):
            self._info = tfds.builder(self.name, data_dir=self.data_dir).info
        return self._info

    @property
    def splits(self):
        return self.info.splits

    @property
    def num_classes(self) -> int:
        try:
            features = self.info.features
            if "label" in features:
                return features["label"].num_classes
            if "labels" in features:
                return features["labels"].feature.num_classes
            if "objects" in features and "label" in features["objects"]:
                return features["objects"]["label"].num_classes
        except Exception:
            pass
        raise ValueError("Unable to determine the number of classes.")

    def num_examples(self, split) -> int:
        """Compute the number of examples in a given split."""

        return sum(self.splits[s].num_examples for s in base_splits(split))

    def load(self, split, decoders, shuffle) -> tf.data.Dataset:
        """Return a `tf.data.Dataset` object representing the requested split."""

        return tfds.load(
            name=self.name,
            split=split,
            data_dir=self.data_dir,
            decoders=decoders,
            as_dataset_kwargs={"shuffle_files": shuffle},
        )

    def train(self, decoders=None) -> Tuple[tf.data.Dataset, int]:
        return (
            self.load(self.train_split, decoders=decoders, shuffle=True),
            self.num_examples(self.train_split),
        )

    def validation(self, decoders=None) -> Tuple[tf.data.Dataset, int]:
        if self.validation_split is None:
            raise ValueError(
                f"Dataset {self.__class__.__name__} is not configured with a "
                "validation split."
            )
        return (
            self.load(self.validation_split, decoders=decoders, shuffle=False),
            self.num_examples(self.validation_split),
        )


class MultiTFDSDataset(Dataset):
    """
    A wrapper around multiple TensorFlowDatasets datasets. This allows a model
    to be trained on data that is combined from multiple datasets.
    """

    # A non-empty mapping from dataset names as keys to splits as values. The
    # training data will be the concatenation of the datasets loaded from each
    # (key, value) pair.
    train_splits: Dict[str, str]

    # As above, a mapping from dataset names as keys to splits as values. May be
    # empty, indicating no validation data.
    validation_splits: Dict[str, str] = {}

    # The directory that the dataset is stored in.
    data_dir: Optional[str] = None

    def num_examples(self, splits) -> int:
        """
        Compute the total number of examples in the splits specified by the
        dictionary `splits`.
        """

        return sum(
            tfds.builder(name, data_dir=self.data_dir).info.splits[s].num_examples
            for name, split in splits
            for s in base_splits(split)
        )

    def load(self, splits, decoders, shuffle) -> tf.data.Dataset:
        result = None
        for name, split in splits.items():
            dataset = tfds.load(
                name=name,
                split=split,
                data_dir=self.data_dir,
                decoders=decoders,
                as_dataset_kwargs={"shuffle_files": shuffle},
            )
            result = result.concatenate(dataset) if result is not None else dataset
        return result

    def train(self, decoders=None):
        return (
            self.load(self.train_splits, decoders=decoders, shuffle=True),
            self.num_examples(self.train_splits),
        )

    def validation(self, decoders=None):
        return (
            self.load(self.validation_splits, decoders=decoders, shuffle=False),
            self.num_examples(self.validation_splits),
        )
