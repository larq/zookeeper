import abc
from typing import Dict, Optional, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds

from zookeeper.core import utils
from zookeeper.core.field import Field  # type: ignore


class Dataset(abc.ABC):
    """An abstract base class to encapsulate a dataset.

    Concrete sub-classes must implement the `train` method, and optionally the
    `validation` method.
    """

    @abc.abstractmethod
    def train(self, decoders=None) -> Tuple[tf.data.Dataset, int]:
        """Return a tuple of the training dataset and the number of training examples in
        the dataset."""

        raise NotImplementedError

    def validation(self, decoders=None) -> Tuple[tf.data.Dataset, int]:
        """Return a tuple of the validation dataset and the number of validation
        examples in the dataset.

        By default, raises an error that no validation data is provided.
        """

        raise ValueError(
            f"Dataset '{self.__class__.__name__}' is not configured with validation "
            "data."
        )

    def test(self, decoders=None) -> Tuple[tf.data.Dataset, int]:
        """Return a tuple of the test dataset and the number of test examples in the
        dataset.

        By default, raises an error that no test data is provided.
        """

        raise ValueError(
            f"Dataset '{self.__class__.__name__}' is not configured with test data."
        )


def base_splits(split):
    """Splits can be merged, e.g. `tfds.Split.TRAIN + tfds.Split.Validation` or
    `"train+validation"`.

    For such composite splits, find and return a list of 'base splits'.
    """

    if "+" in split:
        return split.split("+")
    try:
        # We still support older TFDS versions with the old split API, but it
        # was removed in later versions, so we have to wrap this test in a
        # try-except block.
        if isinstance(split, tfds.core.splits._SplitMerged):
            return base_splits(split._split1) + base_splits(split._split2)
    except Exception:
        pass
    return [split]


class TFDSDataset(Dataset):
    """A wrapper around a TensorFlowDatasets dataset."""

    # The TensorFlowDatasets name, which may specify a builder config and/or
    # version, e.g. "imagenet2012:4.0.0"
    name: str = Field()

    # The directory that the dataset is stored in.
    data_dir: Optional[str] = Field(None)

    # Whether or not to download the dataset (if it's not already downloaded).
    download: bool = Field(False)

    # Train, validation, and test splits. Neither a validation nor a test split
    # is required.
    train_split: str = Field()
    validation_split: Optional[str] = Field(None)
    test_split: Optional[str] = Field(None)

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
            if "labels" in features and hasattr(features["labels"], "feature"):
                return features["labels"].feature.num_classes
            if "objects" in features and "label" in features["objects"]:
                return features["objects"]["label"].num_classes
        except Exception:
            pass
        raise ValueError("Unable to determine the number of classes automatically.")

    def num_examples(self, split) -> int:
        """Compute the number of examples in a given split."""

        return sum(self.splits[s].num_examples for s in base_splits(split))

    def load(self, split, decoders, shuffle) -> tf.data.Dataset:
        """Return a `tf.data.Dataset` object representing the requested split."""

        try:
            return tfds.load(
                name=self.name,
                split=split,
                data_dir=self.data_dir,
                download=self.download,
                decoders=decoders,
                as_dataset_kwargs={"shuffle_files": shuffle},
            )
        except AssertionError as e:
            if not self.download:
                utils.warn(
                    f"Field 'download' of component {self.__class__.__name__} is False. "
                    "If the TFDS dataset is not downloaded, set 'download' to True to "
                    "call `download_and_prepare()` automatically."
                )
            raise e from None

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

    def test(self, decoders=None) -> Tuple[tf.data.Dataset, int]:
        if self.test_split is None:
            raise ValueError(
                f"Dataset {self.__class__.__name__} is not configured with a "
                "test split."
            )
        return (
            self.load(self.test_split, decoders=decoders, shuffle=False),
            self.num_examples(self.test_split),
        )


class MultiTFDSDataset(Dataset):
    """A wrapper around multiple TensorFlowDatasets datasets.

    This allows a model to be trained on data that is combined from multiple datasets.
    """

    # The directory that the dataset is stored in.
    data_dir: Optional[str] = Field(None)

    # Whether or not to download the dataset (if it's not already downloaded).
    download: bool = Field(False)

    # A non-empty mapping from dataset names as keys to splits as values. The
    # training data will be the concatenation of the datasets loaded from each
    # (key, value) pair.
    train_split: Dict[str, str] = Field()

    # As above, a mapping from dataset names as keys to splits as values. May be
    # empty, indicating no validation data.
    validation_split: Dict[str, str] = Field(lambda: {})

    # As above, a mapping from dataset names as keys to splits as values. May be
    # empty, indicating no test data.
    test_split: Dict[str, str] = Field(lambda: {})

    def num_examples(self, splits) -> int:
        """Compute the total number of examples in the splits specified by the
        dictionary `splits`."""

        return sum(
            tfds.builder(name, data_dir=self.data_dir).info.splits[s].num_examples
            for name, split in splits.items()
            for s in base_splits(split)
        )

    def load(self, splits, decoders, shuffle) -> tf.data.Dataset:
        result = None
        for name, split in splits.items():
            try:
                dataset = tfds.load(
                    name=name,
                    split=split,
                    data_dir=self.data_dir,
                    download=self.download,
                    decoders=decoders,
                    as_dataset_kwargs={"shuffle_files": shuffle},
                )
            except AssertionError as e:
                if not self.download:
                    utils.warn(
                        f"Field 'download' of component {self.__class__.__name__} is "
                        "False. If the TFDS dataset is not downloaded, set 'download' "
                        "to True to call `download_and_prepare()` automatically."
                    )
                raise e from None
            result = result.concatenate(dataset) if result is not None else dataset
        return result

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

    def test(self, decoders=None) -> Tuple[tf.data.Dataset, int]:
        if self.test_split is None:
            raise ValueError(
                f"Dataset {self.__class__.__name__} is not configured with a "
                "test split."
            )
        return (
            self.load(self.test_split, decoders=decoders, shuffle=False),
            self.num_examples(self.test_split),
        )
