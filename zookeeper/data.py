import functools
import glob
import os
import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset:
    def __init__(
        self,
        dataset_name,
        preprocess_cls,
        use_val_split=False,
        cache_dir=None,
        data_dir=None,
        version=None,
    ):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.version = version

        self.info = tfds.builder(self.dataset_name_str, data_dir=data_dir).info
        splits = self.info.splits
        self.preprocessing = preprocess_cls(features=self.info.features)

        if tfds.Split.TRAIN not in splits:
            raise ValueError("To train we require a train split in the dataset.")
        if tfds.Split.TEST not in splits and tfds.Split.VALIDATION not in splits:
            raise ValueError("We require a test or validation split in the dataset.")

        self.train_split = tfds.Split.TRAIN
        self.train_examples = splits[self.train_split].num_examples
        if tfds.Split.TEST in splits:
            self.test_split = tfds.Split.TEST
            self.test_examples = splits[self.test_split].num_examples
        if tfds.Split.VALIDATION in splits:
            self.validation_split = tfds.Split.VALIDATION
            self.validation_examples = splits[self.validation_split].num_examples
        else:
            if use_val_split == True:
                self.train_split = tfds.Split.TRAIN.subsplit(tfds.percent[:90])
                all_train_examples = splits[tfds.Split.TRAIN].num_examples
                self.train_examples = int(all_train_examples * 0.9)
                self.validation_split = tfds.Split.TRAIN.subsplit(tfds.percent[-10:])
                self.validation_examples = all_train_examples - self.train_examples
            else:
                self.validation_split = self.test_split
                self.validation_examples = self.test_examples

    @property
    def dataset_name_str(self):
        return (
            f"{self.dataset_name}:{self.version}" if self.version else self.dataset_name
        )

    def load_split(self, split, shuffle=True):
        return tfds.load(
            name=self.dataset_name_str,
            split=split,
            data_dir=self.data_dir,
            decoders=self.preprocessing.decoders,
            as_dataset_kwargs={"shuffle_files": shuffle},
        )

    def get_cache_path(self, split_name):
        if self.cache_dir is None:
            return None
        if self.cache_dir == "memory":
            return ""
        # We need to check for trailing lockfiles here:
        # https://github.com/tensorflow/tensorflow/issues/28798
        for i in range(3):
            cache_dir = os.path.join(
                self.cache_dir,
                self.dataset_name_str if i == 0 else f"{self.dataset_name_str}_{i}",
            )
            if not glob.glob(f"{cache_dir}/*.lockfile"):
                os.makedirs(cache_dir, exist_ok=True)
                return os.path.join(cache_dir, split_name)
        raise RuntimeError(
            f"Out of retries! Cache lockfile already exists ({cache_dir}). "
            "If you are sure no other running TF computations are using this cache prefix, "
            "delete the lockfile and restart training."
        )

    def maybe_cache(self, dataset, split_name):
        if self.cache_dir is None:
            return dataset
        return dataset.cache(self.get_cache_path(split_name))

    def train_data(self, batch_size):
        return (
            self.maybe_cache(self.load_split(self.train_split), "train")
            .shuffle(10 * batch_size)
            .repeat()
            .map(
                functools.partial(self.preprocessing, training=True),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .batch(batch_size)
            .prefetch(1)
        )

    def validation_data(self, batch_size):
        dataset = self.maybe_cache(
            self.load_split(self.validation_split, shuffle=False), "eval"
        )
        return self._get_eval_data(dataset, batch_size)

    def test_data(self, batch_size):
        dataset = self.maybe_cache(
            self.load_split(self.test_split, shuffle=False), "test"
        )
        return self._get_eval_data(dataset, batch_size)

    def _get_eval_data(self, dataset, batch_size):
        return (
            dataset.repeat()
            .map(self.preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(batch_size)
            .prefetch(1)
        )
