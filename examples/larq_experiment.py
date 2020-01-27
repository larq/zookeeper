"""An example of how to use Zookeeper to run a Larq BinaryNet experiment on MNIST."""

from functools import partial
from typing import Tuple, Union

import larq as lq
import tensorflow as tf

from zookeeper import cli, component, task
from zookeeper.tf import Dataset, Experiment, Model, Preprocessing, TFDSDataset


@component
class Mnist(TFDSDataset):
    name = "mnist"
    train_split = "train"
    validation_split = "test"


@component
class PadCropAndFlip(Preprocessing):
    pad_size: int

    def input(self, data, training):
        image = data["image"]
        if training:
            image = tf.image.resize_with_crop_or_pad(
                image, self.pad_size, self.pad_size
            )
            image = tf.image.random_crop(image, self.input_shape)
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.resize_with_crop_or_pad(image, *self.input_shape[:2])
        return tf.cast(image, tf.float32) / (255.0 / 2.0) - 1.0

    def output(self, data):
        return data["label"]


@component
class BinaryNet(Model):
    dataset: Dataset
    preprocessing: Preprocessing

    filters: int = 128
    dense_units: int = 1024
    kernel_size: Union[int, Tuple[int, int]] = (3, 3)

    def build(self, input_shape):
        kwhparams = dict(
            input_quantizer="ste_sign",
            kernel_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            use_bias=False,
        )

        return tf.keras.models.Sequential(
            [
                # Don't quantize inputs in first layer
                lq.layers.QuantConv2D(
                    self.filters,
                    self.kernel_size,
                    kernel_quantizer="ste_sign",
                    kernel_constraint="weight_clip",
                    use_bias=False,
                    input_shape=input_shape,
                ),
                tf.keras.layers.BatchNormalization(scale=False),
                lq.layers.QuantConv2D(
                    self.filters, self.kernel_size, padding="same", **kwhparams
                ),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                tf.keras.layers.BatchNormalization(scale=False),
                lq.layers.QuantConv2D(
                    2 * self.filters, self.kernel_size, padding="same", **kwhparams
                ),
                tf.keras.layers.BatchNormalization(scale=False),
                lq.layers.QuantConv2D(
                    2 * self.filters, self.kernel_size, padding="same", **kwhparams
                ),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                tf.keras.layers.BatchNormalization(scale=False),
                lq.layers.QuantConv2D(
                    4 * self.filters, self.kernel_size, padding="same", **kwhparams
                ),
                tf.keras.layers.BatchNormalization(scale=False),
                lq.layers.QuantConv2D(
                    4 * self.filters, self.kernel_size, padding="same", **kwhparams
                ),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                tf.keras.layers.BatchNormalization(scale=False),
                tf.keras.layers.Flatten(),
                lq.layers.QuantDense(self.dense_units, **kwhparams),
                tf.keras.layers.BatchNormalization(scale=False),
                lq.layers.QuantDense(self.dense_units, **kwhparams),
                tf.keras.layers.BatchNormalization(scale=False),
                lq.layers.QuantDense(self.dataset.num_classes, **kwhparams),
                tf.keras.layers.BatchNormalization(scale=False),
                tf.keras.layers.Activation("softmax"),
            ]
        )


@task
class BinaryNetMnist(Experiment):
    dataset = Mnist()
    preprocessing = PadCropAndFlip(pad_size=32, input_shape=(28, 28, 1))
    model = BinaryNet()

    epochs = 100
    batch_size = 128

    loss = "sparse_categorical_crossentropy"

    metrics = ["accuracy"]

    learning_rate = 5e-3

    @property
    def optimizer(self):
        return tf.keras.optimizers.Adam(self.learning_rate)

    def run(self):
        train_data, num_train_examples = self.dataset.train()
        train_data = (
            train_data.cache()
            .shuffle(10 * self.batch_size)
            .repeat()
            .map(partial(self.preprocessing, training=True))
            .batch(self.batch_size)
        )
        validation_data, num_validation_examples = self.dataset.validation()
        validation_data = (
            validation_data.cache()
            .repeat()
            .map(self.preprocessing)
            .batch(self.batch_size)
        )

        model = self.model.build(input_shape=self.preprocessing.input_shape)

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        lq.models.summary(model)

        model.fit(
            train_data,
            epochs=self.epochs,
            steps_per_epoch=num_train_examples // self.batch_size,
            validation_data=validation_data,
            validation_steps=num_validation_examples // self.batch_size,
        )


if __name__ == "__main__":
    cli()
