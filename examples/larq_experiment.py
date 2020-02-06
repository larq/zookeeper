"""An example of how to use Zookeeper to run a Larq BinaryNet experiment on MNIST."""

from functools import partial
from typing import Sequence, Tuple, Union

import larq as lq
import tensorflow as tf

from zookeeper import ComponentField, Field, cli, component, factory, task
from zookeeper.tf import Dataset, Experiment, Preprocessing, TFDSDataset


@component
class Mnist(TFDSDataset):
    name = Field("mnist")
    train_split = Field("train")
    validation_split = Field("test")


@component
class PadCropAndFlip(Preprocessing):
    pad_size: int = Field()

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


@factory
class BinaryNet:
    dataset: Dataset = ComponentField()
    preprocessing: Preprocessing = ComponentField()

    filters: int = Field(128)
    dense_units: int = Field(1024)
    kernel_size: Union[int, Tuple[int, int]] = Field((3, 3))

    input_shape: Tuple[int, int, int] = Field()

    def build(self) -> tf.keras.models.Model:
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
                    input_shape=self.input_shape,
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
    dataset = ComponentField(Mnist)
    input_shape: Tuple[int, int, int] = Field((28, 28, 1))
    preprocessing = ComponentField(PadCropAndFlip, pad_size=32)
    model: tf.keras.models.Model = ComponentField(BinaryNet)

    epochs = Field(100)
    batch_size = Field(128)
    learning_rate: float = Field(5e-3)

    loss = Field("sparse_categorical_crossentropy")
    metrics: Sequence[str] = Field(lambda: ["accuracy"])

    @Field
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

        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=self.metrics
        )

        lq.models.summary(self.model)

        self.model.fit(
            train_data,
            epochs=self.epochs,
            steps_per_epoch=num_train_examples // self.batch_size,
            validation_data=validation_data,
            validation_steps=num_validation_examples // self.batch_size,
        )


if __name__ == "__main__":
    cli()
