import tensorflow as tf
from zookeeper import cli, build_train, HParams, registry, Preprocessing


class ImageClassification(Preprocessing):
    @property
    def kwargs(self):
        return {
            "input_shape": self.features["image"].shape,
            "num_classes": self.features["label"].num_classes,
        }

    def inputs(self, data):
        return tf.cast(data["image"], tf.float32)

    def outputs(self, data):
        return tf.one_hot(data["label"], self.features["label"].num_classes)


@registry.register_preprocess("mnist")
class default(ImageClassification):
    def inputs(self, data):
        return super().inputs(data) / 255


@registry.register_model
def cnn(hp, input_shape, num_classes):
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                hp.filters[0], (3, 3), activation=hp.activation, input_shape=input_shape
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(hp.filters[1], (3, 3), activation=hp.activation),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(hp.filters[2], (3, 3), activation=hp.activation),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(hp.filters[3], activation=hp.activation),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )


@registry.register_hparams(cnn)
class basic(HParams):
    epochs = 100
    activation = "relu"
    batch_size = 32
    filters = [64, 64, 64, 64]
    learning_rate = 1e-3

    @property
    def optimizer(self):
        return tf.keras.optimizers.Adam(self.learning_rate)


@registry.register_hparams(cnn)
class small(basic):
    filters = [16, 32, 32, 32]


@cli.command()
@build_train()
def train(build_model, dataset, hparams, output_dir):
    model = build_model(hparams, **dataset.preprocessing.kwargs)
    model.compile(
        optimizer=hparams.optimizer,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy", "top_k_categorical_accuracy"],
    )

    model.fit(
        dataset.train_data(hparams.batch_size),
        epochs=hparams.epochs,
        steps_per_epoch=dataset.train_examples // hparams.batch_size,
        validation_data=dataset.validation_data(hparams.batch_size),
        validation_steps=dataset.validation_examples // hparams.batch_size,
    )


if __name__ == "__main__":
    cli()
