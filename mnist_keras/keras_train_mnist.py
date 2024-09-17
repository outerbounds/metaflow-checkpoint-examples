import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint


def create_model():
    model = Sequential(
        [
            layers.Flatten(input_shape=(28, 28, 1)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(10),
        ]
    )
    return model


def compile_model(model):
    model.compile(
        optimizer=Adam(1e-3),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=[SparseCategoricalAccuracy()],
    )
    return model


def load_dataset():
    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = keras.datasets.mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_images = train_images[..., tf.newaxis]
    test_images = test_images[..., tf.newaxis]
    return (train_images, train_labels), (test_images, test_labels)


def train(checkpoint_path=None, num_epochs=[], callbacks=[]):
    (train_images, train_labels), _ = load_dataset()

    model = create_model()
    model = compile_model(model)

    if checkpoint_path:
        model.load_weights(checkpoint_path)
        print(f"Model loaded from {checkpoint_path}")

    model.fit(
        train_images,
        train_labels,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=callbacks,
    )
    return model


def test(checkpoint_path):
    _, (test_images, test_labels) = load_dataset()

    model = create_model()
    model.load_weights(checkpoint_path)
    print(f"Model loaded from {checkpoint_path}")

    model = compile_model(model)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print("\nTest accuracy:", test_acc)


if __name__ == "__main__":
    # To train a new model or continue training from a checkpoint:
    # If continuing from a checkpoint, specify its path as the argument.
    train_checkpoint_path = None  # Example: 'path/to/checkpoint.h5'
    train(checkpoint_path=train_checkpoint_path)

    # To test the model, specify the path of the checkpoint you want to load.
    test_checkpoint_path = "path/to/checkpoint.h5"  # Adjust the path accordingly.
    test(checkpoint_path=test_checkpoint_path)
