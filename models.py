from keras import Sequential, Input, Model
from keras.layers import RandomFlip, RandomTranslation, RandomZoom, Rescaling, Conv2D, BatchNormalization, \
    MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense, ReLU
from keras.layers import Resizing
from keras.applications.mobilenet_v2 import MobileNetV2

from config import IMG_SIZE


def make_augmentation():
    return Sequential([
        RandomFlip("horizontal"),
        RandomTranslation(0.1, 0.1, fill_mode="reflect"),
        RandomZoom(0.1, 0.1),
    ])


def build_cnn(augment=False):
    layers_list = [Input(shape=(32, 32, 3))]

    if augment:
        layers_list.append(make_augmentation())

    layers_list.append(
        Rescaling(1 / 255)
    )
    # 1st block
    layers_list.extend([
        Conv2D(32, 3, padding="same", use_bias=False),
        BatchNormalization(momentum=0.9),
        ReLU(),
        Conv2D(32, 3, padding="same", use_bias=False),
        BatchNormalization(momentum=0.9),
        ReLU(),
        MaxPooling2D(2),
        Dropout(0.1),
    ])

    # 2nd block
    layers_list.extend([
        Conv2D(64, 3, padding="same", use_bias=False),
        BatchNormalization(momentum=0.9),
        ReLU(),
        Conv2D(64, 3, padding="same", use_bias=False),
        BatchNormalization(momentum=0.9),
        ReLU(),
        MaxPooling2D(2),
        Dropout(0.2),
    ])

    # 3rd block
    layers_list.extend([
        Conv2D(128, 3, padding="same", use_bias=False),
        BatchNormalization(momentum=0.9),
        ReLU(),
        Conv2D(128, 3, padding="same", use_bias=False),
        BatchNormalization(momentum=0.9),
        ReLU(),
        MaxPooling2D(2),
        Dropout(0.3),
    ])

    # wrap it
    layers_list.extend([
        GlobalAveragePooling2D(),
        Dense(10, activation="softmax"),
    ])

    model = Sequential(layers_list)
    return model


def build_transfer(augment=False):
    # creating mobilenet_v2 without head
    backbone = MobileNetV2(weights="imagenet", include_top=False,
                           input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling="avg")
    backbone.trainable = False

    # preparing input and transfer learning for model
    inputs = Input(shape=(32, 32, 3))
    x = inputs

    if augment:
        x = make_augmentation()(x)

    x = Resizing(IMG_SIZE, IMG_SIZE)(x)
    x = Rescaling(1 / 127.5, offset=-1)(x)
    x = backbone(x, training=False)
    x = Dropout(0.2)(x)
    outputs = Dense(10, activation="softmax")(x)

    return Model(inputs, outputs), backbone


def enable_fine_tuning(backbone, n_last):
    backbone.trainable = True
    for layer in backbone.layers[: -n_last]:
        layer.trainable = False

    for layer in backbone.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False


if __name__ == "__main__":
    model = build_cnn()
    model.summary()

    mobilenet, backbone = build_transfer()
    mobilenet.summary()
