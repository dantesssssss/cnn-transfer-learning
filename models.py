from keras import Sequential, Input
from keras.layers import RandomFlip, RandomTranslation, RandomZoom, Rescaling, Conv2D, BatchNormalization, \
    MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense, ReLU


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





if __name__ == "__main__":
    model = build_cnn()
    model.summary()