import os
import pathlib
import tensorflow as tf
import json
import typing


class DefaultModelBase:
    mel_bands = 256
    mel_spec_img_shape = (369, 496, 4)

    _num_classes: int
    _labels: typing.List[str]
    _model: tf.keras.Model

    @staticmethod
    def get_num_classes():
        return len(DefaultModelBase._labels)


class GTZANDefaultModel(DefaultModelBase):
    _labels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    _fma_format_labels = ["Blues", "Classical", "Country", "Disco", "Hip-Hop", "Jazz", "Metal", "Pop", "Reggae", "Rock"]
    model: tf.keras.Model

    def __init__(self):
        self.model = GTZANDefaultModel.model_blueprint()

    @staticmethod
    def model_blueprint(input_size=DefaultModelBase.mel_spec_img_shape, classes=len(_labels)) -> tf.keras.Model:
        model = tf.keras.models.Sequential(layers=[
            tf.keras.Input(input_size),

            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(rate=0.3),
            tf.keras.layers.Dense(classes, activation='softmax')
        ])
        return model

    @staticmethod
    def _get_labels_in_fma_format() -> typing.List[str]:
        return GTZANDefaultModel._fma_format_labels.copy()

    @staticmethod
    def get_labels() -> typing.List[str]:
        return GTZANDefaultModel._labels.copy()

    # def load_weights(self):
        self.model.load_weights()
    pass

class FMADefaultModel(DefaultModelBase):
    _labels = ["Electronic", "Experimental", "Folk", "Hip-Hop", "Instrumental", "International", "Pop", "Rock"]

    @staticmethod
    def get_labels() -> typing.List[str]:
        return FMADefaultModel._labels.copy()
    pass

    def model_blueprint(input_size=(369, 496, 4), classes=8) -> tf.keras.Model:
        model = tf.keras.models.Sequential(layers=[
            tf.keras.Input(input_size),

            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(rate=0.3),
            tf.keras.layers.Dense(classes, activation='softmax')
        ])
        return model


class MixedDefaultModel(DefaultModelBase):
    pass

    def define_model(input_size=(369, 496, 4), classes=15) -> tf.keras.Model:
        model = tf.keras.models.Sequential(layers=[
            tf.keras.Input(input_size),

            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(rate=0.3),
            tf.keras.layers.Dense(classes, activation='softmax')
        ])
        return model

CLASSES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]





if __name__ == "__main__":


    checkpoint_pth = pathlib.Path("/home/aleksy/checkpoints50-512-mixed/")

    os.makedirs(checkpoint_pth, exist_ok=True)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_pth.joinpath("{epoch:02d}-{val_accuracy:.2f}"),
        monitor="val_accuracy",
        save_weights_only=False,
        save_freq="epoch",
    )

    main_dataset = "/home/aleksy/gtzan_versions/5sec/gtzan_spec_5_sec_50_512_mixed"
    main_dataset_path = pathlib.Path(main_dataset)

    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    val_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    train_flow = train_generator.flow_from_directory(directory=main_dataset_path.joinpath("train"),
                                                     target_size=(369, 496),
                                                     color_mode="rgba",
                                                     class_mode='categorical',
                                                     batch_size=8,
                                                     shuffle=True
                                                     )
    test_flow = test_generator.flow_from_directory(directory=main_dataset_path.joinpath("test"),
                                                   target_size=(369, 496),
                                                   color_mode="rgba",
                                                   class_mode='categorical',
                                                   shuffle=True,
                                                   batch_size=8)
    val_flow = val_generator.flow_from_directory(directory=main_dataset_path.joinpath("val"),
                                                 target_size=(369, 496),
                                                 color_mode="rgba",
                                                 class_mode='categorical',
                                                 shuffle=True,
                                                 batch_size=8)

    model = GTZANDefaultModel.model_blueprint()
    with open(checkpoint_pth.joinpath("model.json"), "w+") as f:
        f.write(model.to_json())

    #callback_checkpoint = tf.keras.callbacks.ModelCheckpoint()

    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    #opt = tf.keras.optimizers.SGD(learning_rate=0.0001)
    #opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    #opt = tf.keras.optimizers.SGD(learning_rate=0.001)

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(train_flow, epochs=50, validation_data=test_flow, callbacks=[checkpoint_callback])

    try:
        with open(checkpoint_pth.joinpath("history.json"), "w+") as f:
            json.dump(obj=dict(hist.history), fp=f)
    except TypeError:
        pass

    model.save_weights("/home/aleksy/weights", save_format='h5', overwrite=True)
    model.load_weights("/home/aleksy/weights")
    histeval = model.evaluate(test_flow, return_dict=True)
    print(histeval)
    print(histeval["accuracy"])
    print(histeval["loss"])