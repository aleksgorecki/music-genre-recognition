import os
import pathlib

import numpy as np
import numpy.typing
import tensorflow as tf
import audio_processing
import visual
import keras_preprocessing.image
import shutil
from matplotlib import pyplot as plt
import matplotlib
from PIL import Image
import io

from visual import draw_class_distribution, prepare_class_for_class_distribution
from gtzan_utils import GTZANLabels
from fma_utils import FMALabels


GTZAN_classes = GTZANLabels
MELS = 256 #128
FRAGMENT_DURATION = 5.0


def preprocess_img_array(img_arr: np.typing.ArrayLike) -> np.typing.ArrayLike:
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape((1, img_arr.shape[0], img_arr.shape[1], img_arr.shape[2]))
    return img_arr


def load_and_preprocess_image(img_path: str) -> numpy.typing.ArrayLike:
    img = keras_preprocessing.image.load_img(img_path, color_mode="rgba", target_size=(369, 496))
    img = preprocess_img_array(img)
    return img


def fig_to_array(fig: plt.Figure, dpi: int = 100) -> numpy.typing.ArrayLike:
    with io.BytesIO() as buff:
        plt.savefig(buff, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
        buff.seek(0)
        img = Image.open(buff)
        img_arr = np.array(img)
    return img_arr


def prepare_model(model_obj: tf.keras.Model, weights_path: str) -> None:
    pass


def run_single_prediction(compiled_model: tf.keras.Model,
                          audio_data: audio_processing.LibrosaMonoTimeseries,
                          fragment_duration: float = 5.0,
                          offset_percent: float = 0.50
                          ):
    pass

if __name__ == "__main__":
    label_idx_dict = dict()
    for idx, label in enumerate(sorted(GTZAN_classes)):
        label_idx_dict.update({idx: label})


    # with open("/home/aleksy/Desktop/model1.json", "r") as json_f:
    #     json_str = json_f.read()
    #
    # model: tf.keras.Model = tf.keras.models.model_from_json(json_str)
    # model.load_weights("/home/aleksy/Desktop/weights")

    model = tf.keras.models.load_model("/home/aleksy/checkpoints50-256/90-0.88")
    model.summary()


    audio_path = "/home/aleksy/Full_Songs/Silent Hill 4 - Waiting For you.mp3"


    audio = audio_processing.load_to_mono(audio_path)

    plots_interval_sec = 120
    audio_in_sec = len(audio.timeseries)/audio.sr
    plot_points = np.arange(plots_interval_sec, len(audio.timeseries)/audio.sr - plots_interval_sec, plots_interval_sec)
    mels = list()
    for pp in plot_points:
        audio_sample = audio_processing.get_fragment_of_timeseries(audio, offset_sec=pp, fragment_duration_sec=FRAGMENT_DURATION)
        sample_mel = audio_processing.mel_from_timeseries(audio_sample, mel_bands=MELS)
        mels.append(sample_mel)

    sample_files = "/home/aleksy/tmp_mels"

    if os.path.exists(sample_files):
        shutil.rmtree(sample_files)
    os.makedirs(sample_files, exist_ok=True)
    for idx, mel in enumerate(mels):
        visual.save_mel_only(sample_files + f"/{idx}", mel_data=mel)

    images = []
    for file in pathlib.Path(sample_files).iterdir():
        image = keras_preprocessing.image.load_img(file, color_mode="rgba", target_size=(369, 496))
        image = keras_preprocessing.image.img_to_array(image)/255
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        images.append(image)


    outputs = []
    for image in images:
        output = model.predict(x=image, batch_size=1)
        outputs.append(output)

    sum_output = np.sum(outputs, axis=0)
    mean_output = sum_output/len(outputs)


    output = mean_output

    probability_dict = dict()
    for idx, prob in enumerate(list(output[0])):
        probability_dict.update({label_idx_dict[idx]: prob})

    for item in probability_dict.items():
        print(item)

    print("=========")

    for item in probability_dict.items():
        if item[1] > 0.09:
            proc = item[1] * 100
            print(item[0], proc)

    print("==========")
    print(pathlib.Path(audio_path).stem)
    print(max(probability_dict.items(), key=lambda item: item[1])[0])

    outputs_scaled = [int(x * 100) for x in output[0]]
    labels = [x.capitalize() for x in sorted(GTZAN_classes)]

    d = {
        "class": labels,
        "prob": outputs_scaled
    }

    fig, ax = plt.subplots()
    prepare_class_for_class_distribution(fig, ax)
    draw_class_distribution(ax, labels, outputs_scaled)
    plt.show()

