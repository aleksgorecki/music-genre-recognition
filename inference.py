import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import numpy.typing
import tensorflow as tf
tf.get_logger().setLevel('INFO')
import audio_processing
import visual
import keras_preprocessing.image
from matplotlib import pyplot as plt
import argparse
from PIL import Image
import io
import warnings

from gtzan_utils import GTZANLabels
from fma_utils import FMALabels


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
                          file_path: str,
                          fragment_duration: float = 5.0,
                          offset_sec: float = 0.0
                          ):
    audio_data = audio_processing.load_to_mono(file_path, offset_sec=offset_sec, duration=fragment_duration)
    sample_mel = audio_processing.mel_from_timeseries(audio_data, MELS)
    plt.switch_backend("Agg")
    fig, ax = plt.subplots()
    visual.mel_only_on_ax(sample_mel, ax)
    mel_img = fig_to_array(fig)
    mel_img = preprocess_img_array(mel_img)

    output = compiled_model.predict(x=mel_img, batch_size=1)
    return output[0]


def run_multiple_predictions(compiled_model: tf.keras.Model,
                             file_path: str,
                             fragment_duration: float = 5.0,
                             window_interval: float = 30
                             ):
    audio_data = audio_processing.load_to_mono(file_path)
    audio_duration_sec = int(len(audio_data.timeseries) / audio_data.sr)
    plot_points = np.arange(0, audio_duration_sec - fragment_duration, window_interval)

    mels = list()
    for point in plot_points:
        window = audio_processing.get_fragment_of_timeseries(audio_data, point, fragment_duration)
        mel = audio_processing.mel_from_timeseries(window, mel_bands=MELS)
        mels.append(mel)

    plt.switch_backend("Agg")
    fig, ax = plt.subplots()
    imgs = list()
    for mel in mels:
        visual.mel_only_on_ax(mel, ax)
        mel_img = fig_to_array(fig)
        mel_img = preprocess_img_array(mel_img)
        imgs.append(mel_img)

    outputs = list()
    for img in imgs:
        output = compiled_model.predict(x=img, batch_size=1)
        outputs.append(output[0])

    return outputs


def process_single_output(output: np.typing.ArrayLike, labels: list) -> dict:
    class_dict = dict()
    for idx, label in enumerate(labels):
        class_dict.update({label: output[idx]})

    return class_dict
    pass


def mean_average_output(outputs: list, labels: list) -> dict:
    mean_output = np.sum(outputs, axis=0) / len(outputs)
    class_dict = dict()
    for idx, label in enumerate(labels):
        class_dict.update({label: mean_output[idx]})

    return class_dict


def weighted_average_output(outputs: list, labels: list) -> dict:
    mean_output = np.sum(outputs, axis=0) / len(outputs)
    class_dict = dict()
    for idx, label in enumerate(labels):
        class_dict.update({label: mean_output[idx]})

    return class_dict


def sort_output_dict(output_dict: dict) -> dict:
    return {k: v for k, v in sorted(output_dict.items(), key=lambda item: item[1], reverse=True)}


def print_output_dict(output_dict: dict) -> None:
    for item in output_dict.items():
        print(item[0].capitalize(), ": ", int(item[1] * 100), "%")


def most_probable_class_from_class_dict(output_dict: dict):
    return max(output_dict, key=output_dict.get)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, required=True)
    parser.add_argument("--model", type=str, default="gtzan", choices=["fma", "gtzan"])
    parser.add_argument("--window", "-w", type=str, choices=["single", "multiple"], required=False, default="multiple")
    parser.add_argument("--window-len", type=float, required=False, default=5.0)
    parser.add_argument("--window-interval", type=float, default=30.0)
    parser.add_argument("--offset", type=float, default=0)
    parser.add_argument("--print-all", default=False, action="store_true")
    args = parser.parse_args()

    if args.model == "gtzan":
        model = tf.keras.models.load_model("/home/aleksy/checkpoints50-256/90-0.88")
        labels = GTZANLabels
    else:
        model = tf.keras.models.load_model("/home/aleksy/checkpoints50-256-mixed-fma-100-0.0001/49-0.79")
        labels = FMALabels
    model.compile()

    audio_path = args.file

    warnings.filterwarnings(category=UserWarning, action="ignore")

    if args.window == "multiple":
        outputs = run_multiple_predictions(model, audio_path, fragment_duration=args.window_len,
                                           window_interval=args.window_interval)
        classes = mean_average_output(outputs, labels)
    else:
        outputs = run_single_prediction(model, audio_path, fragment_duration=args.window_len, offset_sec=args.offset)
        classes = process_single_output(outputs, labels)

    classes = sort_output_dict(classes)
    print("Predicted genre: ", most_probable_class_from_class_dict(classes).capitalize())
    if args.print_all:
        print_output_dict(classes)

