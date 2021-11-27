from matplotlib import pyplot as plt
import numpy as np
from numpy import typing
import typing
import librosa.display


def mel_only_on_ax(mel: typing.Any, ax: plt.Axes) -> None:
    ax.clear()
    ax.axis("off")
    librosa.display.specshow(mel, ax=ax)


def spec_to_plot(spec_data: typing.Any) -> np.typing.ArrayLike:
    pass


def spec_plot(spec_data: typing.Any) -> np.typing.ArrayLike:
    pass


def wave_plot():
    pass


def wave_pretty():
    pass
