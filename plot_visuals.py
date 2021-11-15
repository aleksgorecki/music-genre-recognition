from matplotlib import pyplot as plt
import numpy as np
from numpy import typing
import typing
import librosa.display


def spec_to_img(spec_data: typing.Any) -> np.typing.ArrayLike:
    fig, ax = plt.subplots()
    librosa.display.specshow(spec_data, ax=ax)
    ax.axis('off')
    ax.imshow(spec_data)
    return fig


def spec_to_plot(spec_data: typing.Any) -> np.typing.ArrayLike:
    pass


def spec_plot(spec_data: typing.Any) -> np.typing.ArrayLike:
    pass


def wave_plot():
    pass


def wave_pretty():
    pass
