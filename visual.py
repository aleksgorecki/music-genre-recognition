import matplotlib.pyplot
from matplotlib import pyplot as plt
import typing
import librosa.display
import json


def mel_only_on_ax(mel: typing.Any, ax: plt.Axes) -> None:
    ax.clear()
    ax.axis("off")
    librosa.display.specshow(mel, ax=ax)


def spec_only_on_ax(spec: typing.Any, ax: plt.Axes, log_freq: bool) -> None:
    ax.clear()
    ax.axis("off")
    if log_freq:
        librosa.display.specshow(spec, ax=ax, y_axis="log")
    else:
        librosa.display.specshow(spec, ax=ax, y_axis="hz")


def save_mel_only(filepath: str, mel_data: typing.Any, dpi: int = 100) -> None:
    fig, ax = plt.subplots()
    plt.close(fig)
    mel_only_on_ax(mel_data, ax)
    fig.savefig(fname=filepath, bbox_inches="tight", pad_inches=0.0, dpi=dpi)


# def output_class_distribution_on_ax(ax: plt.Axes) -> None:
#     ax.clear()
#     ax.axis("off")
#     pass


def prepare_class_for_class_distribution(fig: plt.Figure, ax: plt.Axes):
    fig.set_size_inches(7.5, 2.5, forward=True)
    # ax.set_ylabel("Prediction confidence [%]")
    # ax.set_xlabel("Possible genres")
    ax.set_ylim([0, 100])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    ax.tick_params(length=0)
    # ax.get_yaxis().set_visible(False)
    # ax.get_xaxis().set_visible(False)
    fig.tight_layout()
    return fig, ax


def draw_class_distribution(ax: plt.Axes, labels: typing.List[str], outputs_scaled: typing.List[int]):
    if len(labels) == 10:
        colors = ["aquamarine", "turquoise", "mediumspringgreen", "lightseagreen", "paleturquoise"]
    elif len(labels) == 8:
        colors = ["slateblue", "mediumslateblue", "rebbecapurple", "mediumpurple"]
    else:
        colors = ["darkorange", "navajowhite", "gold", "moccasin"]
    ax.bar(labels, outputs_scaled, color=colors)


def confusion_matrix():
    pass


def training_history(history_json_path: str):
    with open(history_json_path, "r") as f:
        history = json.load(f)
    train_acc = history["accuracy"]
    train_loss = history["loss"]
    val_acc = history["val_accuracy"]
    val_loss = history["val_loss"]

    fig, ax = plt.plot()
    ax.plot(train_loss)
    ax.plot(val_loss)
    ax.set_title("Funkcja strat")
    ax.set_ylabel("Wartość funkcji strat")
    ax.set_xlabel("Epoka")
    ax.legend(["Zbiór uczący", "Zbiór testowy"])
    fig.show()

    fig, ax = plt.plot()
    ax.plot(train_acc)
    ax.plot(val_acc)
    ax.set_title("Dokładność modelu")
    ax.set_ylabel("Dokładność")
    ax.set_xlabel("Epoka")
    ax.legend(["Zbiór uczący", "Zbiór testowy"])
    fig.show()

if __name__ == "__main__":
    pass