import audio_processing
import plot_visuals

import audioread.exceptions
import os
import numpy as np
import pathlib
from matplotlib import pyplot as plt


class GTZANPreparation:
    GTZANLabels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]


def split_dataset():
    pass


def convert_dataset_to_mel_spectrograms(original_gtzan_dir_path: str,
                                        target_dir_path: str,
                                        mel_bands: int = 128,
                                        sr_overwrite: int = None,
                                        splits_duration: float = None,
                                        spec_log_scale: bool = True,
                                        save_as_binary: bool = False
                                        ) -> None:
    for genre_dir in (pathlib.Path(original_gtzan_dir_path).joinpath("Data/genres_original/")).iterdir():
        print(f"Starting {genre_dir.stem}")
        for audio_file_name in genre_dir.iterdir():
            try:
                audio_data = audio_processing.load_audio_file_and_convert_to_mono(str(audio_file_name),
                                                                                  sr_overwrite)
            except (RuntimeError, audioread.exceptions.NoBackendError):
                print(f"File {audio_file_name.stem} was skipped because it couldnt be loaded and may be corrupted")
                continue

            genre_target_dir = pathlib.Path(target_dir_path).joinpath(genre_dir.stem)
            os.makedirs(genre_target_dir, exist_ok=True)

            specs_to_save = list()
            if splits_duration is not None:
                splits_timeseries = audio_processing.split_timeseries(audio_data, fragment_duration_sec=splits_duration)
                specs_to_save = [audio_processing.mel_spectrogram_from_timeseries(split,
                                                                                  mel_bands,
                                                                                  sr_overwrite,
                                                                                  log_scale=spec_log_scale)
                                 for split in splits_timeseries]
            else:
                mel_spec = audio_processing.mel_spectrogram_from_timeseries(audio_data,
                                                                            mel_bands,
                                                                            sr_overwrite,
                                                                            log_scale=spec_log_scale
                                                                            )
                specs_to_save.append(mel_spec)

            if save_as_binary:
                for spec in specs_to_save:
                    np.save(str(pathlib.Path(genre_target_dir).joinpath(audio_file_name.stem)), arr=spec)
            else:
                for part_num, spec in enumerate(specs_to_save):
                    spec_fig = plot_visuals.spec_to_img(spec)
                    spec_fig.savefig(fname=pathlib.Path(genre_target_dir).joinpath(
                        f"{audio_file_name.stem}_part{part_num}.png"), bbox_inches="tight", pad_inches=0.0)
                    plt.close(spec_fig)
            print(f"Done {audio_file_name.stem}")


def prepare_gtzan_dataset():
    pass


if __name__ == "__main__":
    convert_dataset_to_mel_spectrograms(
        original_gtzan_dir_path="/home/aleksy/dev/datasets/gtzan",
        target_dir_path="/home/aleksy/gtzan_spec_test",
        splits_duration=5.0
    )
