import audioread.exceptions

import audio_processing
import typing
import os
import numpy as np
import pathlib


class GTZANDataset:
    num_classes: int = 10
    class_labels: typing.List[str] = ["blues", "classical", "country", "disco",
                                      "hiphop", "jazz", "metal", "pop", "reggae", "rock"]


def convert_dataset_to_mel_spectrograms(original_gtzan_dir_path: str,
                                        target_dir_path: str,
                                        mel_bands: int = 128,
                                        sr_overwrite: int = None,
                                        splits_duration: int = None,
                                        spec_log_scale: bool = True,
                                        save_as_images: bool = False
                                        ) -> None:
    for genre_dir in (pathlib.Path(original_gtzan_dir_path).joinpath("Data/genres_original/")).iterdir():
        print(f"Starting {genre_dir.stem}")
        for audio_file_name in genre_dir.iterdir():

            try:
                audio_data = audio_processing.load_audio_file_and_convert_to_mono(str(audio_file_name),
                                                                                  sr_overwrite)
            except RuntimeError and audioread.exceptions.NoBackendError:
                print(f"File {audio_file_name.stem} was skipped because it couldnt be loaded and may be corrupted")
                continue

            mel_spec = audio_processing.mel_spectrogram_from_timeseries(audio_data,
                                                                        mel_bands,
                                                                        sr_overwrite,
                                                                        log_scale=spec_log_scale
                                                                        )
            genre_target_dir = pathlib.Path(target_dir_path).joinpath(genre_dir.stem)
            os.makedirs(genre_target_dir, exist_ok=True)

            if splits_duration is not None:
                pass

            np.save(str(pathlib.Path(genre_target_dir).joinpath(audio_file_name.stem)), arr=mel_spec)

        print(f"Converted {genre_dir.stem} to {genre_target_dir}")


if __name__ == "__main__":
    convert_dataset_to_mel_spectrograms(
        original_gtzan_dir_path="/home/aleksy/dev/datasets/gtzan",
        target_dir_path="/home/aleksy/gtzan_spec_test"
    )
