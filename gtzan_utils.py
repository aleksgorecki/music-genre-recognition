import audio_processing
import plot_visuals


import audioread.exceptions
import os
import pathlib
from matplotlib import pyplot as plt
import shutil
import argparse
import tqdm
import random
import time


GTZANLabels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]


def prepare_destination_dir(dest: str, clear_existing_dest: bool) -> bool:

    if dest is None and clear_existing_dest is False:
        return False

    dest_path = pathlib.Path(dest)

    if dest_path.exists():
        if not dest_path.is_dir():
            return False
        if len(os.listdir(dest_path)) != 0 and not clear_existing_dest:
            return False
        shutil.rmtree(dest_path)
    else:
        os.makedirs(dest_path)
        return True


def genres_audio_to_mel(genres_dir: str,
                        dest_dir: str,
                        clear_dest: bool = True,
                        mel_bands: int = 128,
                        sr_overwrite: int = None,
                        split_duration: float = None,
                        spec_log_scale: bool = True,
                        ) -> None:

    fig, ax = plt.subplots()
    plt.close(fig)

    if not prepare_destination_dir(dest_dir, clear_dest):
        raise Exception

    for genre_dir in pathlib.Path(genres_dir).iterdir():
        print(f"Starting converting genre: {genre_dir.stem}...")

        for audio_file_path in tqdm.tqdm(genre_dir.iterdir()):

            try:
                audio_data = audio_processing.load_to_mono(str(audio_file_path), sr_overwrite)
            except (RuntimeError, audioread.exceptions.NoBackendError):
                print(f"(!) File {audio_file_path.stem} was skipped because it couldnt be loaded and may be corrupted.")
                continue

            genre_target_dir = pathlib.Path(dest_dir).joinpath(genre_dir.stem)
            os.makedirs(genre_target_dir, exist_ok=True)

            mels_to_save = list()
            if split_duration is not None:
                splits = audio_processing.split_timeseries(audio_data, fragment_duration_sec=split_duration)
                mels_to_save = [
                    audio_processing.mel_from_timeseries(x, mel_bands, sr_overwrite, spec_log_scale)
                    for x in splits
                ]
            else:
                mel = audio_processing.mel_from_timeseries(audio_data, mel_bands,
                                                           sr_overwrite, log_scale=spec_log_scale)
                mels_to_save.append(mel)

            for part_num, mel in enumerate(mels_to_save):
                plot_visuals.mel_only_on_ax(mel, ax)
                mel_save_filename = pathlib.Path(genre_target_dir).joinpath(f"{audio_file_path.stem}_part{part_num}.png")
                fig.savefig(fname=mel_save_filename, bbox_inches="tight", pad_inches=0.0)
            if dest_dir is None:
                os.remove(audio_file_path)


def split_dataset(dest_dir: str,
                  clear_existing_dest: bool = True,
                  shuffle: bool = True,
                  seed: int = None,
                  train_ratio: float = 0.70,
                  test_ratio: float = 0.20,
                  val_ratio: float = 0.10
                  ) -> None:

    if round(train_ratio + test_ratio + val_ratio, ndigits=1) != 1.0:
        raise ValueError

    if not prepare_destination_dir(dest_dir, clear_existing_dest):
        raise Exception

    os.makedirs(dest_dir, exist_ok=True)

    dest_dir_path = pathlib.Path(dest_dir)

    files_to_split = list(dest_dir_path.iterdir())
    if shuffle:
        if seed is not None:
            gen = random.Random(seed)
        else:
            gen = random.Random()
        gen.shuffle(files_to_split)

    num_all_files = len(files_to_split)
    num_train_files = int(train_ratio * num_all_files)
    num_test_files = int(test_ratio * num_all_files)
    num_val_files = int(val_ratio * num_all_files)

    print(f"Missing files: {num_all_files - (num_val_files + num_train_files + num_test_files)}")

    train_files = files_to_split[0:num_train_files]
    test_files = files_to_split[num_train_files:num_test_files]
    val_files = files_to_split[num_test_files:num_val_files]

    splits = [train_files, test_files, val_files]
    splits_dir_names = ['train', 'test', 'val']

    for idx, split in enumerate(splits):
        split_dir_name = splits_dir_names[idx]
        print(f"Copying {split_dir_name} files...")
        for file in tqdm.tqdm(split):
            shutil.copy(file, dest_dir_path.joinpath(split_dir_name))
            if dest_dir is None:
                os.remove(file)



if __name__ == "__main__":
    DEFAULT_SPLIT_DEST = "/home/aleksy/gtzan_split/"
    DEFAULT_MEL_DEST = "/home/aleksy/gtzan_mel/"
    DEFAULT_SPLIT_DURATION = 5.0

    parser = argparse.ArgumentParser()
    parser.add_argument()

    split_dataset()
    genres_audio_to_mel()