import typing

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
from dataclasses import dataclass


GTZANLabels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]


@dataclass
class DatasetSplitDict:
    split_dict: typing.Dict[str, typing.Dict[str, typing.List[pathlib.Path]]]
    splits_names = ["train", "test", "val"]

    def __init__(self, original_genres_path: str = None):
        self.split_dict = {
            "train": {},
            "test": {},
            "val": {}
        }
        self.original_genres_path = original_genres_path

    def add_genre_record(self, genre_name: str, record: typing.Dict[str, typing.List[pathlib.Path]]) -> None:
        assert(sorted(DatasetSplitDict.splits_names) == sorted(list(record.keys())))

        for split in DatasetSplitDict.splits_names:
            self.split_dict[split].update(
                {genre_name: record[split]}
            )

    # def add_genre_record(self, genre: str, other: "DatasetSplitDict"):
    #     assert(genre in other.split_dict["train"].keys()
    #            and genre in other.split_dict["test"].keys()
    #            and genre in other.split_dict["val"].keys()
    #            )
    #
    #     assert(genre not in self.split_dict["train"].keys()
    #            and genre not in self.split_dict["test"].keys()
    #            and genre not in self.split_dict["test"].keys()
    #            )
    #
    #     for split in ["train", "test", "val"]:
    #         self.split_dict[split].update(
    #             {genre: other.split_dict[split][genre]}
    #         )


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
    os.makedirs(dest_path)
    return True


def ratio_to_rounded_ids(ratio: typing.Tuple[float, float, float], whole_dir_num: int) -> typing.Tuple[int, int, int]:
    num_train_files = int(ratio[0] * whole_dir_num)
    num_test_files = int(ratio[1] * whole_dir_num)
    num_val_files = int(ratio[2] * whole_dir_num)

    unassigned_files = whole_dir_num - (num_val_files + num_train_files + num_test_files)

    if unassigned_files == 1:
        num_train_files += 1
    elif unassigned_files == 2:
        num_train_files += 1
        num_test_files += 1

    train_idx = num_train_files
    test_idx = num_train_files + num_test_files
    val_idx = num_train_files + num_test_files + num_val_files

    return train_idx, test_idx, val_idx


def shuffle_file_list(fl: typing.List[pathlib.Path], seed: typing.Any = None):
    if seed is not None:
        gen = random.Random(seed)
    else:
        gen = random.Random()
    shuffled = fl.copy()
    gen.shuffle(shuffled)
    return shuffled


def get_splits_for_genre(genre_dir: str,
                         shuffle: bool = True,
                         seed: int = None,
                         train_ratio: float = 0.70,
                         test_ratio: float = 0.20,
                         val_ratio: float = 0.10
                         ) -> typing.Dict[str, typing.List[pathlib.Path]]:

    assert(round(train_ratio + test_ratio + val_ratio) == 1)

    genres_dir_path = pathlib.Path(genre_dir)
    files_to_split = list(genres_dir_path.iterdir())
    if shuffle:
        files_to_split = shuffle_file_list(files_to_split, seed)

    train_idx, test_idx, val_idx = ratio_to_rounded_ids((train_ratio, test_ratio, val_ratio), len(files_to_split))

    files = {
        "train": files_to_split[:train_idx],
        "test": files_to_split[train_idx:test_idx],
        "val": files_to_split[test_idx:]
    }

    return files


def get_splits_for_dataset(genres_dir: str,
                           shuffle: bool = True,
                           seed: int = None,
                           train_ratio: float = 0.70,
                           test_ratio: float = 0.20,
                           val_ratio: float = 0.10
                           ) -> DatasetSplitDict:

    dataset_split_dict = DatasetSplitDict()
    for genre_dir in pathlib.Path(genres_dir).iterdir():
        genre_split = get_splits_for_genre(genre_dir, shuffle, seed, train_ratio, test_ratio, val_ratio)
        dataset_split_dict.add_genre_record(genre_dir.stem, genre_split)
    return dataset_split_dict


def genre_to_mel():
    pass


def split_to_mel():
    pass


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

def save_split_to_dir_original():
    pass


def save_split_to_dir_as_mels():
    pass

if __name__ == "__main__":
    # DEFAULT_SPLIT_DEST = "/home/aleksy/gtzan_split/"
    # DEFAULT_MEL_DEST = "/home/aleksy/gtzan_mel/"
    # DEFAULT_SPLIT_DURATION = 5.0
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument()

    get_splits_for_genre(
        genre_dir="/home/aleksy/genres_original/blues"

    )
    get_splits_for_dataset("/home/aleksy/genres_original")
    # genres_audio_to_mel()