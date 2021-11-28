import typing

import audio_processing
import plot_visuals


import audioread.exceptions
import os
import pathlib
from matplotlib import pyplot as plt
import shutil
import argparse
import random
from dataclasses import dataclass
import numpy as np


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


def ratio_to_rounded_indices(ratio: typing.Tuple[float, float, float],
                             whole_dir_num: int
                             ) -> typing.Tuple[int, int, int]:

    # Lists of numbers of files inside split (train, test and val) calculated from ratio,
    # as floats and as integers after floor
    nums_f = [ratio[0] * whole_dir_num, ratio[1] * whole_dir_num, ratio[2] * whole_dir_num]
    nums = [int(num_f) for num_f in nums_f]

    # Check if flooring the floats made the sum lower than all the files to use
    unassigned_files = whole_dir_num - sum(nums)
    assert(unassigned_files >= 0)

    # Choose the splits with biggest decimals as the candidates for incrementation to assign the unassigned files
    if unassigned_files != 0:
        for idx in np.argsort(nums_f)[-unassigned_files:]:
            nums[idx] += 1

    # indices for slicing the list of all files inside directory to create the splits
    train_idx = nums[0]
    test_idx = nums[0] + nums[1]
    val_idx = nums[0] + nums[1] + nums[2]

    return train_idx, test_idx, val_idx


def shuffle_file_list(fl: typing.List[pathlib.Path], seed: typing.Any = None) -> typing.List[pathlib.Path]:
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

    train_idx, test_idx, val_idx = ratio_to_rounded_indices((train_ratio, test_ratio, val_ratio), len(files_to_split))

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
        genre_split = get_splits_for_genre(str(genre_dir), shuffle, seed, train_ratio, test_ratio, val_ratio)
        dataset_split_dict.add_genre_record(genre_dir.stem, genre_split)
    return dataset_split_dict


def validate_destination_dir(dest: str, clear_existing_dest: bool) -> bool:

    dest_path = pathlib.Path(dest)

    if dest_path.exists():
        if not dest_path.is_dir() and not clear_existing_dest:
            return False
        if len(os.listdir(dest_path)) != 0 and not clear_existing_dest:
            return False
        shutil.rmtree(dest_path)
    os.makedirs(dest_path)
    return True


def save_single_split_as_mels(split: typing.Dict["str", typing.List[pathlib.Path]],
                              split_name: str,
                              destination_dir: str,
                              mel_bands: int = 128,
                              sr_overwrite: int = None,
                              split_duration: float = None,
                              spec_log_scale: bool = True,
                              dpi: int = 100
                              ) -> None:

    fig, ax = plt.subplots()
    plt.close(fig)

    for genre_name in split.keys():

        genre_destination_dir = pathlib.Path(destination_dir).joinpath(split_name).joinpath(genre_name)
        os.makedirs(genre_destination_dir, exist_ok=True)

        source_files = split[genre_name]
        for source_file in source_files:

            try:
                audio_data = audio_processing.load_to_mono(str(source_file), sr_overwrite)
            except (RuntimeError, audioread.exceptions.NoBackendError):
                print(f"(!) File {source_file.stem} was skipped because it couldnt be loaded and may be corrupted.")
                continue

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
                mel_save_filename = pathlib.Path(genre_destination_dir).joinpath(
                    f"{source_file.stem}_part{part_num}.png"
                )
                fig.savefig(fname=mel_save_filename, bbox_inches="tight", pad_inches=0.0, dpi=dpi)

            print(f"Current state: Done {split_name}: {genre_name}: {source_file.stem}", flush=True)


def save_splits_as_mels(dataset_splits: DatasetSplitDict,
                        destination_dir: str,
                        clear_dest: bool = True,
                        mel_bands: int = 128,
                        sr_overwrite: int = None,
                        split_duration: float = None,
                        spec_log_scale: bool = True,
                        dpi: int = 100
                        ) -> None:

    assert(len(dataset_splits.split_dict.keys()) > 0)

    if not validate_destination_dir(destination_dir, clear_dest):
        raise Exception

    for split_name in dataset_splits.split_dict.keys():
        os.makedirs(pathlib.Path(destination_dir).joinpath(split_name))
        save_single_split_as_mels(dataset_splits.split_dict[split_name],
                                  split_name,
                                  destination_dir,
                                  mel_bands,
                                  sr_overwrite,
                                  split_duration,
                                  spec_log_scale,
                                  dpi
                                  )

    print("Saved.")


# def genres_audio_to_mel(genres_dir: str,
#                         dest_dir: str,
#                         clear_dest: bool = True,
#                         mel_bands: int = 128,
#                         sr_overwrite: int = None,
#                         split_duration: float = None,
#                         spec_log_scale: bool = True,
#                         ) -> None:
#
#     fig, ax = plt.subplots()
#     plt.close(fig)
#
#     if not validate_destination_dir(dest_dir, clear_dest):
#         raise Exception
#
#     for genre_dir in pathlib.Path(genres_dir).iterdir():
#         print(f"Starting converting genre: {genre_dir.stem}...")
#
#         for audio_file_path in tqdm.tqdm(genre_dir.iterdir()):
#
#             try:
#                 audio_data = audio_processing.load_to_mono(str(audio_file_path), sr_overwrite)
#             except (RuntimeError, audioread.exceptions.NoBackendError):
#                 print(f"(!) File {audio_file_path.stem} was skipped because it couldnt be loaded and may be corrupted.")
#                 continue
#
#             genre_target_dir = pathlib.Path(dest_dir).joinpath(genre_dir.stem)
#             os.makedirs(genre_target_dir, exist_ok=True)
#
#             mels_to_save = list()
#             if split_duration is not None:
#                 splits = audio_processing.split_timeseries(audio_data, fragment_duration_sec=split_duration)
#                 mels_to_save = [
#                     audio_processing.mel_from_timeseries(x, mel_bands, sr_overwrite, spec_log_scale)
#                     for x in splits
#                 ]
#             else:
#                 mel = audio_processing.mel_from_timeseries(audio_data, mel_bands,
#                                                            sr_overwrite, log_scale=spec_log_scale)
#                 mels_to_save.append(mel)
#
#             for part_num, mel in enumerate(mels_to_save):
#                 plot_visuals.mel_only_on_ax(mel, ax)
#                 mel_save_filename = pathlib.Path(genre_target_dir).joinpath(f"{audio_file_path.stem}_part{part_num}.png")
#                 fig.savefig(fname=mel_save_filename, bbox_inches="tight", pad_inches=0.0)
#             if dest_dir is None:
#                 os.remove(audio_file_path)


if __name__ == "__main__":
    # DEFAULT_SPLIT_DEST = "/home/aleksy/gtzan_split/"
    # DEFAULT_MEL_DEST = "/home/aleksy/gtzan_mel/"
    # DEFAULT_SPLIT_DURATION = 5.0
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument()

    main_split_dict = get_splits_for_dataset(
        genres_dir="/home/aleksy/dev/datasets/gtzan/Data/genres_original"
    )
    save_splits_as_mels(main_split_dict, destination_dir="/home/aleksy/gtzan_spec_2", split_duration=5.0)
