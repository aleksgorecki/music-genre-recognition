import os
import typing

import pandas as pd
import pathlib
import shutil
import warnings
import librosa
import multiprocessing


import gtzan_utils

FMALabels = ["Electronic", "Experimental", "Folk", "Hip-Hop", "Instrumental", "International", "Pop", "Rock"]


def get_id_from_filename(filename: str) -> int:
    return int(filename.lstrip('0'))


def get_filepath_from_track_id(track_id: int, fma_dir: str) -> str:
    zfilled_id = str(track_id).zfill(6)
    dir_name = zfilled_id[:3]
    filepath = pathlib.Path(fma_dir).joinpath(dir_name).joinpath(zfilled_id + ".mp3")
    return str(filepath)


def get_track_id_and_top_genre_pairing(tracks_csv_path: str) -> pd.DataFrame:
    # read only columns with track_id, subset name and top genre
    tracks_df = pd.read_csv(tracks_csv_path, usecols=[0, 32, 40])
    # drop rows with NaNs in top genre column
    tracks_df = tracks_df.dropna(subset=[tracks_df.columns[2]])
    # keep only rows with tracks belonging to small subset
    tracks_df = tracks_df[tracks_df.iloc[:, 1] == "small"]
    # keep only columns with track id and top genre
    pairing_df = tracks_df.iloc[:, [0, 2]]
    return pairing_df


def unnest_files_to_directory(fma_small_dir: str, dest_dir: str, clear_dest: bool = True) -> None:

    gtzan_utils.validate_destination_dir(dest_dir, clear_dest)

    dest_path = pathlib.Path(dest_dir)

    fma_small_path = pathlib.Path(fma_small_dir)
    nesting_dirs = [x for x in fma_small_path.iterdir() if x.is_dir()]
    assert(len(nesting_dirs) == 156)

    for ndir in nesting_dirs:
        for file in ndir.iterdir():
            shutil.copy(file, dest_path)


def dataframe_pairing_to_genre_dict(genre_filepath_pairing: pd.DataFrame) -> typing.Dict[str, typing.List[str]]:
    genres_names = genre_filepath_pairing.iloc[:, 1].unique()
    genre_dict = dict()
    for genre in genres_names:
        genre_files = genre_filepath_pairing[(genre_filepath_pairing.iloc[:, 1] == genre)].iloc[:, 2].tolist()
        genre_dict.update({genre: genre_files})

    return genre_dict


def get_filepath_to_top_genre_assignment(genre_id_pairing: pd.DataFrame, fma_small_dir: str) -> pd.DataFrame:
    new_pairing = genre_id_pairing.copy(deep=True)
    filepaths = list()
    for track_id in genre_id_pairing.iloc[:, 0]:
        filepath_str = get_filepath_from_track_id(track_id, fma_small_dir)
        filepaths.append(filepath_str)

    new_pairing.insert(loc=len(new_pairing.columns),column=2 ,value=filepaths)
    return new_pairing


def save_to_dir_in_gtzan_format(genre_dict: typing.Dict[str, typing.List[str]], dest_dir: str, clear_dest: bool = True):

    gtzan_utils.validate_destination_dir(dest_dir, clear_dest)

    dest_path = pathlib.Path(dest_dir)

    for genre in genre_dict.keys():
        genre_dir = dest_path.joinpath(genre)
        os.makedirs(genre_dir)
        for filepath in genre_dict[genre]:
            shutil.copy(filepath, genre_dir)


def shrink(origin_dir: str,
           dest_dir: str,
           max_items_per_genre: 200,
           shuffle: bool = True,
           seed: typing.Any = None,
           clear_dest: bool = True
           ):

    gtzan_utils.validate_destination_dir(dest_dir, clear_dest)
    dest_path = pathlib.Path(dest_dir)

    for genre_dir in pathlib.Path(origin_dir).iterdir():
        files = list(genre_dir.iterdir())
        assert(len(files) >= 200)
        if shuffle:
            files = gtzan_utils.shuffle_file_list(files, seed)
        files_shrinked = files[:max_items_per_genre]

        new_genre_path = dest_path.joinpath(genre_dir.stem)
        os.makedirs(new_genre_path)
        for file in files_shrinked:
            shutil.copy(file, new_genre_path)


def combine_with_gtzan(gtzan_original: str,
                   fma_gtzanlike_original: str,
                   dest_dir: str,
                   clear_dest: bool = True,
                   shuffle: bool = True,
                   seed: typing.Any = None
                   ):

    overlapping_gtzan = {
        "hiphop": "Hip-Hop",
        "rock": "Rock",
        "pop": "Pop",
    }

    overlapping_fma = {
        "Pop": "pop",
        "Rock": "rock",
        "Hip-Hop": "hiphop"
    }

    gtzan_utils.validate_destination_dir(dest_dir, clear_dest)

    dest_path = pathlib.Path(dest_dir)

    gtzan_path = pathlib.Path(gtzan_original)
    fma_path = pathlib.Path(fma_gtzanlike_original)

    for genre in gtzan_path.iterdir():
        if genre.stem not in overlapping_gtzan.keys():
            os.makedirs(dest_path.joinpath(genre.stem))
            for file in genre.iterdir():
                shutil.copy(file, dest_path.joinpath(genre.stem))

    for genre in fma_path.iterdir():
        if genre.stem not in overlapping_fma.keys():
            files = list(genre.iterdir())
            if shuffle:
                files = gtzan_utils.shuffle_file_list(files)
            files_to_copy = files[:100]
            os.makedirs(dest_path.joinpath(genre.stem))
            for file in files_to_copy:
                shutil.copy(file, dest_path.joinpath(genre.stem))

    for genre in overlapping_gtzan.keys():
        genre_path = gtzan_path.joinpath(genre)
        files = list(genre_path.iterdir())
        if shuffle:
            files = gtzan_utils.shuffle_file_list(files)
        files_to_copy = files[:50]
        os.makedirs(dest_path.joinpath(genre))
        for file in files_to_copy:
            shutil.copy(file, dest_path.joinpath(genre))
        genre_path = fma_path.joinpath(overlapping_gtzan[genre])
        files = list(genre_path.iterdir())
        if shuffle:
            files = gtzan_utils.shuffle_file_list(files)
        files_to_copy = files[:50]
        for file in files_to_copy:
            shutil.copy(file, dest_path.joinpath(genre))



if __name__ == "__main__":
    # df = get_track_id_and_top_genre_pairing("/home/aleksy/dev/datasets/fma_metadata/tracks.csv")
    # dff = get_filepath_to_top_genre_assignment(df, "/home/aleksy/dev/datasets/fma_small")
    # d = dataframe_pairing_to_genre_dict(genre_filepath_pairing=dff)
    # save_to_dir_in_gtzan_format(d, "/home/aleksy/fma_gtzanlike")
    # main_split_dict = gtzan_utils.get_splits_for_dataset(
    #     genres_dir="/home/aleksy/fma_gtzanlike"
    # )
    warnings.filterwarnings(category=UserWarning, action="ignore")
    # gtzan_utils.save_splits_as_mels(main_split_dict, destination_dir="/home/aleksy/fma_versions/fma_gtzanlike_5_sec_50", split_duration=5.0, overlap_ratio=0.5)
    # gtzan_utils.save_splits_as_mels(main_split_dict, destination_dir="/home/aleksy/fma_versions/fma_gtzanlike_5_sec_50_256",mel_bands=256 , split_duration=5.0, overlap_ratio=0.5)
    #gtzan_utils.shuffle_mix_song_parts("/home/aleksy/fma_versions/fma_gtzanlike_5_sec_50_256", destination_dir="/home/aleksy/fma_versions/fma_gtzanlike_5_sec_50_256_mixed", shuffle=True)
    # shrink("/home/aleksy/fma_gtzanlike", dest_dir="/home/aleksy/fma_versions/fma_shrinked_200", max_items_per_genre=200)
    # shrink("/home/aleksy/fma_gtzanlike", dest_dir="/home/aleksy/fma_versions/fma_shrinked_100", max_items_per_genre=100)

    # main_split_dict = gtzan_utils.get_splits_for_dataset(
    #      genres_dir="/home/aleksy/combined_dataset"
    # )
    # gtzan_utils.save_splits_as_mels(main_split_dict, destination_dir="/home/aleksy/combined_5_sec_50_256", mel_bands=256, split_duration=5.0, overlap_ratio=0.5)
    # gtzan_utils.shuffle_mix_song_parts("/home/aleksy/combined_5_sec_50_256", destination_dir="/home/aleksy/combined_5_sec_50_256_mixed", shuffle=True)
