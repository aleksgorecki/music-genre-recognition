import typing

import pandas as pd
import pathlib
import shutil


import gtzan_utils


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


def dataframe_pairing_to_genre_dict(genre_filepath_pairing: pd.DataFrame) -> typing.Dict[str, typing.List[int]]:
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


if __name__ == "__main__":
    df = get_track_id_and_top_genre_pairing("/home/aleksy/dev/datasets/fma_metadata/tracks.csv")
    dff = get_filepath_to_top_genre_assignment(df, "/home/aleksy/dev/datasets/fma_small")
    d = dataframe_pairing_to_genre_dict(genre_filepath_pairing=dff)

    #df2 = get_filepath_to_top_genre_assignment(df, "/home/aleksy/dev/datasets/fma_small")
    #unnest_files_to_directory("/home/aleksy/dev/datasets/fma_small", "/home/aleksy/fma_unnested")
    pass
