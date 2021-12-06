import numpy as np
import numpy.typing
import typing
import librosa
import librosa.core
import audiofile
import os
from dataclasses import dataclass


@dataclass
class LibrosaMonoTimeseries:

    timeseries: numpy.typing.ArrayLike
    sr: int

    def __init__(self, librosa_timeseries_tuple: typing.Tuple[numpy.typing.ArrayLike, int]):
        self.timeseries = librosa_timeseries_tuple[0]
        self.sr = librosa_timeseries_tuple[1]


def load_to_mono(file_path: str,
                 sr_overwrite: typing.Optional[int] = None,
                 offset_sec: float = 0.0,
                 duration: float = None
                 ) -> LibrosaMonoTimeseries:

    if sr_overwrite is not None:
        sr = sr_overwrite
    else:
        sr = 22050
    audio_data = librosa.core.load(path=file_path, sr=sr, mono=True, offset=offset_sec, duration=duration)
    return LibrosaMonoTimeseries(audio_data)


def mel_from_timeseries(audio_data: LibrosaMonoTimeseries,
                        mel_bands: int = 128,
                        sr_overwrite: typing.Optional[int] = None,
                        log_scale: bool = True
                        ) -> typing.Any:

    if sr_overwrite is not None:
        sr = sr_overwrite
    else:
        sr = audio_data.sr
    mel_spec = librosa.feature.melspectrogram(y=audio_data.timeseries, sr=sr, n_mels=mel_bands)
    if log_scale:
        mel_spec = librosa.core.power_to_db(mel_spec, ref=np.max)
    return mel_spec


def split_timeseries(audio_data: LibrosaMonoTimeseries,
                     fragment_duration_sec: float,
                     overlap_ratio: float = 0.50,
                     ) -> typing.List[LibrosaMonoTimeseries]:

    fragment_duration_in_samples = int(fragment_duration_sec * audio_data.sr)

    # split_points_indices = list(range(fragment_duration_in_samples,
    #                                   len(audio_data.timeseries),
    #                                   fragment_duration_in_samples)
    #                             )
    #
    # timeseries_splits = np.split(audio_data.timeseries, indices_or_sections=split_points_indices)

    timeseries_splits = list()

    offset_len = int(fragment_duration_in_samples * (1 - overlap_ratio))

    for offset in range(0, len(audio_data.timeseries) - fragment_duration_in_samples, offset_len):
        timeseries_splits.append(audio_data.timeseries[offset:offset + fragment_duration_in_samples])

    if len(timeseries_splits) != 0:
        if len(timeseries_splits[-1]) != fragment_duration_in_samples:
            timeseries_splits.pop()

    splits = list()
    for split in timeseries_splits:
        splits.append(LibrosaMonoTimeseries((split, audio_data.sr)))

    return splits


def get_fragment_of_timeseries(audio_data: LibrosaMonoTimeseries,
                               offset_sec: float,
                               fragment_duration_sec: float
                               ) -> LibrosaMonoTimeseries:

    fragment_duration_in_samples = int(fragment_duration_sec * audio_data.sr)
    offset_in_samples = int(offset_sec * audio_data.sr)
    fragment = np.array(audio_data.timeseries[offset_in_samples:offset_in_samples + fragment_duration_in_samples],
                        dtype=audio_data.timeseries.dtype)

    return LibrosaMonoTimeseries((fragment, audio_data.sr))


def stream_generator(file: str, tmp_filepath: str, frame_sec: float = 5.0):
    frame_length = int(frame_sec * audiofile.sampling_rate(file))
    hop_length = frame_length
    block_len = 1

    try:
        stream = librosa.core.stream(path=tmp_filepath,
                                     block_length=block_len,
                                     frame_length=frame_length,
                                     hop_length=hop_length,
                                     mono=True
                                     )
    except RuntimeError:
        if os.path.exists(tmp_filepath):
            os.remove(tmp_filepath)
        audiofile.convert_to_wav(file, tmp_filepath)
        stream = librosa.core.stream(path=tmp_filepath,
                                     block_length=block_len,
                                     frame_length=frame_length,
                                     hop_length=hop_length,
                                     mono=True
                                     )
    return stream
