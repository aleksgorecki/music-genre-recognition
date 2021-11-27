import numpy as np
import numpy.typing
import typing
import librosa
import librosa.core
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
                     ) -> typing.List[LibrosaMonoTimeseries]:

    fragment_duration_in_samples = int(fragment_duration_sec * audio_data.sr)

    split_points_indices = list(range(fragment_duration_in_samples,
                                      len(audio_data.timeseries),
                                      fragment_duration_in_samples)
                                )

    timeseries_splits = np.split(audio_data.timeseries, indices_or_sections=split_points_indices)
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
