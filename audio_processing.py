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


def load_audio_file_and_convert_to_mono(file_path: str,
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


def mel_spectrogram_from_timeseries(audio_data: LibrosaMonoTimeseries,
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


def pad_timeseries(audio_data: LibrosaMonoTimeseries,
                   target_length: int,
                   ) -> LibrosaMonoTimeseries:
    num_samples_to_add = int(target_length - len(audio_data.timeseries))
    padded_timeseries = np.pad(audio_data.timeseries, (0, num_samples_to_add))
    return LibrosaMonoTimeseries((padded_timeseries, audio_data.sr))


def split_timeseries(audio_data: LibrosaMonoTimeseries,
                     fragment_duration_sec: float,
                     pad_incomplete: bool = False
                     ) -> typing.List[LibrosaMonoTimeseries]:
    fragment_duration_in_samples = int(fragment_duration_sec * audio_data.sr)
    splits = list()
    for i in range(0, len(audio_data.timeseries), fragment_duration_in_samples):
        split = np.ndarray(audio_data.timeseries[i:i + fragment_duration_in_samples],
                           dtype=audio_data.timeseries.dtype)
        splits.append(LibrosaMonoTimeseries((split, audio_data.sr)))

    num_fragments = len(splits)
    if pad_incomplete and fragment_duration_in_samples*num_fragments < len(audio_data.timeseries):
        incomplete_split = np.ndarray(audio_data.timeseries[fragment_duration_in_samples*num_fragments:],
                                      dtype=audio_data.timeseries.dtype)
        padded_split = pad_timeseries(LibrosaMonoTimeseries((incomplete_split, audio_data.sr)),
                                      fragment_duration_in_samples)
        splits.append(padded_split)

    return splits


def get_fragment_of_timeseries(audio_data: LibrosaMonoTimeseries,
                               offset_sec: float,
                               fragment_duration_sec: float
                               ) -> LibrosaMonoTimeseries:
    fragment_duration_in_samples = int(fragment_duration_sec * audio_data.sr)
    offset_in_samples = int(offset_sec * audio_data.sr)
    fragment = np.ndarray(audio_data.timeseries[offset_in_samples:offset_in_samples + fragment_duration_in_samples],
                          dtype=audio_data.timeseries.dtype)
    return LibrosaMonoTimeseries((fragment, audio_data.sr))
