import numpy as np
import numpy.typing
import librosa
import librosa.core
import typing


class LibrosaMonoTimeseries:
    timeseries: numpy.typing.ArrayLike
    sr: int

    def __init__(self, librosa_timeseries_tuple: typing.Tuple[numpy.typing.ArrayLike, int]):
        self.timeseries = librosa_timeseries_tuple[0]
        self.sr = librosa_timeseries_tuple[1]


def load_audio_and_convert_to_mono(file_path: str,
                                   sr_overwrite: typing.Optional[int] = None
                                   ) -> LibrosaMonoTimeseries:
    if sr_overwrite is not None:
        sr = sr_overwrite
    else:
        sr = 22050
    audio_data = librosa.core.load(path=file_path, sr=sr, mono=True)
    return LibrosaMonoTimeseries(audio_data)


def mel_spectrogram_from_timeseries(audio_data: LibrosaMonoTimeseries,
                                    mel_bands: int = 128,
                                    sr_overwrite: typing.Optional[int] = None,
                                    log_scale: bool = True
                                    ) -> typing.Any:
    if sr_overwrite is not None:
        fs = sr_overwrite
    else:
        fs = audio_data.sr
    mel_spec = librosa.feature.melspectrogram(y=audio_data.timeseries, sr=fs, n_mel=mel_bands)
    if log_scale:
        mel_spec = librosa.core.power_to_db(mel_spec, ref=np.max)
    return mel_spec


def pad_timeseries(audio_data: LibrosaMonoTimeseries,
                   target_length: int,
                   padding_value: float = 0
                   ) -> LibrosaMonoTimeseries:
    num_samples_to_add = int(target_length - len(audio_data.timeseries))
    padded_timeseries = np.pad(audio_data.timeseries, (0, num_samples_to_add))
    return LibrosaMonoTimeseries((padded_timeseries, audio_data.sr))


def split_timeseries(audio_data: LibrosaMonoTimeseries,
                     fragment_duration_ms: int,
                     pad_incomplete: bool = False
                     ) -> typing.List[LibrosaMonoTimeseries]:
    fragment_duration_in_samples = int(fragment_duration_ms * audio_data.sr / 1000)
    num_fragments = int(len(audio_data.timeseries) / fragment_duration_in_samples)
    for i in range(0, num_fragments):
        pass


def get_fragment_of_timeseries(audio_data: LibrosaMonoTimeseries,
                               fragment_duration_ms: int
                               ) -> LibrosaMonoTimeseries:
    fragment_duration_in_samples = int(fragment_duration_ms) * audio_data.sr
    pass
