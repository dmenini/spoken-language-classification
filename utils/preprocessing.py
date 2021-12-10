import librosa as lr
import numpy as np


def audio_to_melspectrogram(path, sampling_rate=16_000, trim_length=100_000, height=192, width=192):
    signal = load_audio(path, sampling_rate=sampling_rate)
    signal = trim_audio(signal, trim_length)
    signal = normalize_audio(signal)
    img = get_mel_spectrogram(signal, sampling_rate, height, width)
    return img


def audio_to_mfcc(path, sampling_rate=16_000, trim_length=100_000):
    try:
        signal = load_audio(path, sampling_rate=sampling_rate)
        signal = trim_audio(signal, trim_length)
        # signal = normalize_audio(signal)
        mfcc = get_mfcc(signal, sampling_rate)
        return mfcc
    except:
        print("error with ", path)


def load_audio(path, sampling_rate=16_000):
    audio, _ = lr.load(path, res_type='kaiser_fast', sr=sampling_rate)
    return np.array(audio)


def trim_audio(signal, trim_length=100_000):
    signal_length = signal.shape[0]
    if signal_length <= trim_length:
        signal = np.pad(signal, (0, trim_length - signal_length), mode='wrap')
    elif trim_length < signal_length <= trim_length * 1.1:
        signal = signal[(signal_length - trim_length):]
    else:
        signal = signal[int(trim_length * 0.1):int(trim_length * 1.1)]
    return signal.astype(np.float)


def normalize_audio(signal):
    return (signal - np.mean(signal, axis=0)) / np.abs(signal).max(axis=0)


def get_mel_spectrogram(signal, sr, height=192, width=192):
    hl = signal.shape[0] // width
    spec = lr.feature.melspectrogram(signal, sr=sr, n_mels=height, hop_length=int(hl))
    img = lr.power_to_db(spec)
    start = (img.shape[1] - width) // 2
    return img[:, start:start + width]


def get_mfcc(signal, sr):
    mfcc = lr.feature.mfcc(signal, sr=sr)
    return mfcc
