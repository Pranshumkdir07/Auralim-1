"""
features.py
===========
Turns one 5-second audio clip into ONE fixed-length list of numbers (a
"feature vector"), so that a machine-learning model can compare clips.

Why not just feed raw audio to the model?
    A 5 s clip at 16 kHz is 80,000 numbers. Models need the SAME number of
    inputs for every clip, and raw samples depend on *when* the sound happens.
    So instead we summarise WHAT the sound is made of:
      - MFCCs           -> the "texture"/timbre of the sound (what it sounds like)
      - deltas          -> how fast that texture is changing (e.g. a siren sweeping)
      - spectral centroid -> how "bright" the sound is (low = truck, high = hiss)
      - zero-crossing rate -> how "noisy"/non-tonal the sound is

Output = 40 MFCC means + 40 MFCC stds
       + 40 delta means + 40 delta stds
       + 2 extra features (centroid, ZCR)  ->  162 numbers per clip.
"""

import librosa
import numpy as np

# All clips are converted to 16 kHz mono. This keeps the maths small/fast and
# is the same sample rate the live microphone and the level meter use.
SAMPLE_RATE = 16000
N_MFCC = 40          # number of MFCC coefficients per frame
N_FFT = 1024         # window size for the short-time Fourier transform
HOP_LENGTH = 256     # step between frames (16 kHz / 256 = 62.5 frames per second)


def extract_mfcc_features(y, sr=SAMPLE_RATE):
    """
    Core feature extraction: works on ANY clip length.

    `y` is a 1-D numpy array of samples in the range [-1, 1].
    Returns an array of shape (2*N_MFCC + 2*N_MFCC + 2,) = (162,).
    """
    y = np.asarray(y, dtype=np.float32)

    # 1) MFCCs. "librosa.feature.mfcc" compresses the spectrum into 40 numbers
    #    per frame, which roughly describes the shape of a human's hearing.
    #    Result shape: (40 coefficients, number_of_frames).
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)

    # 2) Deltas = the difference between neighbouring frames. This captures how
    #    the sound CHANGES over time (a siren wails, an engine is steady).
    delta = librosa.feature.delta(mfcc)

    # 3) Each of the above is a (40, frames) matrix. We compress the time axis
    #    by taking the mean and the standard deviation (spread) of every row.
    #    Now we have 40 + 40 + 40 + 40 = 160 numbers that do not depend on
    #    clip length. (librosa returns real-valued arrays, so we don't need
    #    librosa.feature.delta with width issues; `.mean(axis=1)` is float.)
    parts = [mfcc.mean(axis=1), mfcc.std(axis=1),
             delta.mean(axis=1), delta.std(axis=1)]

    # 4) Spectral centroid: the "centre of gravity" of the frequency content,
    #    in Hz. Averaged over time -> 1 number.
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT,
                                                 hop_length=HOP_LENGTH)
    parts.append(np.array([centroid.mean()], dtype=np.float32))

    # 5) Zero-crossing rate: how often the waveform crosses zero. Noisy sounds
    #    (rain, hiss) cross a lot, low rumble (engine) crosses rarely. 1 number.
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT,
                                             hop_length=HOP_LENGTH)
    parts.append(np.array([zcr.mean()], dtype=np.float32))

    # np.concatenate glues the pieces into one flat vector of 162 values.
    return np.concatenate(parts).astype(np.float32)


def load_clip(path, sr=SAMPLE_RATE):
    """
    Read a WAV file from disk as 16 kHz mono float32 samples, then return
    (signal, feature_vector).
    """
    # mono=True mixes stereo to one channel; sr=16000 resamples for us.
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y, extract_mfcc_features(y, sr)


class FeatureExtractor:
    """
    A tiny "transformer" wrapper whose only job is to make this feature code
    usable inside a scikit-learn Pipeline. It behaves like a class with a
    `.transform(list_of_signals)` method.
    """

    def fit(self, X, y=None):
        # Nothing to learn here - features are computed with fixed formulas.
        return self

    def transform(self, X):
        return np.vstack([extract_mfcc_features(sig, SAMPLE_RATE) for sig in X])
