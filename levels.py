"""
levels.py
=========
Hospital sound levels, implemented from scratch with numpy/scipy.

Everything here is built from the textbook definitions instead of using a
black-box dB library, so every step can be explained on a slide:

    RMS        -> Leq (equivalent continuous level)  -> Lmax (max short-term level)
    A-weighting filter (IEC 61672) -> dB(A) levels
    Butterworth band-pass filter   -> octave-band spectrum

IMPORTANT - RELATIVE dB:
    We do NOT have a calibrated measurement microphone, so absolute dB SPL
    cannot be claimed. We use a digital full-scale reference (RMS = 1.0 -> 0 dB)
    exactly like dBFS. All levels are therefore RELATIVE dB re full scale and
    can only be compared with each other (or after adding a calibration offset).
"""

import numpy as np
from scipy import signal
from scipy.signal import bilinear, butter, sosfilt, sosfiltfilt, sosfreqz

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FS = 16000                      # sample rate of all audio in this project
REF = 1.0                       # full-scale reference -> 0 dB relative
MIN_LEVEL = -120.0              # "silence floor": avoids log10(0) = -inf

# 1/1-octave band centre frequencies (Hz), IEC 61260 preferred series.
# 8 kHz stays in the list, but its upper edge is capped below Nyquist at 16 kHz.
OCTAVE_BANDS = [63, 125, 250, 500, 1000, 2000, 4000, 8000]


# ---------------------------------------------------------------------------
# 1. Basic level maths
# ---------------------------------------------------------------------------
def rms(x):
    """Root-mean-square amplitude: sqrt(mean(x^2)). Energy-like 'loudness'."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x ** 2)))


def to_db(amplitude, ref=REF):
    """Convert a linear amplitude (RMS or peak) to dB relative to `ref`."""
    # 20*log10 because dB is defined for amplitude, not for power.
    return 20.0 * np.log10(max(float(amplitude), 1e-12) / ref)


def from_db(db, ref=REF):
    """Inverse of to_db: dB -> linear amplitude."""
    return ref * 10.0 ** (np.asarray(db, dtype=np.float64) / 20.0)


def leq(x, ref=REF):
    """
    Equivalent continuous sound level of the whole buffer:
        Leq = 10*log10( mean(x^2) / ref^2 )  =  20*log10( RMS / ref )
    It is the "steady sound that would carry the same energy" as the signal.
    """
    return max(to_db(rms(x), ref), MIN_LEVEL)


def short_term_levels(x, fs=FS, frame_seconds=0.125, ref=REF):
    """
    Split the signal into short frames (125 ms by default, the "fast"
    response of a sound level meter) and return the level of each frame.

    Frames overlap by 50% so a brief burst between two frame boundaries is
    not missed.
    """
    x = np.asarray(x, dtype=np.float64)
    frame_len = max(int(round(frame_seconds * fs)), 1)
    hop = max(frame_len // 2, 1)
    # Number of complete frames that fit; always at least one.
    n_frames = max(1, 1 + (len(x) - frame_len) // hop)
    levels = np.empty(n_frames)
    for i in range(n_frames):
        start = i * hop
        levels[i] = to_db(rms(x[start:start + frame_len]), ref)
    return np.maximum(levels, MIN_LEVEL)


def lmax(x, fs=FS, frame_seconds=0.125, ref=REF):
    """Lmax = the loudest short-term level in the buffer (125 ms frames)."""
    return float(np.max(short_term_levels(x, fs, frame_seconds, ref)))


# ---------------------------------------------------------------------------
# 2. A-weighting filter (IEC 61672-1)
# ---------------------------------------------------------------------------
# IEC 61672 defines A-weighting by an analogue transfer function with:
#     zeros: 4x at 0 Hz
#     poles: 20.6 Hz (f1, DOUBLE), 107.7 Hz (f2), 737.9 Hz (f3),
#            12194 Hz (f4, DOUBLE)   ->  4 zeros, 6 poles
#
# H(s) =        s^2 * (f1*f4)^2
#        -----------------------------------
#        (s+f1)^2 (s+f2) (s+f3) (s+f4)^2
#
# We convert this analogue filter into a DIGITAL filter with the bilinear
# transform.
F1, F2, F3, F4 = 20.598997, 107.65265, 737.86223, 12194.217
F_REF = 1000.0                  # the standard's 0 dB reference frequency


def a_weighting_analog(warp=1.0):
    """
    Analogue A-weighting as (zeros, poles, gain) in rad/s.

    `warp` multiplies every corner frequency. It implements 1 kHz
    pre-warping: this scipy version's bilinear() does NOT pre-warp (verified
    by experiment), so we scale the analogue frequencies ourselves.
    """
    f1, f2, f3, f4 = (warp * f for f in (F1, F2, F3, F4))
    zeros = np.zeros(4, dtype=complex)                    # 4 zeros at s = 0
    poles = -2.0 * np.pi * np.array([f1, f1, f2, f3, f4, f4], dtype=complex)
    # Numerator scale of the expression above: s^2 * (f1*f4)^2
    gain = (2 * np.pi * f4) ** 2 * (2 * np.pi * f1) ** 2
    return zeros, poles, gain


def a_weighting_sos(fs=FS):
    """
    Digital A-weighting filter as second-order sections (SOS) for `fs`.

    Steps:
      1. Pre-warp: choose the analogue scale so that the analogue 1 kHz point
         lands exactly on 1 kHz in the digital domain.
      2. Expand poles/zeros into polynomial coefficients b, a (scipy zpk2tf).
      3. Bilinear transform (scipy bilinear) -> digital filter bz, az.
      4. Normalise the gain so that |H(1 kHz)| = 1 (0 dB at the reference).
    """
    # --- step 1: pre-warping -------------------------------------------------
    # The bilinear transform maps analogue f_a to digital f_d:
    #     f_d = (fs/pi) * atan(pi * f_a / fs)
    # Solving it for the analogue frequency that maps onto F_REF gives factor k.
    k = (fs / np.pi) * np.tan(np.pi * F_REF / fs) / F_REF

    # --- step 2: analogue filter (pre-warped) --------------------------------
    zeros, poles, gain = a_weighting_analog(warp=k)
    b, a = signal.zpk2tf(zeros, poles, gain)

    # --- step 3: bilinear transform -----------------------------------------
    bz, az = bilinear(b, a, fs=fs)

    # --- step 4: normalise to 0 dB at 1 kHz ---------------------------------
    _, h = signal.freqz(bz, az, worN=[2 * np.pi * F_REF / fs])
    bz = bz / abs(h[0])

    # Second-order sections are numerically stable for a 6th-order filter.
    return signal.tf2sos(bz, az)


def a_weight(x, fs=FS, zero_phase=False):
    """
    Apply the A-weighting filter and return the filtered signal.

    zero_phase=False (default) runs the filter ONCE -> the result is the true
    IEC A-weighting (-26.3 dB at 63 Hz), which is what a sound level meter
    shows. No phase delay matters for a level measurement.

    zero_phase=True runs it forward AND backward (sosfiltfilt). That removes
    phase delay but squares the magnitude, i.e. DOUBLES the weighting in dB
    (-52.8 dB at 63 Hz). Only useful for visual inspection, never for levels.
    """
    sos = a_weighting_sos(fs)
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 16:                       # too short for filtfilt's padding
        return sosfilt(sos, x)
    return sosfiltfilt(sos, x) if zero_phase else sosfilt(sos, x)


def a_weight_spectrum(x, fs=FS):
    """
    A-weighting in the FREQUENCY domain: FFT -> multiply by |H_A(f)| -> FFT^-1.

    This weights every frequency exactly once (no double filtering), so the
    attenuation at 63 Hz is the standard's ~26 dB while keeping zero phase.
    """
    x = np.asarray(x, dtype=np.float64)
    # Ask for the response exactly at the rfft bin frequencies (0 ... fs/2),
    # so one gain value per bin and the arrays can be multiplied directly.
    bin_hz = np.fft.rfftfreq(len(x), d=1.0 / fs)
    _, h = sosfreqz(a_weighting_sos(fs), worN=bin_hz, fs=fs)
    return np.fft.irfft(np.fft.rfft(x) * np.abs(h), n=len(x))


def a_weighting_db(freqs):
    """
    Theoretical A-weighting in dB at the given frequencies (IEC 61672 formula):

        R_A(f) = (12194^2 f^4) /
                 ((f^2+20.6^2) sqrt((f^2+107.7^2)(f^2+737.9^2)) (f^2+12194^2))
        A(f)   = 20 log10(R_A(f)) + 2.0        (+2 because R_A(1k) = -2 dB)

    The self-test compares our digital filter against this curve.
    """
    f = np.asarray(freqs, dtype=np.float64)
    f2 = f ** 2
    ra = (F4 ** 2 * f ** 4) / ((f2 + F1 ** 2) *
                               np.sqrt((f2 + F2 ** 2) * (f2 + F3 ** 2)) *
                               (f2 + F4 ** 2))
    return 20 * np.log10(np.maximum(ra, 1e-30)) + 2.0


def a_weighted_leq(x, fs=FS, ref=REF):
    """Leq after A-weighting - i.e. dB(A), which follows human perception."""
    return leq(a_weight_spectrum(x, fs), ref)


# ---------------------------------------------------------------------------
# 3. Octave-band analysis
# ---------------------------------------------------------------------------
def octave_band_edges(center, fs=FS):
    """
    Edges of a 1/1-octave band: centre/sqrt(2) and centre*sqrt(2).
    The 8 kHz band would reach 11.3 kHz; that is allowed at 16 kHz, but we cap
    it at 0.98 * Nyquist so the filter stays strictly below Nyquist.
    """
    lo = center / np.sqrt(2.0)
    hi = center * np.sqrt(2.0)
    hi = min(hi, 0.98 * (fs / 2.0))
    lo = max(lo, 1.0)
    return lo, hi


def octave_band_levels(x, fs=FS, bands=OCTAVE_BANDS, order=4, ref=REF):
    """
    Octave-band levels in dB.

    For each band we build a Butterworth BAND-PASS filter (order 4), apply it
    (forward+backward -> zero phase) and take the Leq of the result.
    This shows WHICH frequency region carries the energy, e.g. a generator is
    strong at 63-125 Hz while an alarm sits at 2-4 kHz.
    """
    x = np.asarray(x, dtype=np.float64)
    out = {}
    for c in bands:
        lo, hi = octave_band_edges(c, fs)
        if hi <= lo:
            out[c] = MIN_LEVEL
            continue
        sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
        filtered = sosfiltfilt(sos, x) if len(x) > 32 else sosfilt(sos, x)
        out[c] = float(max(leq(filtered, ref), MIN_LEVEL))
    return out


def broadband_metrics(x, fs=FS, ref=REF):
    """
    One dict with everything the demo/dashboard needs for a chunk of audio.
    Values are relative dB (re full scale) - see the note at the top.
    """
    x = np.asarray(x, dtype=np.float64)
    return {
        "leq": leq(x, ref),                        # unweighted broadband
        "leq_a": a_weighted_leq(x, fs, ref),       # dB(A), perception weighted
        "lmax": lmax(x, fs, ref=ref),              # loudest 125 ms frame
        "peak_db": to_db(np.max(np.abs(x)) if len(x) else 0.0, ref),
        "bands": octave_band_levels(x, fs, ref=ref),
    }


# ---------------------------------------------------------------------------
# 4. Self-test:  python auralium/levels.py
# ---------------------------------------------------------------------------
def _tone(freq, seconds=1.0, fs=FS, amplitude=0.5):
    """Generate a pure sine wave."""
    t = np.arange(int(seconds * fs)) / fs
    x = amplitude * np.sin(2 * np.pi * freq * t)
    # Short fade-in/out: avoids clicks at the start/end, which would spray
    # broadband energy into the filters and disturb the level measurements.
    n_fade = max(int(0.05 * fs), 1)
    if len(x) > 2 * n_fade:
        env = np.ones(len(x))
        env[:n_fade] = np.linspace(0.0, 1.0, n_fade)
        env[-n_fade:] = np.linspace(1.0, 0.0, n_fade)
        x = x * env
    return x


def self_test():
    ok = True

    def check(name, condition, detail=""):
        nonlocal ok
        status = "PASS" if condition else "FAIL"
        ok = ok and condition
        print(f"  [{status}] {name}  {detail}")

    print("1) Pure 1 kHz tone at amplitude 0.5 (RMS = 0.5/sqrt(2) = 0.3536)")
    x = _tone(1000, amplitude=0.5)
    expected_leq = 20 * np.log10(0.5 / np.sqrt(2))       # = -9.03 dB
    # Measure the steady middle part: the fades lower the whole-buffer RMS.
    x_mid = x[int(0.05 * FS):-int(0.05 * FS)]
    got = leq(x_mid)
    check("Leq matches 20*log10(RMS) within 0.1 dB",
          abs(got - expected_leq) < 0.1,
          f"expected {expected_leq:.2f} dB, got {got:.2f} dB")

    got_a = a_weighted_leq(x_mid)
    check("A-weighted Leq at 1 kHz ~= unweighted (0 dB gain at 1 kHz)",
          abs(got_a - got) < 0.1, f"{got_a:.2f} dB(A) vs {got:.2f} dB")

    print("\n2) A-weighting shape vs the IEC 61672 formula")
    f_test = [63, 125, 250, 500, 1000, 2000, 4000]
    sos = a_weighting_sos(FS)
    _, h = sosfreqz(sos, worN=f_test, fs=FS)
    filter_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
    theory_db = a_weighting_db(f_test)
    for f, fd, td in zip(f_test, filter_db, theory_db):
        print(f"      {f:>5d} Hz : filter {float(fd):7.2f} dB | IEC {float(td):7.2f} dB")
    max_diff = float(np.max(np.abs(filter_db - theory_db)))
    check("digital filter within 0.6 dB of the IEC curve, 63 Hz - 4 kHz",
          max_diff < 0.6, f"max diff {max_diff:.2f} dB")

    print("\n3) A-weighting attenuates a 63 Hz tone far more than a 1 kHz tone")
    x63 = _tone(63, seconds=2.0, amplitude=0.5)
    x63_mid = x63[int(0.05 * FS):-int(0.05 * FS)]
    flat63, filt63 = leq(x63_mid), a_weighted_leq(x63_mid)
    atten63 = flat63 - filt63            # positive = attenuation in dB
    atten1k = got - got_a
    print(f"      63 Hz : unweighted {flat63:.2f} dB -> weighted {filt63:.2f} dB "
          f"({atten63:.2f} dB attenuation)")
    # Same check with the zero-phase frequency-domain weighting, for detail.
    filt63_spec = leq(a_weight_spectrum(x63_mid, FS))
    print(f"              frequency-domain A-weighting: {filt63_spec:.2f} dB "
          f"({flat63 - filt63_spec:.2f} dB attenuation)")
    print(f"      1 kHz : unweighted {got:.2f} dB -> weighted {got_a:.2f} dB "
          f"({atten1k:.2f} dB attenuation)")
    check("63 Hz attenuated more than 1 kHz by at least 10 dB",
          atten63 > atten1k + 10, f"delta = {atten63 - atten1k:.1f} dB")
    check("63 Hz attenuation is near the standard ~26 dB",
          20 < atten63 < 35, f"{atten63:.1f} dB")

    print("\n4) Octave bands localise energy correctly")
    x_low = _tone(100, seconds=2.0, amplitude=0.3)       # inside the 125 Hz band
    x_high = _tone(3000, seconds=2.0, amplitude=0.3)     # between 2k and 4k
    b_low = octave_band_levels(x_low)
    b_high = octave_band_levels(x_high)
    loud_low = max(b_low, key=b_low.get)
    loud_high = max(b_high, key=b_high.get)
    print("       100 Hz tone band levels:", {k: round(float(v), 1) for k, v in b_low.items()})
    print("      3000 Hz tone band levels:", {k: round(float(v), 1) for k, v in b_high.items()})
    check("100 Hz tone peaks in the 125 Hz band", loud_low == 125, f"peak {loud_low} Hz")
    check("3 kHz tone peaks in the 2k or 4k band", loud_high in (2000, 4000),
          f"peak {loud_high} Hz")
    check("8 kHz band upper edge is below Nyquist",
          octave_band_edges(8000)[1] < FS / 2,
          f"edges {octave_band_edges(8000)[0]:.0f}-{octave_band_edges(8000)[1]:.0f} Hz")

    print("\n5) Lmax is the loudest 125 ms frame and never below the Leq")
    # quiet 1 s, a loud 0.25 s burst, then quiet again
    burst = np.concatenate([_tone(500, 1.0, amplitude=0.02),
                            _tone(500, 0.25, amplitude=0.8),
                            _tone(500, 1.0, amplitude=0.02)])
    lmax_v, leq_v = lmax(burst), leq(burst)
    # The steady part of the burst, used as the expected Lmax.
    burst_steady = burst[int(1.05 * FS):int(1.20 * FS)]
    print(f"      Leq {leq_v:.1f} dB | Lmax {lmax_v:.1f} dB "
          f"(burst level {leq(burst_steady):.1f} dB)")
    check("Lmax >= Leq and Lmax is within 1.5 dB of the burst level",
          lmax_v >= leq_v and abs(lmax_v - leq(burst_steady)) < 1.5)

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_test() else 1)
