"""
demo.py
=======
The live Auralium demo: listen for 1 second -> measure -> classify -> log.

Two modes:
    python auralium/demo.py                       # default microphone
    python auralium/demo.py --file audio/x.wav    # play a WAV file through the
                                                  # same pipeline (no mic needed)
    --threshold -20                               # alert threshold in dB (relative)

For every 1-second chunk it prints ONE line:
    timestamp  label  confidence  Leq(dB)  Lmax(dB)  [ALERT]

PRIVACY ("raw audio never leaves the device"):
    The microphone audio only ever exists in the `chunk` variable in RAM. It is
    processed into a level + a class label and then dropped. Nothing calls
    soundfile.write / scipy.io.wavfile.write here, so no recording is created.
    ONLY the numeric results are written to logs/events.csv. This is what lets
    Auralium claim that patient conversations are never stored or transmitted -
    the classifier runs entirely on the device (laptop now, microcontroller
    later) and only a class label + a number leave the microphone routine.
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import SAMPLE_RATE, extract_mfcc_features, load_clip  # noqa: E402
from levels import broadband_metrics, leq, lmax  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "classifier.joblib")
LOGS_DIR = os.path.join(HERE, "logs")
LOG_PATH = os.path.join(LOGS_DIR, "events.csv")

FS = SAMPLE_RATE          # 16 kHz, same as training
CHUNK_SECONDS = 1.0       # one decision per second
DEFAULT_THRESHOLD = -25.0 # relative dB re full scale; see levels.py note

CSV_COLUMNS = ["timestamp", "source", "label", "confidence", "leq_db",
               "lmax_db", "threshold_db", "alert"]


# ---------------------------------------------------------------------------
# Model loading and one-chunk processing (shared with app.py)
# ---------------------------------------------------------------------------
def load_classifier(path=MODEL_PATH):
    """Load the trained model bundle saved by train.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No model at {path}. Run: python auralium/train.py")
    return joblib.load(path)


def classify_chunk(y, artifact, fs=FS):
    """
    Classify ONE chunk of audio.

    The model was trained on 5-second clips, so a 1-second live chunk is
    "looped" to 5 seconds before feature extraction. This keeps the feature
    statistics (means/stds over time) on the same footing as during training.
    Returns (label, confidence).
    """
    clip_len = 5 * fs
    if len(y) < clip_len:
        reps = int(np.ceil(clip_len / max(len(y), 1)))
        y5 = np.tile(y, reps)[:clip_len]
    else:
        y5 = y[:clip_len]

    feats = extract_mfcc_features(y5, fs).reshape(1, -1)
    model, classes = artifact["model"], artifact["classes"]
    probs = model.predict_proba(feats)[0]
    idx = int(np.argmax(probs))
    return classes[idx], float(probs[idx])


def process_chunk(y, artifact, fs=FS):
    """
    Everything the demo and the dashboard need for one chunk of audio:
    class, confidence, Leq, Lmax, octave bands.
    """
    label, confidence = classify_chunk(y, artifact, fs)
    metrics = broadband_metrics(y, fs)
    return {
        "label": label,
        "confidence": confidence,
        "leq_db": metrics["leq"],
        "leq_a_db": metrics["leq_a"],
        "lmax_db": metrics["lmax"],
        "bands": metrics["bands"],
    }


# ---------------------------------------------------------------------------
# Logging: numbers only, never audio
# ---------------------------------------------------------------------------
def init_log(path=LOG_PATH):
    """Create logs/events.csv with a header if it does not exist yet."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(CSV_COLUMNS)


def log_event(result, threshold, source="mic", path=LOG_PATH):
    """
    Append ONE row to the CSV: timestamp, label, confidence and metrics.
    Deliberately no audio, no filename of a recording, no free text.
    """
    init_log(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow([
            datetime.now().isoformat(timespec="seconds"),
            source,
            result["label"],
            f"{result['confidence']:.3f}",
            f"{result['leq_db']:.1f}",
            f"{result['lmax_db']:.1f}",
            f"{threshold:.1f}",
            "ALERT" if result["leq_db"] > threshold else "",
        ])


def read_log(path=LOG_PATH):
    """Recent events for the dashboard. Returns [] if no log exists yet."""
    import pandas as pd
    if not os.path.exists(path):
        return pd.DataFrame(columns=CSV_COLUMNS)
    return pd.read_csv(path)


def format_line(result, threshold):
    """Build the one-line console output."""
    alert = "  ALERT" if result["leq_db"] > threshold else ""
    return (f"{datetime.now():%H:%M:%S}  {result['label']:<16s} "
            f"conf={result['confidence']:.2f}  "
            f"Leq={result['leq_db']:6.1f} dB  "
            f"Lmax={result['lmax_db']:6.1f} dB{alert}")


# ---------------------------------------------------------------------------
# Mode 1: microphone (live)
# ---------------------------------------------------------------------------
def run_microphone(artifact, threshold, max_chunks=0, device=None):
    """
    Record 1-second chunks from the default microphone.

    The audio buffer `chunk` is the ONLY place the samples live; it is
    overwritten on every iteration and never written to disk.
    """
    try:
        import sounddevice as sd
    except OSError as exc:                    # PortAudio not available
        print(f"sounddevice could not open PortAudio: {exc}")
        print("Use --file <wav> to demo without a microphone.")
        return 1

    print(f"Listening at {FS} Hz in {CHUNK_SECONDS:.0f}s chunks. "
          f"Alert above {threshold:.1f} dB (relative). Ctrl+C to stop.")
    print("Raw audio is processed in memory only - nothing is recorded.")

    n = 0
    try:
        while True:
            # blocking read of exactly 1 s of mono float32 audio
            chunk = sd.rec(int(CHUNK_SECONDS * FS), samplerate=FS,
                           channels=1, dtype="float32", device=device)
            sd.wait()
            chunk = chunk[:, 0]

            result = process_chunk(chunk, artifact)
            print(format_line(result, threshold))
            log_event(result, threshold, source="mic")

            n += 1
            if max_chunks and n >= max_chunks:
                break
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as exc:                  # no input device, permissions...
        print(f"Microphone error: {exc}")
        print("Tip: use --file <wav> to run the same pipeline without a mic.")
        return 1

    print(f"Logged {n} event(s) to {LOG_PATH}")
    return 0


# ---------------------------------------------------------------------------
# Mode 2: WAV file (no microphone needed)
# ---------------------------------------------------------------------------
def run_file(path, artifact, threshold, simulate=False, max_chunks=0):
    """
    Run the identical pipeline on a WAV file, 1 second at a time.

    simulate=True plays the file in real time (1 s per 1 s). The default is
    faster-than-real-time so the terminal output is quick to screenshot.
    """
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return 1

    y, _ = load_clip(path)                    # 16 kHz mono
    print(f"Analysing {os.path.basename(path)} "
          f"({len(y) / FS:.1f}s) above {threshold:.1f} dB (relative)")

    step = int(CHUNK_SECONDS * FS)
    n = 0
    for start in range(0, len(y), step):
        chunk = y[start:start + step]
        if len(chunk) < step // 4:            # ignore an tiny trailing stub
            break
        if simulate:
            time.sleep(CHUNK_SECONDS)

        result = process_chunk(chunk, artifact)
        print(format_line(result, threshold))
        log_event(result, threshold, source="file")
        n += 1
        if max_chunks and n >= max_chunks:
            break

    print(f"Processed {n} chunk(s); results in {LOG_PATH} (no audio saved).")
    return 0


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Auralium live demo: classify + measure 1 s of audio.")
    parser.add_argument("--file", help="analyse a WAV file instead of the mic")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"alert threshold in relative dB "
                             f"(default {DEFAULT_THRESHOLD})")
    parser.add_argument("--simulate", action="store_true",
                        help="with --file: wait 1 s per chunk (real-time feel)")
    parser.add_argument("--max-chunks", type=int, default=0,
                        help="stop after N chunks (0 = run until stopped)")
    args = parser.parse_args(argv)

    artifact = load_classifier()
    print(f"Model: {artifact['model_name']}  "
          f"(CV accuracy {artifact['cv_mean_accuracy']:.3f})")

    if args.file:
        return run_file(args.file, artifact, args.threshold,
                        simulate=args.simulate, max_chunks=args.max_chunks)
    return run_microphone(artifact, args.threshold, max_chunks=args.max_chunks)


if __name__ == "__main__":
    sys.exit(main())
