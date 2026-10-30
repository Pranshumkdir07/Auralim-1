# Auralium - noise monitoring for hospitals (prototype)

Auralium listens to a hospital room, **classifies** the sound (siren, horn,
generator, construction, equipment alarm, or "nothing relevant") and
**measures** how loud it is (Leq, Lmax, octave bands), then raises an alert if
the level crosses a threshold. Everything runs **on the device**: raw audio is
never written to disk or transmitted - only a class label, a confidence and a
few numbers are logged.

This is a course prototype for an Innovation and Entrepreneurship project. It
proves the signal chain works end to end on a laptop; it is not a certified
sound level meter.

---

## 1. Setup

From the project root (the folder containing `audio/` and `meta/`):

```powershell
python -m venv auralium\.venv
auralium\.venv\Scripts\python.exe -m pip install -r auralium\requirements.txt
```

All commands below assume the venv's Python is used explicitly.

## 2. How to run each script

| Goal | Command |
|---|---|
| Train + evaluate the classifier (writes `results/` and `models/`) | `auralium\.venv\Scripts\python.exe auralium\train.py` |
| Run the sound-level self-test (no audio needed) | `auralium\.venv\Scripts\python.exe auralium\levels.py` |
| Live demo from the microphone | `auralium\.venv\Scripts\python.exe auralium\demo.py` |
| Demo without a microphone, on a WAV file | `auralium\.venv\Scripts\python.exe auralium\demo.py --file audio\1-31482-A-42.wav` |
| Demo with a custom alert threshold (relative dB) | `auralium\.venv\Scripts\python.exe auralium\demo.py --threshold -20` |
| Real-time feel on a file (1 s per chunk) | `auralium\.venv\Scripts\python.exe auralium\demo.py --file audio\x.wav --simulate` |
| Dashboard | `auralium\.venv\Scripts\python.exe -m streamlit run auralium\app.py` |

`train.py` must be run once before `demo.py` or `app.py`, because they load
`models/classifier.joblib`.

## 3. What each file does

| File | Role |
|---|---|
| `auralium/features.py` | Turns a clip into **one 162-number vector**: 40 MFCCs + 40 deltas, each summarised by mean and std (160), plus spectral centroid and zero-crossing rate (2). All audio at 16 kHz mono. |
| `auralium/train.py` | Maps ESC-50 categories onto the 6 Auralium classes, balances them, runs 5-fold CV on the dataset's `fold` column for RandomForest vs SVM, saves the confusion matrix, the comparison table and the final model. |
| `auralium/levels.py` | Sound-level maths from scratch: RMS -> **Leq**, 125 ms frames -> **Lmax**, **IEC 61672 A-weighting** (analogue poles/zeros, manual 1 kHz pre-warp, bilinear transform), and **octave bands** 63 Hz - 8 kHz with Butterworth band-passes. Includes a self-test. |
| `auralium/demo.py` | The live loop: 1 s chunks from the microphone (or a WAV file), measures + classifies, prints one line per chunk, appends metrics only to `logs/events.csv`. |
| `auralium/app.py` | Streamlit dashboard: current Leq/Lmax/class, octave-band chart, recent alerts, plus an "Upload WAV" tab. Reuses `demo.process_chunk` so it cannot drift from the CLI. |
| `auralium/requirements.txt` | Python dependencies. |
| `auralium/results/` | `confusion_matrix.png`, `confusion_matrix_counts.png`, `model_comparison.csv`, `one_second_evaluation.csv`, `classification_report.txt`, `features.csv` (cached features). |
| `auralium/models/classifier.joblib` | The trained model + its class names + metadata + scaler (for the SVM). |
| `auralium/logs/events.csv` | Timestamp, source, label, confidence, Leq, Lmax, threshold, alert flag. **No audio.** |

### Class mapping

| ESC-50 categories | Auralium class |
|---|---|
| `siren` | `siren` |
| `car_horn` | `horn` |
| `engine` | `generator` |
| `chainsaw`, `hand_saw` | `construction` |
| `clock_alarm` | `equipment_alarm` |
| `rain`, `wind`, `keyboard_typing`, `footsteps` | `other` |

Balanced at 40 clips per class (240 clips in total); 40 of the 80 "other"
candidates are sampled. 

## 4. Results 

### 5-fold cross-validation (headline accuracy)

Folds come from `meta/esc50.csv`, so clips from the same source recording never
appear in both train and test. Train on 4 folds, test on 1, rotate.

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **Mean** | Std |
|---|---|
| **RandomForest** (saved) | 0.654 | 0.660 | 0.739 | 0.780 | 0.822 | **0.731** | 0.066 |
| SVM (RBF + StandardScaler) | 0.692 | 0.575 | 0.696 | 0.740 | 0.733 | 0.687 | 0.060 |

Per-class performance of the saved RandomForest (from the out-of-fold
predictions behind `confusion_matrix.png`):

| Class | Precision | Recall | F1 |
|---|---|
| siren | 0.88 | 0.88 | 0.88 |
| horn | 0.81 | 0.72 | 0.76 |
| generator | 0.61 | 0.62 | 0.62 |
| construction | 0.73 | 0.82 | 0.78 |
| equipment_alarm | 0.91 | 0.75 | 0.82 |
| other | 0.51 | 0.57 | 0.54 |

The generator/other confusion (0.30 of `other` clips predicted as generator) is
the weakest point: moving engines and background rumble share the same low-
frequency energy.

### The live-demo condition (1-second chunks)

The live loop decides on **1-second** chunks, but the model is trained on 5-second
clips, so `train.py` also scores every clip split into 5 x 1 s chunks with the
same fold models:

| Condition | Accuracy |
|---|---|
| 5 s clips, fold CV (headline) | **0.731** |
| 1 s chunks, same fold models | **0.896** |

Shorter chunks actually score higher. That is plausible: an 80,000-sample clip
averages over a lot of background, while 1 s captures one clean event. It is
encouraging for the demo but should be read with care - see the limitations.

*(An additional "leave-one-out" diagnostic in `train.py`, where the model trains
on the other four seconds of the same recording, currently returns 0.17 and is
unexplained. It is kept only as a diagnostic, labelled as such in the code, and
is not used for any reported result.)*

### Sound-level self-test (`levels.py`)

```
[PASS] Leq matches 20*log10(RMS) within 0.1 dB          (expected -9.03, got -9.03)
[PASS] A-weighted Leq at 1 kHz ~= unweighted            (-9.03 dB(A))
[PASS] digital filter within 0.6 dB of the IEC curve    (max diff 0.42 dB, 63 Hz-4 kHz)
[PASS] 63 Hz attenuated more than 1 kHz by at least 10 dB
[PASS] 63 Hz attenuation is near the standard ~26 dB    (25.7 dB)
[PASS] 100 Hz tone peaks in the 125 Hz band
[PASS] 3 kHz tone peaks in the 2k or 4k band
[PASS] 8 kHz band upper edge is below Nyquist           (5657-7840 Hz)
[PASS] Lmax >= Leq and Lmax within 1.5 dB of burst level
ALL CHECKS PASSED
```

Two bugs were found and fixed while getting there, both worth a slide:

1. **No pre-warping.** `scipy.signal.bilinear` does not pre-warp (verified by
   experiment), which put every corner frequency a factor of ~2.2 too low. The
   fix is to scale the analogue frequencies by
   `k = (fs/pi) * tan(pi * 1000 / fs) / 1000` before transforming.
2. **Double counting.** Running the filter with `sosfiltfilt` gives zero phase
   but squares the magnitude, i.e. **doubles** the weighting in dB (52.8 dB
   instead of 26.3 dB at 63 Hz). Levels now use a single pass, or the
   frequency-domain weighting (`a_weight_spectrum`), so 63 Hz is 25.7 dB - the
   standard value.

### Relative dB, not dB SPL

There is no calibrated measurement microphone, so absolute sound pressure level
cannot be claimed. All levels use a full-scale reference (RMS = 1.0 -> 0 dB,
like dBFS) and are **only comparable with each other**, or after adding a
calibration offset measured with a reference meter.

## 5. Privacy design

> Raw audio never leaves the device.

In `demo.py` the microphone buffer (`chunk`) exists only inside one iteration of
the loop: it is measured, classified, and dropped. Nothing writes a WAV; there is
no network code anywhere in the project. `logs/events.csv` receives only
`timestamp, source, label, confidence, leq_db, lmax_db, threshold_db, alert`.
That is what allows Auralium to monitor a room where patients talk, without ever
storing or transmitting those conversations - and it is why the classifier is a
small on-device model rather than a cloud API call.

## 6. Limitations

- **Small dataset.** 40 clips per class (240 clips). Per-class accuracy therefore
  swings by ~2.5 percentage points per clip, which is why the fold-to-fold spread
  (0.65 to 0.82) is as large as it is.
- **Lab recordings, not hospital audio.** ESC-50 is sourced from Freesound and
  YouTube. It contains no patient rooms, no ward noise, no medical device alarms
  recorded in situ. `clock_alarm` is only a stand-in for an equipment alarm, and
  `engine` only a stand-in for a generator.
- **Uncalibrated laptop microphone.** Levels are relative dB; the mic's own
  frequency response and automatic gain control also colour the signal.
- **Laptop, not a microcontroller.** "On-device" here means a laptop. A real
  deployment would run on an ESP32 or similar, which cannot host
  librosa/scikit-learn as-is and would need a much smaller feature extractor and
  a quantised model.
- **Chunk-length mismatch.** The model sees 5 s clips in training but 1 s in
  live use; `demo.py` loops the 1 s chunk to 5 s to reduce the difference.
- **Silence.** Most ESC-50 clips start with silence; the dashboard exposes a
  "second to analyse" control for that reason.
- **Class order bug (fixed).** An early version saved the class names in a
  different order from `model.classes_`, so the demo mislabelled every clip.
  `train.py` now asserts the two match before saving.

## 7. Dataset credit

This prototype uses the **ESC-50** dataset:

> K. J. Piczak, "ESC: Dataset for Environmental Sound Classification",
> *Proceedings of the 23rd ACM International Conference on Multimedia*,
> pp. 1015-1018, Brisbane, Australia, 2015.

- Website: <https://github.com/karolpiczak/ESC-50>
- Licence: **Creative Commons Attribution-NonCommercial 3.0 (CC BY-NC 3.0)** -
  non-commercial use only, attribution required.

No files in `audio/` or `meta/` were modified or deleted.
