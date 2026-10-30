"""
app.py
======
The Auralium dashboard (Streamlit).

Two tabs, both built on the SAME functions as demo.py (process_chunk,
log_event), so the demo and the dashboard can never drift apart:

    Live        - reads 1 s from the microphone on each refresh, shows the
                  current Leq, the octave-band chart, the predicted class and
                  the recent alerts logged to logs/events.csv.
    Upload WAV  - runs the same pipeline over an uploaded file.

Run:
    streamlit run auralium/app.py
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import demo  # noqa: E402  (same pipeline: process_chunk / log_event / read_log)
from levels import OCTAVE_BANDS  # noqa: E402

FS = demo.FS

st.set_page_config(page_title="Auralium | hospital noise monitor",
                   page_icon="🔈", layout="wide")


# ---------------------------------------------------------------------------
# Cached model loading (Streamlit reruns the script on every interaction)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model():
    return demo.load_classifier()


def record_seconds(seconds, device=None):
    """Record from the default microphone. The audio stays in memory only."""
    import sounddevice as sd
    chunk = sd.rec(int(seconds * FS), samplerate=FS, channels=1,
                   dtype="float32", device=device)
    sd.wait()
    return chunk[:, 0]


# ---------------------------------------------------------------------------
# Shared widgets
# ---------------------------------------------------------------------------
def show_result(result, threshold):
    """Metrics + octave-band chart + class, for one analysed chunk."""
    c1, c2, c3 = st.columns([1, 1, 1.4])
    alert = result["leq_db"] > threshold
    c1.metric("Leq (broadband)", f"{result['leq_db']:.1f} dB",
              delta="above threshold" if alert else "within limit",
              delta_color="inverse" if alert else "normal")
    c2.metric("Lmax (125 ms)", f"{result['lmax_db']:.1f} dB")
    c3.metric("Leq (A-weighted)", f"{result['leq_a_db']:.1f} dB(A)")

    # The class gets its own line: inside a narrow metric column Streamlit
    # truncates long values like "equipment_alarm" to "equipmen...".
    pretty = result["label"].replace("_", " ")
    st.markdown(f"### Predicted class: **{pretty}** "
                f"<span style='font-size:0.8em;color:gray'>"
                f"confidence {result['confidence']:.0%}</span>",
                unsafe_allow_html=True)

    if alert:
        st.error(f"ALERT: Leq {result['leq_db']:.1f} dB exceeds the "
                 f"{threshold:.1f} dB threshold.")

    # Octave-band bar chart. Keys are ints (63 ... 8000) -> label as strings.
    band_df = pd.DataFrame({
        "band_Hz": [str(k) for k in result["bands"].keys()],
        "level_dB": [round(float(v), 1) for v in result["bands"].values()],
    }).set_index("band_Hz")
    st.subheader("Octave-band levels")
    # Plotly rather than st.bar_chart: the default chart orders x labels by
    # value ("1000, 125, 2000...") and cuts off the axis. Here the bands stay in
    # ascending frequency order and every label is readable.
    try:
        import plotly.express as px
    except ImportError:                 # plotly is optional in this prototype
        st.bar_chart(band_df, height=280, color="#1f6feb")
    else:
        fig = px.bar(band_df.reset_index(), x="band_Hz", y="level_dB",
                     labels={"band_Hz": "Octave band (Hz)",
                             "level_dB": "Level (dB)"},
                     color_discrete_sequence=["#1f6feb"])
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis=dict(range=[-110, 0]))
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Relative dB re full scale (uncalibrated microphone) - "
               "compare bands with each other, not with dB SPL values.")


def show_alerts(log_df, limit=10):
    """The most recent threshold crossings from logs/events.csv."""
    st.subheader("Recent alerts")
    if log_df.empty:
        st.info("No events logged yet.")
        return
    alerts = log_df[log_df["alert"] == "ALERT"].tail(limit).iloc[::-1]
    if alerts.empty:
        st.success("No alerts logged so far.")
    else:
        show = alerts[["timestamp", "source", "label", "confidence",
                       "leq_db", "lmax_db"]]
        st.dataframe(show, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def live_tab(artifact):
    st.caption("Microphone audio is analysed in memory and discarded. Only "
               "labels and numbers are written to logs/events.csv.")

    threshold = st.sidebar.slider("Alert threshold (relative dB)",
                                  min_value=-60, max_value=0,
                                  value=int(demo.DEFAULT_THRESHOLD), step=1)
    live = st.toggle("Live capture (records 1 s on every refresh)", value=False)
    if live:
        st.button("Refresh now")

    if live:
        try:
            chunk = record_seconds(1.0)
            result = demo.process_chunk(chunk, artifact)
            demo.log_event(result, threshold, source="dashboard")
        except Exception as exc:                       # no mic / permissions
            st.warning(f"Microphone unavailable: {exc}")
            st.info("Use the 'Upload WAV' tab to demo without a microphone.")
            show_alerts(demo.read_log())
            return
        show_result(result, threshold)
    else:
        st.info("Turn on live capture, or use the factory demo below.")

    # A "factory demo" button: play a real ESC-50 clip through the pipeline,
    # so the dashboard can be shown on a laptop without a microphone.
    st.divider()
    st.subheader("Demo clip")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Short, curated demo list with one clip per class so the dropdown opens
    # instantly and the presentation never has to scroll through 200 files.
    demo_files = {
        "siren":          "1-31482-A-42.wav",
        "horn":           "1-17124-A-43.wav",
        "generator":      "1-18527-A-44.wav",
        "construction":   "1-116765-A-41.wav",
        "equipment_alarm": "1-13613-A-37.wav",
        "other (dog)":    "1-100032-A-0.wav",
    }
    demo_files = {k: v for k, v in demo_files.items()
                  if os.path.exists(os.path.join(root, "audio", v))}
    choice = st.selectbox("ESC-50 clip", list(demo_files), index=0)
    choice = demo_files[choice]
    second = st.slider("Second of the clip to analyse", 1, 5, 3)
    if st.button("Analyse clip"):
        from features import load_clip
        y, _ = load_clip(os.path.join(root, "audio", choice))
        # Analyse the chosen second, not the start: many ESC-50 clips begin
        # with silence, which would show a -120 dB floor and a random label.
        start = (second - 1) * FS
        result = demo.process_chunk(y[start:start + FS], artifact)
        demo.log_event(result, threshold, source="clip")
        show_result(result, threshold)

    st.divider()
    show_alerts(demo.read_log())


def upload_tab(artifact):
    st.caption("Uploaded audio is analysed in memory only; it is not saved "
               "by the app.")
    threshold = st.number_input("Alert threshold (relative dB)",
                                value=float(demo.DEFAULT_THRESHOLD), step=1.0)
    up = st.file_uploader("WAV file", type=["wav"])
    if up is None:
        return

    # librosa reads straight from the uploaded bytes - nothing hits the disk.
    import librosa
    y, _ = librosa.load(up, sr=FS, mono=True)
    st.write(f"{len(y) / FS:.1f} s of audio at {FS} Hz")

    step = int(FS * 1.0)
    rows = []
    for start in range(0, len(y), step):
        chunk = y[start:start + step]
        if len(chunk) < step // 4:
            break
        r = demo.process_chunk(chunk, artifact)
        rows.append({"second": start // step + 1,
                     "label": r["label"],
                     "confidence": round(r["confidence"], 2),
                     "leq_db": round(r["leq_db"], 1),
                     "lmax_db": round(r["lmax_db"], 1),
                     "alert": r["leq_db"] > threshold})
        last = r

    if not rows:
        st.warning("No full second of audio found.")
        return

    df = pd.DataFrame(rows)
    st.subheader("Whole-file summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean Leq", f"{df['leq_db'].mean():.1f} dB")
    c2.metric("Peak Lmax", f"{df['lmax_db'].max():.1f} dB")
    c3.metric("Most frequent label", df["label"].mode().iloc[0])
    st.line_chart(df.set_index("second")[["leq_db", "lmax_db"]])
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Last second of the file")
    show_result(last, threshold)


def main():
    st.title("🔈 Auralium")
    st.caption("Hospital noise monitoring - class + level, on-device. "
               "All dB values are RELATIVE (uncalibrated microphone).")

    artifact = get_model()
    st.sidebar.header("Model")
    st.sidebar.write(f"**{artifact['model_name']}**")
    st.sidebar.write(f"5-fold CV accuracy: {artifact['cv_mean_accuracy']:.3f}")
    st.sidebar.write("Auralium prototype - ESC-50 subset")

    tab_live, tab_upload = st.tabs(["Live", "Upload WAV"])
    with tab_live:
        live_tab(artifact)
    with tab_upload:
        upload_tab(artifact)


main()
