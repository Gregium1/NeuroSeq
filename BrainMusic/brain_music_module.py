"""
Brain Music Module
Converts (timeline, licks, stimuli) into a multi-track MIDI file.
Designed to be called from main.py via:

    music_mod.run_music(cfg)

where cfg is a SimpleNamespace created by the GUI.
"""

import os
import pandas as pd
import numpy as np

# ============================================================
# Safety helpers for MIDI
# ============================================================

def safe_note(n, lo=36, hi=96, wrap=True):
    try:
        n_int = int(round(float(n)))
    except Exception:
        n_int = lo
    if wrap:
        span = hi - lo + 1
        return lo + ((n_int - lo) % span)
    return max(0, min(127, n_int))

def safe_vel(v, lo=1, hi=127):
    try:
        v_int = int(round(float(v)))
    except Exception:
        v_int = lo
    return max(0, min(127, v_int))

def safe_ticks(t):
    try:
        return max(0, int(round(float(t))))
    except Exception:
        return 0


# ============================================================
# Column helper
# ============================================================

def _need(df, colname):
    cols = {c.lower(): c for c in df.columns}
    key = colname.lower()
    if key not in cols:
        raise ValueError(f"Missing required column '{colname}' in timeline")
    return cols[key]


# ============================================================
# Load licks / stimuli
# ============================================================

def load_lick_times(path):
    if not os.path.exists(path):
        print(f"[WARN] Lick file not found: {path}")
        return []

    lk = pd.read_csv(path)
    time_col = None

    for c in lk.columns:
        if c.strip().lower() in ("time", "t", "lick_time", "licks", "lick"):
            time_col = c
            break

    if time_col is None:
        for c in lk.columns:
            if pd.api.types.is_numeric_dtype(lk[c]):
                time_col = c
                break

    if time_col is None:
        raise ValueError(f"Couldn't find a numeric time column in {path}")

    times = pd.to_numeric(lk[time_col], errors="coerce").dropna().values.tolist()
    times = [float(t) for t in times if np.isfinite(t) and t >= 0]
    return sorted(times)


def load_stim_events(path):
    if not os.path.exists(path):
        print(f"[WARN] Stim file not found: {path}")
        return []

    df_st = pd.read_csv(path)
    tcol = None

    for c in df_st.columns:
        if c.strip().lower() in ("time", "t", "start", "onset", "stim_time",
                                 "stimulus_time", "delivery_time"):
            tcol = c
            break

    if tcol is None:
        for c in df_st.columns:
            if pd.api.types.is_numeric_dtype(df_st[c]):
                tcol = c
                break

    if tcol is None:
        raise ValueError(f"Couldn't find a numeric time column in {path}")

    # label column (optional)
    lcol = None
    for c in df_st.columns:
        if c.strip().lower() in ("label", "type", "stimulus", "name", "cond", "condition"):
            lcol = c
            break

    out = []
    times = pd.to_numeric(df_st[tcol], errors="coerce")

    for i, t in times.items():
        if pd.isna(t) or float(t) < 0:
            continue
        lbl = df_st.loc[i, lcol] if lcol and pd.notna(df_st.loc[i, lcol]) else None
        out.append((float(t), lbl))

    out.sort(key=lambda x: x[0])
    return out


# ============================================================
# Stimulus → Note Mapping
# ============================================================

def stim_note_for(lbl, default_note):
    if not lbl:
        return default_note

    s = str(lbl).lower().strip()
    if "kcl" in s:
        return 79   # G5
    if "nacl" in s:
        return 84   # C6
    if "suc" in s:
        return 76   # E5
    if "qui" in s or "quinine" in s:
        return 82   # A#5
    return default_note


# ============================================================
# Label → pitch for organ states
# ============================================================

MAJOR_STEPS = [0, 2, 4, 5, 7, 9, 11]

def label_to_index(label):
    idx = 0
    for ch in str(label):
        if ch.isalpha():
            idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return max(0, idx - 1)

def index_to_pitch(i, base=60):
    octave = i // len(MAJOR_STEPS)
    step   = i % len(MAJOR_STEPS)
    return base + 12 * octave + MAJOR_STEPS[step]

def label_to_pitch(label):
    return index_to_pitch(label_to_index(label))


# ============================================================
# Normalize GUI cfg → lowercase attrs
# ============================================================

def _normalize_music_cfg(cfg):
    def copy(lower, upper, default=None):
        if hasattr(cfg, lower):
            return
        if hasattr(cfg, upper):
            setattr(cfg, lower, getattr(cfg, upper))
        elif default is not None:
            setattr(cfg, lower, default)

    # paths
    copy("timeline_csv", "TIMELINE_CSV")
    copy("lick_csv",     "LICK_CSV")
    copy("stim_csv",     "STIM_CSV")
    copy("out_dir",      "OUT_DIR")

    # MIDI settings
    copy("bpm",              "BPM",              120)
    copy("ticks_per_beat",   "TICKS_PER_BEAT",   480)
    copy("organ_program",    "ORGAN_PROGRAM",    19)
    copy("bass_program",     "BASS_PROGRAM",     32)
    copy("stim_program",     "STIM_PROGRAM",     14)
    copy("bass_note",        "BASS_NOTE",        36)
    copy("bass_len_s",       "BASS_LEN_S",       0.08)
    copy("bass_velocity",    "BASS_VELOCITY",    95)
    copy("stim_default_note","STIM_DEFAULT_NOTE",72)
    copy("stim_len_s",       "STIM_LEN_S",       0.35)
    copy("stim_velocity",    "STIM_VELOCITY",    110)


# ============================================================
# MAIN MIDI BUILDER
# ============================================================

def build_brain_music(cfg):
    """
    Given normalized cfg (paths + parameters),
    generate a full 4-track MIDI and return midi_path.
    """
    # ---------------- LOAD DATA -----------------
    timeline = pd.read_csv(cfg.timeline_csv)
    licks    = load_lick_times(cfg.lick_csv)
    stim_ev  = load_stim_events(cfg.stim_csv)

    # Timeline columns
    start_col  = _need(timeline, "Start")
    dur_col    = _need(timeline, "Duration")

    label_col = None
    for opt in ("Label_map", "Label", "State"):
        if opt.lower() in {c.lower():c for c in timeline.columns}:
            label_col = {c.lower():c for c in timeline.columns}[opt.lower()]
            break
    if label_col is None:
        raise ValueError("Timeline missing any of Label_map / Label / State")

    # Clean timeline
    timeline[start_col] = pd.to_numeric(timeline[start_col], errors="coerce")
    timeline[dur_col]   = pd.to_numeric(timeline[dur_col], errors="coerce")
    timeline = timeline.dropna(subset=[start_col, dur_col])
    timeline = timeline[timeline[dur_col] > 0].sort_values(by=start_col).reset_index(drop=True)

    # ---------------- MIDI SETUP -----------------
    try:
        from mido import Message, MidiFile, MidiTrack, MetaMessage
    except ImportError:
        raise SystemExit("Install mido:\n   pip install mido python-rtmidi")

    mid = MidiFile(ticks_per_beat=cfg.ticks_per_beat)

    tempo_us = int(60_000_000 / cfg.bpm)
    ticks_per_second = cfg.ticks_per_beat * cfg.bpm / 60.0

    # Track 0: tempo/meta
    meta = MidiTrack()
    meta.append(MetaMessage("set_tempo", tempo=tempo_us, time=0))
    mid.tracks.append(meta)

    # Track 1: organ (states)
    organ = MidiTrack()
    organ.append(Message("program_change", program=cfg.organ_program, time=0, channel=0))
    mid.tracks.append(organ)

    # Track 2: bass (licks)
    bass = MidiTrack()
    bass.append(Message("program_change", program=cfg.bass_program, time=0, channel=1))
    mid.tracks.append(bass)

    # Track 3: stimulus bells
    stim = MidiTrack()
    stim.append(Message("program_change", program=cfg.stim_program, time=0, channel=2))
    mid.tracks.append(stim)

    # ---------------- ORGAN STATES -----------------
    current_ticks_org = 0
    for _, r in timeline.iterrows():
        start_s = float(r[start_col])
        dur_s   = float(r[dur_col])
        lab     = str(r[label_col])

        start_ticks = safe_ticks(start_s * ticks_per_second)
        dur_ticks   = safe_ticks(dur_s * ticks_per_second)

        delta = max(0, start_ticks - current_ticks_org)

        pitch_raw = label_to_pitch(lab)
        pitch     = safe_note(pitch_raw, lo=36, hi=96, wrap=True)

        velocity = safe_vel(85)

        organ.append(Message("note_on",  note=pitch, velocity=velocity, time=delta,     channel=0))
        organ.append(Message("note_off", note=pitch, velocity=64,       time=dur_ticks, channel=0))

        current_ticks_org = start_ticks + dur_ticks

    # ---------------- BASS LICKS -----------------
    lick_len_ticks = safe_ticks(cfg.bass_len_s * ticks_per_second)
    current_ticks_b = 0

    for t in licks:
        start_ticks = safe_ticks(t * ticks_per_second)
        delta = max(0, start_ticks - current_ticks_b)

        bass.append(Message("note_on",  note=cfg.bass_note, velocity=cfg.bass_velocity,
                             time=delta, channel=1))
        bass.append(Message("note_off", note=cfg.bass_note, velocity=64,
                             time=lick_len_ticks, channel=1))

        current_ticks_b = start_ticks + lick_len_ticks

    # ---------------- STIMULI BELLS -----------------
    stim_len_ticks = safe_ticks(cfg.stim_len_s * ticks_per_second)
    current_ticks_s = 0

    for t, lbl in stim_ev:
        start_ticks = safe_ticks(t * ticks_per_second)
        delta = max(0, start_ticks - current_ticks_s)

        note = safe_note(stim_note_for(lbl, cfg.stim_default_note), lo=60, hi=96, wrap=False)

        stim.append(Message("note_on",  note=note, velocity=cfg.stim_velocity,
                             time=delta, channel=2))
        stim.append(Message("note_off", note=note, velocity=64,
                             time=stim_len_ticks, channel=2))

        current_ticks_s = start_ticks + stim_len_ticks

    # ---------------- SAVE -----------------
    midi_path = os.path.join(cfg.out_dir, "Brain_Music.mid")
    mid.save(midi_path)
    print(f"[BrainMusic] MIDI saved to: {midi_path}")

    return midi_path


# ============================================================
# Entry point called from GUI (main.py)
# ============================================================

def run_music(cfg):
    _normalize_music_cfg(cfg)

    # Determine out_dir
    if not getattr(cfg, "out_dir", None):
        base = os.path.dirname(os.path.abspath(cfg.timeline_csv))
        cfg.out_dir = os.path.join(base, "..", "Brain_Music")
    os.makedirs(cfg.out_dir, exist_ok=True)

    midi_path = build_brain_music(cfg)

    return {
        "out_dir": cfg.out_dir,
        "midi_path": midi_path,
    }
