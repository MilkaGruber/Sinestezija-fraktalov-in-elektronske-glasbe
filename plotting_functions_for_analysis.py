import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

from main_julia import AudioAnalyzer 
from main_julia import TRACK_PRESETS

audio_file = "music/TheMysteryoftheYetiPart2_track3.wav"
analyzer = AudioAnalyzer(audio_file)

def mmss(x, pos):
    m = int(x // 60)
    s = int(x % 60)
    return f"{m}:{s:02d}"

def plot_RMS(analyzer):
    time_rms = np.arange(len(analyzer.rms)) * analyzer.hop / analyzer.sr

    def mmss(x, pos):
        m = int(x // 60)
        s = int(x % 60)
        return f"{m}:{s:02d}"

    volume = []
    for t in time_rms:
        f = analyzer.features_at(float(t))
        volume.append(f["volume"])

    volume = np.array(volume)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(
        time_rms,
        analyzer.rms,
        linewidth=1.5,
        alpha=0.8,
        label="RMS"
    )
    ax.plot(
        time_rms,
        volume,
        linewidth=2.0,
        alpha=0.9,
        label="Volume (normalized)"
    )

    t1 = 182   
    t2 = 241

    ax.axvline(
        t1,
        linestyle="--",
        color="green",
        linewidth=2,
        label="Start of short segment"
    )

    ax.axvline(
        t2,
        linestyle="--",
        color="green",
        linewidth=2,
        label="End of short segment"
    )

    ax.xaxis.set_major_locator(MultipleLocator(60))   # 1 minute
    ax.xaxis.set_minor_locator(MultipleLocator(10))   # 10 seconds
    ax.xaxis.set_major_formatter(FuncFormatter(mmss))

    ax.set_xlabel("Time (mm:ss)", fontsize=14)
    ax.set_ylabel("RMS Energy", fontsize=14)
    ax.set_title(
        f"RMS Energy Over Time: {os.path.basename(audio_file)}",
        fontsize=16,
        pad=20
    )

    ax.grid(True, which="major", alpha=0.45, linewidth=1.0)
    ax.grid(True, which="minor", alpha=0.25, linewidth=0.6)

    ax.legend(fontsize=12, loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_volume(analyzer, dt=0.01):
    t_values = np.arange(0, analyzer.duration, dt)
    volume = np.array([analyzer.features_at(t)["volume"] for t in t_values])

    _, ax = plt.subplots(figsize=(14, 8))
    ax.plot(t_values, volume, label='Volume')
    
    t1 = 182   
    t2 = 241

    ax.axvline(
        t1,
        linestyle="--",
        color="green",
        linewidth=2,
        label="Start of short segment"
    )

    ax.axvline(
        t2,
        linestyle="--",
        color="green",
        linewidth=2,
        label="End of short segment"
    )

    ax.xaxis.set_major_locator(MultipleLocator(60))   # 1 minute
    ax.xaxis.set_minor_locator(MultipleLocator(10))   # 10 seconds
    ax.xaxis.set_major_formatter(FuncFormatter(mmss))

    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_xlabel("Volume", fontsize=14)
    ax.set_title(
    f"Volume Over Time: {os.path.basename(audio_file)}",
    fontsize=16,
    pad=20
)

    ax.grid(True, which="major", alpha=0.45, linewidth=1.0)
    ax.grid(True, which="minor", alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=12, loc="upper right")

    plt.tight_layout()
    plt.show()

from matplotlib.ticker import MultipleLocator, FuncFormatter

def plot_spectrogram(analyzer, max_freq=20000):
    freqs = analyzer.freqs
    spec = analyzer.spec
    idx_max = np.searchsorted(freqs, max_freq)

    fig, ax = plt.subplots(figsize=(15, 5))

    im = ax.imshow(
        20 * np.log10(spec[:idx_max, :] + 1e-6),
        origin='lower',
        aspect='auto',
        extent=[0, analyzer.duration, freqs[0], freqs[idx_max - 1]],
        cmap='magma'
    )

    
    t1 = 182   
    t2 = 241

    ax.axvline(
        t1,
        linestyle="--",
        color="green",
        linewidth=2,
        label="Start of short segment"
    )

    ax.axvline(
        t2,
        linestyle="--",
        color="green",
        linewidth=2,
        label="End of short segment"
    )

    fig.colorbar(im, ax=ax, label="Magnitude [dB]")

    ax.xaxis.set_major_locator(MultipleLocator(60))   # 1 minute
    ax.xaxis.set_minor_locator(MultipleLocator(10))   # 10 seconds
    ax.xaxis.set_major_formatter(FuncFormatter(mmss))

    ax.set_xlabel("Time (mm:ss)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram")

    ax.grid(False)  

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.patches import Patch


def plot_onset_with_trend(analyzer, dt=0.01):
    t_values = np.linspace(0, analyzer.duration, int(analyzer.duration / dt) + 1)

    onset_vals = []
    trend_vals = []

    for t in t_values:
        f = analyzer.features_at(t)
        onset_vals.append(f["onset"])
        trend_vals.append(f["trend"])

    onset_vals = np.array(onset_vals)
    trend_vals = np.array(trend_vals)

    trend_colors = np.array(['#d0f0c0', '#99e2ff', '#ff9999'])
    bg_colors = trend_colors[trend_vals]

    fig, ax = plt.subplots(figsize=(15, 4))

    for i in range(len(t_values) - 1):
        ax.axvspan(t_values[i], t_values[i + 1],
                   color=bg_colors[i], alpha=0.3, linewidth=0)

    ax.plot(t_values, onset_vals, color="#256680", label='Onset Envelope', lw=0.8)

    
    t1 = 182   
    t2 = 241

    ax.axvline(
        t1,
        linestyle="--",
        color="green",
        linewidth=2,
        label="Start of short segment"
    )

    ax.axvline(
        t2,
        linestyle="--",
        color="green",
        linewidth=2,
        label="End of short segment"
    )
    
    ax.xaxis.set_major_locator(MultipleLocator(60))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x//60)}:{int(x%60):02}"))


    ax.set_xlabel("Time (mm:ss)", fontsize=14)
    ax.set_ylabel("Onset Strength", fontsize=14)
    ax.set_title(
        f"Onset Envelope with Trends: {os.path.basename(audio_file)}",
        fontsize=16,
        pad=20
    )

    ax.grid(True, which="major", alpha=0.45, linewidth=1.0)
    ax.grid(True, which="minor", alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=12, loc="upper right")


    plt.show()



def plot_smooth_onset_thresholds(analyzer: AudioAnalyzer):
    times = np.arange(len(analyzer.smoothed_onset)) * analyzer.hop / analyzer.sr
    smoothed = analyzer.smoothed_onset

    par33 = TRACK_PRESETS[analyzer.filename].par33
    par66 = TRACK_PRESETS[analyzer.filename].par66

    trend_vals = np.array([analyzer.features_at(t)["trend"] for t in times])
    trend_colors = np.array(['#d0f0c0', '#99e2ff', '#ff9999'])
    bg_colors = trend_colors[trend_vals]

    fig, ax = plt.subplots(figsize=(15, 4))

    for i in range(len(times) - 1):
        ax.axvspan(times[i], times[i + 1],
                   color=bg_colors[i], alpha=0.3, linewidth=0)

    ax.plot(times, smoothed, label="Smoothed onset", color="#256680", lw=1.0)
    ax.xaxis.set_major_locator(MultipleLocator(60))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x//60)}:{int(x%60):02}"))

    ax.axhline(par33, color="orange", linestyle="--", label=f"par33 ({par33:.3f})")
    ax.axhline(par66, color="red", linestyle="--", label=f"par66 ({par66:.3f})")

    
    t1 = 182   
    t2 = 241

    ax.axvline(
        t1,
        linestyle="--",
        color="green",
        linewidth=2,
        label="Start of short segment"
    )

    ax.axvline(
        t2,
        linestyle="--",
        color="green",
        linewidth=2,
        label="End of short segment"
    )



    ax.set_xlabel("Time [mm:ss]", fontsize=14)
    ax.set_ylabel("Smoothed onset", fontsize=14)
    ax.set_title(
        f"Smoothed onset with thresholds for {analyzer.filename}",
        fontsize=16,
        pad=20
    )


    ax.grid(True, which="major", alpha=0.45, linewidth=1.0)
    ax.grid(True, which="minor", alpha=0.25, linewidth=0.6)

    trend_legend = [
        Patch(facecolor="#d0f0c0", edgecolor="none", label="Low trend"),
        Patch(facecolor="#99e2ff", edgecolor="none", label="Medium trend"),
        Patch(facecolor="#ff9999", edgecolor="none", label="High trend"),
    ]

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles + trend_legend,
        labels + [p.get_label() for p in trend_legend],
        loc="upper right",
        fontsize=11
    )

    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.patches import Patch

def plot_beat(analyzer, dt=0.01):
    t_values = np.linspace(0, analyzer.duration, int(analyzer.duration / dt) + 1)
    beat_vals = np.array([analyzer.features_at(t)["beat"] for t in t_values])

    trend_vals = np.array([analyzer.features_at(t)["trend"] for t in t_values])
    trend_colors = np.array(['#d0f0c0', '#99e2ff', '#ff9999'])
    bg_colors = trend_colors[trend_vals]

    fig, ax = plt.subplots(figsize=(15, 4))

    for i in range(len(t_values) - 1):
        ax.axvspan(
            t_values[i], t_values[i + 1],
            color=bg_colors[i],
            alpha=0.3,
            linewidth=0
        )

    ax.plot(t_values, beat_vals, color="#256680", label='Beat', lw=0.9)

    
    t1 = 182   
    t2 = 241

    ax.axvline(
        t1,
        linestyle="--",
        color="green",
        linewidth=2,
        label="Start of short segment"
    )

    ax.axvline(
        t2,
        linestyle="--",
        color="green",
        linewidth=2,
        label="End of short segment"
    )

    ax.set_ylim(0, 1.05)
    ax.set_title("Smoothed Beat Over Time")
    ax.set_ylabel("Beat Strength")
    ax.set_xlabel("Time [mm:ss]")

    ax.xaxis.set_major_locator(MultipleLocator(60))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{int(x//60)}:{int(x%60):02}")
    )
    ax.set_xlim(0, analyzer.duration)

    ax.grid(True, which="major", alpha=0.45, linewidth=1.0)
    ax.grid(True, which="minor", alpha=0.25, linewidth=0.6)


    trend_legend = [
        Patch(facecolor="#d0f0c0", edgecolor="none", label="Low trend"),
        Patch(facecolor="#99e2ff", edgecolor="none", label="Medium trend"),
        Patch(facecolor="#ff9999", edgecolor="none", label="High trend"),
    ]

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles + trend_legend,
        labels + [p.get_label() for p in trend_legend],
        fontsize=11,
        loc="upper right"
    )

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.patches import Patch

def plot_pulse(analyzer, dt=0.01, max_duration=60):
    t_values = np.linspace(0, analyzer.duration, int(analyzer.duration / dt) + 1)
    
    t_values = t_values[t_values <= max_duration]

    pulse_vals = np.array([analyzer.features_at(t)["pulse"] for t in t_values])

    trend_vals = np.array([analyzer.features_at(t)["trend"] for t in t_values])
    trend_colors = np.array(['#d0f0c0', '#99e2ff', '#ff9999'])
    bg_colors = trend_colors[trend_vals]

    fig, ax = plt.subplots(figsize=(15, 4))

    for i in range(len(t_values) - 1):
        ax.axvspan(
            t_values[i], t_values[i + 1],
            color=bg_colors[i],
            alpha=0.3,
            linewidth=0
        )

    ax.plot(t_values, pulse_vals, color="#256680", label='Pulse', lw=0.9)

    
    t1 = 182   
    t2 = 241

    ax.axvline(
        t1,
        linestyle="--",
        color="green",
        linewidth=2,
        label="Start of short segment"
    )

    ax.axvline(
        t2,
        linestyle="--",
        color="green",
        linewidth=2,
        label="End of short segment"
    )

    ax.set_ylim(0, 1.05)
    ax.set_title("Pulse Over Time")
    ax.set_ylabel("Pulse Strength")
    ax.set_xlabel("Time [mm:ss]")

    ax.xaxis.set_major_locator(MultipleLocator(60))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{int(x//60)}:{int(x%60):02}")
    )

    ax.set_xlim(0, min(max_duration, analyzer.duration))
    ax.grid(True, which="major", alpha=0.45, linewidth=1.0)
    ax.grid(True, which="minor", alpha=0.25, linewidth=0.6)

    trend_legend = [
        Patch(facecolor="#d0f0c0", edgecolor="none", label="Low trend"),
        Patch(facecolor="#99e2ff", edgecolor="none", label="Medium trend"),
        Patch(facecolor="#ff9999", edgecolor="none", label="High trend"),
    ]

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles + trend_legend,
        labels + [p.get_label() for p in trend_legend],
        fontsize=11,
        loc="upper right"
    )

    plt.tight_layout()
    plt.show()

