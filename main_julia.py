# Usage:
#   py -3.11 main_julia.py audio_for_shorts/TheMysteryoftheYetiPart2_track3_shorter_segment.wav
#   py -3.11 main_julia.py "music/TheMysteryoftheYetiPart2_track3.wav"
#
# Dependencies:
#   py -3.11 -m pip install numpy glfw moderngl librosa imageio imageio-ffmpeg scipy

import os
import sys
import time
import math
import shutil
import subprocess
import numpy as np
import glfw
import moderngl

from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from dataclasses import dataclass

# =========================
# QUALITY / TIMING KNOBS
# =========================
FPS = 60

#OUT_W, OUT_H = 1080, 1920  # vertical for Shorts
OUT_W, OUT_H = 1920, 1080  # for computer screen
SUPERSAMPLE = 1
CRF = 16

# Feedback per tier (baseline) - global parameters for drawing
LOW_FB_AMOUNT = 0.045
MID_FB_AMOUNT = 0.085
HIGH_FB_AMOUNT = 0.140

LOW_FB_DECAY = 0.991
MID_FB_DECAY = 0.9885
HIGH_FB_DECAY = 0.9845

TIER_EMA = 0.08
TIER_HYST = 0.10

GLOBAL_SPEEDUP = 1.12

@dataclass
class TrackConfig:
    # Parameters that are different for every track go here - get them by looking at musical analysis
    par33: float
    par66: float # thresholds for determining low, mid and high parts of the track in AudioAnalyzer
    # Beat influence: 
    BEAT_SPEED_INFLUENCE: float # affects colour change
    STRONG_PULSE_THRESH: float # adjust pulse threshold 
    STARTING_ORBITING_SPEED: float # starting orbiting speed

    SCALING_ORBITING_SPEED: float  # this determines the speed of orbit changing: if zoom is larger, this should be smaller
    SKIP_BEAT: int # if beats are too common - we zoom in on every skip_beat
    BEAT_ZOOM_THRESHOLD: float

    COOLDOWN_LOW: int # how long does the scene cooldown before changing in low energy parts
    COOLDOWN_MID: int # how long does the scene cooldown before changing in mid energy parts
    COOLDOWN_HIGH: int # how long does the scene cooldown before changing in high energy parts

    PAL_RANGE_LOW: tuple
    PAL_RANGE_MID: tuple
    PAL_RANGE_HIGH: tuple
    


TRACK_PRESETS = {
    r"audio_for_shorts/TheMysteryoftheYetiPart2_track3_shorter_segment.wav": 
        TrackConfig(par33=0.03, par66=0.05, BEAT_SPEED_INFLUENCE=0.16, STRONG_PULSE_THRESH=0.95, STARTING_ORBITING_SPEED=0.5, SCALING_ORBITING_SPEED=0.05, # drugi yt short upload
                    SKIP_BEAT=4, BEAT_ZOOM_THRESHOLD=0.20,
                    COOLDOWN_LOW=6, COOLDOWN_MID=4, COOLDOWN_HIGH=4, 
                    # range for colour, range for saturation, range for brightness
                    PAL_RANGE_LOW=(0.08,0.38,0.75,0.85,0.85,0.95),
                    PAL_RANGE_MID=(0.08,0.38,0.75,0.85,0.85,0.95),
                    PAL_RANGE_HIGH=(0.8,0.98,0.85,1,0.85,0.95)
                    ), 
    #r"audio_for_shorts/TheMysteryoftheYetiPart2_track3_shorter_segment.wav": 
    #    TrackConfig(BEAT_SPEED_INFLUENCE=0.16, STRONG_PULSE_THRESH=0.95, STARTING_ORBITING_SPEED=0.5, SCALING_ORBITING_SPEED=0.1,
    #                COOLDOWN_LOW=6, COOLDOWN_MID=4, COOLDOWN_HIGH=2, STARTING_PAL=np.array([0.25, 0.2, 1], dtype=np.float32),
    #                # range for colour, range for saturation, range for brightness
    #                PAL_RANGE_LOW=(0.55, 0.78, 0.25, 0.4, 0.85,0.9),
    #                PAL_RANGE_MID=(0.08,0.38,0.6,0.8,0.85,0.95),
    #                PAL_RANGE_HIGH=(0.8,0.98,0.8,1,0.85,0.95)
    #                )
    r"music/TheMysteryoftheYetiPart2_track3.wav": 
        TrackConfig(par33=0.03, par66=0.05,  # look at musical analysis - Onsets with trends to determine these values
                    BEAT_SPEED_INFLUENCE=0.16, STRONG_PULSE_THRESH=0.95, STARTING_ORBITING_SPEED=0.5, SCALING_ORBITING_SPEED=0.05, 
                    SKIP_BEAT=4, BEAT_ZOOM_THRESHOLD=0.20, # also check Smoothed Beat over time graph
                    COOLDOWN_LOW=6, COOLDOWN_MID=5, COOLDOWN_HIGH=3, 
                    # range for colour, range for saturation, range for brightness
                    PAL_RANGE_LOW=(0.55, 0.78, 0.6,0.7,0.85,0.95),
                    PAL_RANGE_MID=(0.08,0.38,0.7,0.85,0.85,0.95),
                    PAL_RANGE_HIGH=(0.8,0.98,0.85,1,0.85,0.95)
                    ), 
                
}

def clamp(x, a, b):
    return a if x < a else (b if x > b else x)


def lerp(a, b, t):
    t = clamp(float(t), 0.0, 1.0)
    return float(a) * (1.0 - t) + float(b) * t


def lerp3(low, mid, high, x01):
    x = clamp(float(x01), 0.0, 1.0) * 2.0
    if x <= 1.0:
        return lerp(low, mid, x)
    return lerp(mid, high, x - 1.0)


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


# =========================
# AUDIO ANALYZER (trend-based tiers)
# =========================
class AudioAnalyzer:
    def __init__(self, filename: str):
        import librosa
        track_config = TRACK_PRESETS[filename]

        self.filename = filename
        self.y, self.sr = librosa.load(filename, sr=None, mono=True)
        self.duration = float(librosa.get_duration(y=self.y, sr=self.sr))
        self.hop = 512

        self.rms = librosa.feature.rms(y=self.y, hop_length=self.hop)[0].astype(np.float32)
        self.spec = np.abs(librosa.stft(self.y, hop_length=self.hop)).astype(np.float32)
        self.freqs = librosa.fft_frequencies(sr=self.sr).astype(np.float32)

        oe = librosa.onset.onset_strength(y=self.y, sr=self.sr, hop_length=self.hop).astype(np.float32)
        oe -= oe.min()
        if oe.max() > 1e-9:
            oe /= oe.max()
        self.onset_env = oe

        tempo, beat_frames = librosa.beat.beat_track(
            y=self.y, sr=self.sr, hop_length=self.hop, units="frames"
        )
        if tempo is None:
            self.tempo = 140.0
        else:
            arr = np.asarray(tempo).reshape(-1)
            self.tempo = float(arr[0]) if arr.size else 140.0

        if beat_frames is None or len(beat_frames) == 0:
            self.beat_times = np.array([0.0], dtype=np.float32)
        else:
            beat_frames = np.asarray(beat_frames, dtype=np.int32)
            self.beat_times = (beat_frames * self.hop / self.sr).astype(np.float32)

        self.rms_p90 = float(np.percentile(self.rms, 90))
        self.rms_p70 = float(np.percentile(self.rms, 70))

        def detect_trends(onset_env, sr, hop, window_sec=10.0):
            hop_t = hop / sr
            w = int(window_sec / hop_t)
            if w % 2 == 0:
                w += 1
            w = max(w, 3)
 
            # smooth onset
            sm = savgol_filter(onset_env, w, 3, mode="mirror")
 
            # adaptive thresholds
            p33 = track_config.par33
            p66 = track_config.par66
 
            # long-term trend
            trend = gaussian_filter1d(sm, sigma=w / 6.0)
            avg = np.convolve(trend, np.ones(w) / w, mode="same")
 
            labels = np.zeros_like(avg, dtype=np.uint8)
            labels[avg >= p33] = 1   # medium
            labels[avg >= p66] = 2   # high

            print(f"[Trends] Low/Med={p33:.3f}  Med/High={p66:.3f}")
            return (labels, avg)
        
        labels, smoothed_onset = detect_trends(self.onset_env, self.sr, self.hop)
        self.trend_labels = labels
        self.smoothed_onset = smoothed_onset

        self._state = {"volume": 0.0, "low": 0.0, "mid": 0.0, "high": 0.0, "beat": 0.0}
        self.alpha_feat = 0.18
        self.alpha_beat = 0.30

    @staticmethod
    def _ema(prev, x, a):
        return prev * (1.0 - a) + x * a

    def _idx(self, t):
        idx = int(t * self.sr / self.hop)
        return int(np.clip(idx, 0, len(self.rms) - 1))

    def trend_at(self, t: float) -> int:
        idx = self._idx(t)
        return int(self.trend_labels[min(idx, len(self.trend_labels) - 1)])

    def beat_pulse(self, t: float, sigma: float = 0.040) -> float:
        bt = self.beat_times
        if len(bt) < 2:
            return 0.0
        j = int(np.searchsorted(bt, t))
        cand = []
        if 0 <= j < len(bt):
            cand.append(bt[j])
        if 0 <= j - 1 < len(bt):
            cand.append(bt[j - 1])
        d = min(abs(t - c) for c in cand) if cand else 1e9
        return float(np.exp(-(d * d) / (2.0 * sigma * sigma)))

    def beat_index_and_phase(self, t: float):
        bt = self.beat_times
        if len(bt) < 2:
            b = (t * self.tempo / 60.0)
            bi = int(max(0, math.floor(b)))
            phase = float(b - bi)
            return bi, phase

        t = float(np.clip(t, 0.0, self.duration))
        j = int(np.searchsorted(bt, t, side="right")) - 1
        j = int(np.clip(j, 0, len(bt) - 2))
        t0 = float(bt[j])
        t1 = float(bt[j + 1])
        denom = max(1e-6, (t1 - t0))
        phase = (t - t0) / denom
        return j, float(np.clip(phase, 0.0, 0.999999))

    def features_at(self, t: float):
        t = max(0.0, min(t, self.duration))
        idx = self._idx(t)

        rms = float(self.rms[idx])
        spec_col = self.spec[:, idx]

        low_mask = self.freqs < 200
        mid_mask = (self.freqs >= 200) & (self.freqs < 2000)
        high_mask = self.freqs >= 2000

        low = float(spec_col[low_mask].mean()) if np.any(low_mask) else 0.0
        mid = float(spec_col[mid_mask].mean()) if np.any(mid_mask) else 0.0
        high = float(spec_col[high_mask].mean()) if np.any(high_mask) else 0.0

        vol = np.clip(rms / (self.rms_p70 + 1e-9), 0.0, 2.0)
        vol = float(np.sqrt(np.clip(vol, 0.0, 1.0)))

        smoothed_onset_value = (
            float(self.smoothed_onset[idx])
            if hasattr(self, "smoothed_onset") and idx < len(self.smoothed_onset)
            else 0.0)

        def soft_norm(x, k):
            v = np.clip(x / (k + 1e-9), 0.0, 3.0)
            return float(np.sqrt(np.clip(v, 0.0, 1.0)))

        low_n = soft_norm(low, 50.0)
        mid_n = soft_norm(mid, 40.0)
        high_n = soft_norm(high, 30.0)

        onset = float(self.onset_env[idx]) if idx < len(self.onset_env) else 0.0
        gate = 1.0 if rms > self.rms_p90 else 0.0
        beat_raw = float(np.clip(0.60 * onset + 0.40 * gate, 0.0, 1.0))

        self._state["volume"] = self._ema(self._state["volume"], vol, self.alpha_feat)
        self._state["low"] = self._ema(self._state["low"], low_n, self.alpha_feat)
        self._state["mid"] = self._ema(self._state["mid"], mid_n, self.alpha_feat)
        self._state["high"] = self._ema(self._state["high"], high_n, self.alpha_feat)

        a = self.alpha_beat if beat_raw > self._state["beat"] else 0.12
        self._state["beat"] = self._ema(self._state["beat"], beat_raw, a)

        pulse = self.beat_pulse(t, sigma=0.038 + 0.030 * (1.0 - self._state["volume"]))

        return {
            "volume": float(np.clip(self._state["volume"], 0.0, 1.0)),
            "low": float(np.clip(self._state["low"], 0.0, 1.0)),
            "mid": float(np.clip(self._state["mid"], 0.0, 1.0)),
            "high": float(np.clip(self._state["high"], 0.0, 1.0)),
            "beat": float(np.clip(self._state["beat"], 0.0, 1.0)),
            "pulse": float(np.clip(pulse, 0.0, 1.0)),
            "tempo": float(self.tempo),
            "onset": onset,
            "trend": self.trend_at(t),
            "smooth_onset": smoothed_onset_value  
        }


# =========================
# Shaders
# =========================
VERTEX_SHADER = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRACTAL_FS = r"""
#version 330
uniform vec2  u_resolution;
uniform float u_time;

uniform float u_volume;
uniform vec2  u_c;
uniform vec2  u_center;
uniform float u_logZoom;
uniform vec3  u_pal;

in vec2 v_uv;
out vec4 fragColor;

float saturate(float x){ return clamp(x, 0.0, 1.0); }

vec3 hsv2rgb(vec3 c) {
    vec3 rgb = clamp(abs(mod(c.x*6.0 + vec3(0.0,4.0,2.0),6.0)-3.0)-1.0, 0.0, 1.0);
    return c.z * mix(vec3(1.0), rgb, c.y);
}

vec4 juliaSmooth(vec2 z, vec2 c, int max_iter){
    float m2 = 0.0;
    float trap = 1e9;
    for(int i=0;i<max_iter;i++){
        trap = min(trap, dot(z,z));
        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        m2 = dot(z,z);
        if(m2 > 256.0){
            float log_zn = log(m2) / 2.0;
            float nu = log(log_zn / log(2.0)) / log(2.0);
            float smoothIter = float(i) + 1.0 - nu;
            float t = smoothIter / float(max_iter);
            float glow = exp(-2.2 * trap);
            return vec4(t, 1.0, glow, trap);
        }
    }
    float glow = exp(-2.2 * trap);
    return vec4(1.0, 0.0, glow, trap);
}

vec3 rainbowBands(float x, float seed, float detail, float sat_override, float val_override){
    float hue = fract(seed + 1.35*x + 0.03*sin(0.07*u_time) + 0.05*detail);
    float sat = sat_override;
    float val = val_override;

    val *= 0.70 + 0.30*sin(6.28318*(x*2.4 + 0.14*detail));
    val = clamp(val, 0.0, 1.0);

    return hsv2rgb(vec3(hue, sat, val));
}

vec3 shade(vec4 rr, float seed){
    float t = rr.x;
    float escaped = rr.y;
    float glow = rr.z;
    float trap = max(rr.w, 1e-9);

    float edge = pow(saturate(1.0 - t), 2.0);
    vec3 bg = vec3(0.0);

    float bands  = 0.32 * sin(40.0 * t + 0.7 * sin(0.05*u_time));
    float detail = 0.16 * sin(2.9 * log(trap) + 0.03 * u_time);
    float k      = fract(1.85*t + bands + detail);

    vec3 fr = rainbowBands(k, seed, detail, u_pal.y, u_pal.z);
    fr *= 0.78 + 0.95*edge;

    vec3 glowCol = rainbowBands(fract(k + 0.11), seed + 0.17, detail, u_pal.y, u_pal.z);
    fr += glowCol * (0.02 + 1.10*glow) * pow(edge, 1.15);

    fr = clamp(fr, 0.0, 1.0);

    vec3 col = mix(bg, fr, step(0.5, escaped));

    float outline = smoothstep(0.03, 0.22, edge);
    col = mix(col, fr * 1.12, 0.30 * outline);

    col = col / (1.0 + 0.25 * max(max(col.r, col.g), col.b));
    col = pow(clamp(col, 0.0, 1.0), vec3(1.0/1.85));
    col = max(col - 0.020, 0.0);

    return clamp(col, 0.0, 1.0);
}

void main(){
    vec2 uv = (v_uv - 0.5) * 2.0;
    uv.x *= u_resolution.x / u_resolution.y;

    float scale = exp(u_logZoom) * 0.65;
    vec2 z = uv * scale + u_center;

    float zoomFactor = saturate((-u_logZoom - 0.10) / 4.5);
    int max_iter = int(650.0 + 1850.0*zoomFactor + 240.0*u_volume);

    vec4 r = juliaSmooth(z, u_c, max_iter);
    vec3 col = shade(r, u_pal.x);
    fragColor = vec4(col, 1.0);
}
"""

FEEDBACK_FS = r"""
#version 330
uniform sampler2D u_prev;
uniform sampler2D u_curr;

uniform float u_decay;
uniform float u_amount;
uniform float u_kick;

in vec2 v_uv;
out vec4 fragColor;

void main(){
    vec2 uv = v_uv;

    vec2 p = uv - 0.5;
    float z = 1.0 - (0.0010 + 0.0015*u_kick);
    p *= z;
    vec2 uv_prev = p + 0.5;

    vec3 prev = texture(u_prev, uv_prev).rgb * u_decay;
    vec3 curr = texture(u_curr, uv).rgb;

    vec3 trail = prev * (u_amount * 0.85);
    vec3 outc  = 1.0 - (1.0 - curr) * (1.0 - trail);

    outc = clamp(outc, 0.0, 1.0);
    fragColor = vec4(outc, 1.0);
}
"""

COPY_FS = r"""
#version 330
uniform sampler2D u_tex;
uniform vec2 u_texel;
in vec2 v_uv;
out vec4 fragColor;

void main(){
    vec3 c  = texture(u_tex, v_uv).rgb;
    vec3 cx = texture(u_tex, v_uv + vec2(u_texel.x, 0.0)).rgb;
    vec3 cX = texture(u_tex, v_uv - vec2(u_texel.x, 0.0)).rgb;
    vec3 cy = texture(u_tex, v_uv + vec2(0.0, u_texel.y)).rgb;
    vec3 cY = texture(u_tex, v_uv - vec2(0.0, u_texel.y)).rgb;

    vec3 sharp = c * 1.25 - (cx + cX + cy + cY) * 0.0625;
    sharp = clamp(sharp, 0.0, 1.0);
    fragColor = vec4(sharp, 1.0);
}
"""


class SafeUniforms:
    def __init__(self, program: moderngl.Program):
        self.p = program
        self._missing = set()

    def set(self, name: str, value):
        if name in self._missing:
            return
        try:
            self.p[name].value = value
        except KeyError:
            self._missing.add(name)
            print(f"Uniform manjka v shaderju: {name}")


class VideoWriter:
    def __init__(self, out_path, fps=60, crf=16):
        import imageio.v2 as imageio
        self._writer = imageio.get_writer(
            out_path,
            fps=int(fps),
            codec="libx264",
            quality=None,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", str(int(crf))],
        )

    def add_frame(self, rgb_uint8_hwc):
        self._writer.append_data(rgb_uint8_hwc)

    def close(self):
        self._writer.close()


def _find_ffmpeg():
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        if ff and os.path.isfile(ff):
            return ff
    except Exception:
        pass
    return None


def mux_audio(video_in, audio_in, video_out):
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        print("ffmpeg ni na voljo (PATH ali imageio-ffmpeg). Video ostane brez audio.")
        return False

    cmd = [
        ffmpeg, "-y",
        "-i", os.path.abspath(video_in),
        "-i", os.path.abspath(audio_in),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        os.path.abspath(video_out),
    ]
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        print("Mux audio ni uspel. Video bo brez audio.")
        return False


# =========================
# NEW JULIA PRESET 
# import Julia Scene presets
# =========================
from julia_preset_explorer import JULIA_PRESETS_2

JULIA_PRESETS = JULIA_PRESETS_2

def build_orbit_circle(C, P, N):
    # we need this to build orbits for different c values in JULIA_PRESETS
    v = P - C
    r = abs(v)
    a0 = math.atan2(v.imag, v.real)

    orbit = []
    for i in range(N):
        t = 2.0 * math.pi * i / N
        a = a0 + t
        orbit.append(C + complex(
            r * math.cos(a),
            r * math.sin(a)
        ))

    return orbit


def pick_pal_for_tier(rng: np.random.Generator, tier_state: int, track_config: TrackConfig) -> np.ndarray:
    if tier_state == 0:
        h_min, h_max, s_min, s_max, v_min, v_max = track_config.PAL_RANGE_LOW
    elif tier_state == 1:
        h_min, h_max, s_min, s_max, v_min, v_max = track_config.PAL_RANGE_MID
    else:
        h_min, h_max, s_min, s_max, v_min, v_max = track_config.PAL_RANGE_HIGH

    h = float(rng.uniform(h_min, h_max))
    s = float(rng.uniform(s_min, s_max))
    v = float(rng.uniform(v_min, v_max))
    return np.array([h, s, v], dtype=np.float32)

def get_scene(julia_presets, index, rng):
    import numpy as np
    n = len(julia_presets)
    index = index % n
    preset = julia_presets[index]
    c1, c2 = preset["c"]
    c0 = np.array([c1, c2], dtype=np.float32)
    cx, cy = preset["center"]
    center = np.array([cx, cy], dtype=np.float32)
    logZoom0 = float(preset["zoom"][0])
    logZoom1 = float(preset["zoom"][1])
    orbit = preset.get("orbit")
    return {"c0": c0, "center": center, "logZoom0": logZoom0, "logZoom1": logZoom1, "orbit": orbit}

# =========================
# MAIN
# =========================
def main():
    audio_file = sys.argv[1] if len(sys.argv) > 1 else None
    track_config = TRACK_PRESETS[audio_file]
    if not audio_file or not os.path.isfile(audio_file):
        print("Uporaba: py -3.11 fractal_julia_feedback_rhythm_record.py audio/psy.wav")
        sys.exit(1)

    audio = AudioAnalyzer(audio_file)
    print("Audio:", audio_file)
    print("Duration (s):", audio.duration, "Tempo (bpm):", round(audio.tempo, 2))

    RW, RH = int(OUT_W * SUPERSAMPLE), int(OUT_H * SUPERSAMPLE)
    aspect = OUT_W / OUT_H
    print(f"Render res: {RW}x{RH}  Output: {OUT_W}x{OUT_H}  Supersample: {SUPERSAMPLE}x")

    base = os.path.splitext(os.path.basename(audio_file))[0]
    out_noaudio = f"{base}_julia_fixedGeom_colorTier_noaudio.mp4"
    out_final = f"{base}_julia_fixedGeom_colorTier.mp4"

    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
    window = glfw.create_window(OUT_W, OUT_H, "Julia FIXED-GEOM Render", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Could not create GLFW window")

    glfw.make_context_current(window)
    glfw.swap_interval(0)

    ctx = moderngl.create_context()

    prog_fr = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRACTAL_FS)
    prog_fb = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FEEDBACK_FS)
    prog_cp = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=COPY_FS)

    U_fr = SafeUniforms(prog_fr)
    U_fb = SafeUniforms(prog_fb)
    U_cp = SafeUniforms(prog_cp)

    quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
    vbo = ctx.buffer(quad.tobytes())
    vao_fr = ctx.simple_vertex_array(prog_fr, vbo, "in_pos")
    vao_fb = ctx.simple_vertex_array(prog_fb, vbo, "in_pos")
    vao_cp = ctx.simple_vertex_array(prog_cp, vbo, "in_pos")

    curr_tex = ctx.texture((RW, RH), components=3, dtype="f2")
    curr_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

    fb_tex_a = ctx.texture((RW, RH), components=3, dtype="f2")
    fb_tex_b = ctx.texture((RW, RH), components=3, dtype="f2")
    for tex in (fb_tex_a, fb_tex_b):
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

    fbo_curr = ctx.framebuffer(color_attachments=[curr_tex])
    fbo_a = ctx.framebuffer(color_attachments=[fb_tex_a])
    fbo_b = ctx.framebuffer(color_attachments=[fb_tex_b])

    out_tex = ctx.texture((OUT_W, OUT_H), components=3, dtype="f2")
    out_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    fbo_out = ctx.framebuffer(color_attachments=[out_tex])

    fbo_a.use(); ctx.clear(0.0, 0.0, 0.0, 1.0)
    fbo_b.use(); ctx.clear(0.0, 0.0, 0.0, 1.0)

    seed = abs(hash(os.path.abspath(audio_file))) % (2**32)
    rng = np.random.default_rng(seed)
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)

    ########################
    # scene initialization
    ########################
    scene_index = 0
    sceneA = get_scene(JULIA_PRESETS, scene_index, rng)
    scene_index = (scene_index + 1) % len(JULIA_PRESETS)
    sceneB = get_scene(JULIA_PRESETS, scene_index, rng)
    pal = pick_pal_for_tier(rng, 0, track_config)

    orbit_dataA = sceneA.get("orbit", None)  # new*
    orbit_dataB = sceneA.get("orbit", None)  # new*

    # INITIAL orbit settings
    orbit_points = None
    orbit_idx = 0
    orbit_len = 10000

    Ca = complex(*orbit_dataA["C"])
    Pa = complex(*orbit_dataA["P"])
    orbit_len = 128  # number of points in the orbit
    orbit_pointsA = build_orbit_circle(Ca, Pa, orbit_len) 
    Cb = complex(*orbit_dataA["C"])
    Pb = complex(*orbit_dataA["P"])
    orbit_len = 128  # number of points in the orbit
    orbit_pointsB = build_orbit_circle(Cb, Pb, orbit_len) 

    # initial parameters for scene interpolation:
    scene_mix = 0.0         # 0 = sceneA, 1 = sceneB
    scene_interp_speed = 0.02  # fraction per frame (tweak)
    time_since_last_scene = 0.0

    # Palette transition state (ONLY colors change)
    pal_in_transition = False
    pal_t_beats = 0.0
    pal_dur_beats = 6.0
    pal_from = pal.copy()
    pal_to = pal.copy()
    pal = pal.copy()
    pending_pal_change = False
    pending_pal_start_beat = 0

    writer = VideoWriter(out_noaudio, fps=FPS, crf=CRF)
    total_frames = int(math.ceil(audio.duration * FPS))
    start = time.time()

    prev_fb = fb_tex_a
    next_fb = fb_tex_b
    prev_fbo = fbo_a
    next_fbo = fbo_b

    need_clear_feedback = False

    tier_state = 0
    tier_smooth = 0.0
    last_checked_beat = -1

    last_logged_sec = -1
    _prev_tier_state = 0

    ###########################################################
    #### preprocessing for zooming to the beat  ###############
    ###########################################################
    # list of times when beats exceed threshold and cooldown
    beat_zoom_threshold = track_config.BEAT_ZOOM_THRESHOLD
    # PREPROCESS all beats above threshold
    zoom_beats_times = []
    for bi, bt in enumerate(audio.beat_times):
        pulse = audio.beat_pulse(bt)
        if pulse >= beat_zoom_threshold:
            zoom_beats_times.append(bt)
    zoom_beats_idx = 0  # pointer to next beat in the main loop
    # keep every n-th beat:
    n = track_config.SKIP_BEAT
    zoom_beats_times = zoom_beats_times[::n]  

    ###########################################################
    #### preprocessing for changing scenes to pulse  ##########
    ###########################################################
    from scipy.signal import find_peaks
    
    strong_pulse_thresh = track_config.STRONG_PULSE_THRESH
    total_frames = int(audio.duration * FPS)
    t_array = np.arange(total_frames) / FPS
    pulse_array = np.array([audio.features_at(t)["pulse"] for t in t_array])
    pulse_peaks, _ = find_peaks(pulse_array, height=strong_pulse_thresh)
    pulse_peak_times = t_array[pulse_peaks]

    def clear_feedback_buffers():
        nonlocal need_clear_feedback
        fbo_a.use(); ctx.clear(0.0, 0.0, 0.0, 1.0)
        fbo_b.use(); ctx.clear(0.0, 0.0, 0.0, 1.0)
        need_clear_feedback = False

    ###########################################################
    ####################### MAIN LOOP  ########################
    ###########################################################
    try:
        for frame in range(total_frames):
            if glfw.window_should_close(window):
                break
            glfw.poll_events()

            t = frame / FPS
            f = audio.features_at(t) # loads music features
            kick = float(f["pulse"])
            beat = float(f["beat"])
            smooth_onset = float(f["smooth_onset"])
            onset = float(f["onset"])
            trend = audio.trend_at(t) 
            is_low = (trend == 0)
            is_mid = (trend == 1)
            is_high = (trend == 2)

            # --------------------------
            # ORBITING
            # --------------------------
            orbit_points = orbit_pointsA
            starting_orbiting_speed = track_config.STARTING_ORBITING_SPEED
            scaling_orbiting_speed = track_config.SCALING_ORBITING_SPEED
            speed = starting_orbiting_speed +  scaling_orbiting_speed*smooth_onset  # if zoom is high, orbiting will seem too fast
            orbit_idx = (orbit_idx + speed) % orbit_len
            center_c = orbit_points[int(orbit_idx)]
            c = np.array([center_c.real, center_c.imag], dtype=np.float32)

            # --------------------------
            # PULSATING ZOOM ON BEATS
            # --------------------------
            zoom_prog = 0.0 # zoom 
            if zoom_beats_idx < len(zoom_beats_times):
                beat_t = zoom_beats_times[zoom_beats_idx]

                # define half-width of pulse (time before and after beat)
                if zoom_beats_idx + 1 < len(zoom_beats_times):
                    next_beat = zoom_beats_times[zoom_beats_idx + 1]
                    half_width = (next_beat - beat_t) / 2.0  # can tweak if needed
                else:
                    half_width = 0.5  # fallback

                t_rel = t - beat_t  # time relative to this beat
                if abs(t_rel) <= half_width:
                    # cosine pulse: 0 → 1 → 0, peaks at t_rel = 0 (the beat)
                    zoom_prog = 0.5 * (1 + math.cos(math.pi * t_rel / half_width))  # 1 at beat, 0 at edges
                elif t > beat_t + half_width:
                    zoom_beats_idx += 1  # move to next beat

            # compute interpolated logZoom
            logZoom = lerp(sceneA["logZoom0"], sceneA["logZoom1"], zoom_prog)

            ###########################################################################
            ##############################   SCENE CHANGE   ###########################
            ###########################################################################
            # based on strong onset:
            dt = 1.0 / FPS
            time_since_last_scene += dt
            PEAK_WINDOW = 0.020  # 20 ms tolerance

            scene_cooldown_sec_low = track_config.COOLDOWN_LOW 
            scene_cooldown_sec_mid = track_config.COOLDOWN_MID 
            scene_cooldown_sec_high = track_config.COOLDOWN_HIGH 

            if is_low:
                allow = (
                        #onset >= strong_onset_in_low and
                        any(abs(t - pt) <= PEAK_WINDOW for pt in pulse_peak_times) and
                         time_since_last_scene >= scene_cooldown_sec_low)
                         #and kick >= STRONG_PULSE_THRESH ) - usually no pulse in slow parts of the music

            elif is_mid:
                allow = (
                        #onset >= strong_onset_in_mid and
                        any(abs(t - pt) <= PEAK_WINDOW for pt in pulse_peak_times) and
                         time_since_last_scene >= scene_cooldown_sec_mid
                         and kick >= strong_pulse_thresh )

            elif is_high:
                allow = (
                        #onset >= strong_onset_in_high and
                        any(abs(t - pt) <= PEAK_WINDOW for pt in pulse_peak_times) and
                         time_since_last_scene >= scene_cooldown_sec_high
                         and kick >= strong_pulse_thresh )

            else:
                allow = False

            if allow:
                # ---- SCENE CHANGE HERE ----
                print(f"Scene change at f {t:7.3f}, because in: ", is_low, is_mid, is_high)
                print(f"[SCENE CHANGE] t={t:.3f}s, beat={bi}, old scene_index={scene_index}")
                sceneA = sceneB
                orbit_points = orbit_pointsB
                scene_index = (scene_index + 1) % len(JULIA_PRESETS)
                sceneB = get_scene(JULIA_PRESETS, scene_index, rng)
                print(f"[SCENE CHANGE] t={t:.3f}s, beat={bi}, new scene_index={scene_index}")
                scene_mix = 0.0
                time_since_last_scene = 0.0

            # interpolate
            scene_mix = np.clip(scene_mix + scene_interp_speed, 0.0, 1.0)
            m = scene_mix

            # interpolate parameters
            center = (1.0 - m) * sceneA["center"] + m * sceneB["center"]

            base_logZoom   = (1.0 - scene_mix) * sceneA["logZoom0"] + scene_mix * sceneB["logZoom0"] 
            target_logZoom = (1.0 - scene_mix) * sceneA["logZoom1"] + scene_mix * sceneB["logZoom1"] 
            logZoom = lerp(base_logZoom, target_logZoom, zoom_prog) 

            ###########################################################################
            ##############################   COLOR CHANGE   ###########################
            ###########################################################################

            # beat position
            bi, bphase = audio.beat_index_and_phase(t)

            # ---- tier smoothing ----
            tier_raw = int(f.get("trend", 0))
            tier_raw = 0 if tier_raw < 0 else (2 if tier_raw > 2 else tier_raw)

            tier_smooth = float(tier_smooth + TIER_EMA * (float(tier_raw) - tier_smooth))
            tier01 = float(np.clip(tier_smooth / 2.0, 0.0, 1.0))

            if tier_state == 0:
                if tier_smooth > (0.50 + TIER_HYST):
                    tier_state = 1
            elif tier_state == 1:
                if tier_smooth < (0.50 - TIER_HYST):
                    tier_state = 0
                elif tier_smooth > (1.50 + TIER_HYST):
                    tier_state = 2
            else:
                if tier_smooth < (1.50 - TIER_HYST):
                    tier_state = 1


            # console tier per second
            sec = int(t)
            if sec != last_logged_sec:
                last_logged_sec = sec
                tier_name = "LOW" if tier_state == 0 else ("MID" if tier_state == 1 else "HIGH")
                print(f"[TIER] {sec:4d}s -> {tier_name}")

            # ---- per-beat events ----
            if bi != last_checked_beat:
                last_checked_beat = bi
                if tier_state != _prev_tier_state:
                    pending_pal_change = True
                    pending_pal_start_beat = bi + 1

                _prev_tier_state = tier_state

            # global speedup pacing
            beat_speed_influence = track_config.BEAT_SPEED_INFLUENCE
            beat_speed = 1.06 + beat_speed_influence * (0.30 * kick + 0.25 * beat)
            beat_speed = float(np.clip(beat_speed * GLOBAL_SPEEDUP, 1.00, 1.28))

            # --------------------------
            # Palette transition start (on scheduled beat)
            # --------------------------
            if pending_pal_change and bi >= pending_pal_start_beat:
                pending_pal_change = False

                pal_in_transition = True
                pal_t_beats = 0.0
                pal_from = pal.copy()
                pal_to = pick_pal_for_tier(rng, tier_state, track_config)

                # LOW slower, HIGH faster
                pal_dur_beats = lerp3(10.0, 7.0, 4.0, tier01)

                tier_name = "LOW" if tier_state == 0 else ("MID" if tier_state == 1 else "HIGH")
                print(f"\n[PALLETE CHANGE TRIG] t={t:7.3f}s  beat={bi}  tier={tier_name}  {pal_from[0]:.3f}->{pal_to[0]:.3f}")

            # Update palette transition
            if pal_in_transition:
                beats_per_sec = max(1e-6, f["tempo"] / 60.0)
                pal_t_beats += (1.0 / FPS) * beats_per_sec * beat_speed
                u = float(np.clip(pal_t_beats / max(1e-6, pal_dur_beats), 0.0, 1.0))
                uu = smoothstep(u)
                pal = (1.0 - uu) * pal_from + uu * pal_to
                if u >= 1.0:
                    pal_in_transition = False
                    pal = pal_to.copy()

            ###########################################################################
            ##############################   DRAW           ###########################
            ###########################################################################

            # PASS 1: fractal
            fbo_curr.use()
            ctx.viewport = (0, 0, RW, RH)
            ctx.clear(0.0, 0.0, 0.0, 1.0)

            U_fr.set("u_time", float(t))
            U_fr.set("u_resolution", (float(RW), float(RH)))
            U_fr.set("u_volume", float(f["volume"]))
            U_fr.set("u_c", (float(c[0]), float(c[1])))
            U_fr.set("u_center", (float(center[0]), float(center[1])))
            U_fr.set("u_logZoom", float(logZoom))
            U_fr.set("u_pal", (float(pal[0]), float(pal[1]), float(pal[2])))

            vao_fr.render(moderngl.TRIANGLE_STRIP)

            if need_clear_feedback:
                clear_feedback_buffers()

            # PASS 2: feedback (smooth by tier)
            next_fbo.use()
            ctx.viewport = (0, 0, RW, RH)

            prev_fb.use(location=0)
            curr_tex.use(location=1)
            U_fb.set("u_prev", 0)
            U_fb.set("u_curr", 1)

            base_amount = lerp3(LOW_FB_AMOUNT, MID_FB_AMOUNT, HIGH_FB_AMOUNT, tier01)
            base_decay  = lerp3(LOW_FB_DECAY,  MID_FB_DECAY,  HIGH_FB_DECAY,  tier01)

            kick_push = lerp3(0.030, 0.055, 0.085, tier01) * kick
            amount = float(np.clip(base_amount + kick_push, 0.03, 0.18))

            decay_drop = lerp3(0.0030, 0.0048, 0.0070, tier01) * kick
            decay = float(np.clip(base_decay - decay_drop, 0.978, 0.994))

            U_fb.set("u_decay", decay)
            U_fb.set("u_amount", amount)
            U_fb.set("u_kick", float(kick))

            vao_fb.render(moderngl.TRIANGLE_STRIP)

            # PASS 3: downsample + sharpen
            fbo_out.use()
            ctx.viewport = (0, 0, OUT_W, OUT_H)

            next_fb.use(location=0)
            U_cp.set("u_tex", 0)
            U_cp.set("u_texel", (1.0 / float(RW), 1.0 / float(RH)))
            vao_cp.render(moderngl.TRIANGLE_STRIP)

            # Preview
            ctx.screen.use()
            ctx.viewport = (0, 0, OUT_W, OUT_H)
            out_tex.use(location=0)
            U_cp.set("u_tex", 0)
            U_cp.set("u_texel", (1.0 / float(OUT_W), 1.0 / float(OUT_H)))
            vao_cp.render(moderngl.TRIANGLE_STRIP)
            glfw.swap_buffers(window)

            # Read output
            raw = fbo_out.read(components=3, alignment=1)
            img = np.frombuffer(raw, dtype=np.uint8).reshape((OUT_H, OUT_W, 3))
            img = np.flipud(img)


            writer.add_frame(img)

            prev_fb, next_fb = next_fb, prev_fb
            prev_fbo, next_fbo = next_fbo, prev_fbo

            if frame % (FPS * 2) == 0:
                pct = 100.0 * frame / max(1, total_frames)
                tier_name = "LOW " if is_low else ("MID " if is_mid else "HIGH")
                print(
                    f"\rRender: {pct:6.2f}%  frame {frame}/{total_frames}  {tier_name}  ",
                    end=""
                )

        print("\nRender končan.")
    finally:
        writer.close()
        glfw.terminate()

    ok = mux_audio(out_noaudio, audio_file, out_final)
    if ok:
        print(f"Shranjeno: {out_final}")
    else:
        print(f"Shranjeno (brez audio): {out_noaudio}")


if __name__ == "__main__":
    main()
