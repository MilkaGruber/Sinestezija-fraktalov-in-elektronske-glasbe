# fractal_julia_feedback_rhythm_record.py
#
# Uporaba:
#   py -3.10 fractal_julia_feedback_rhythm_record.py audio/psy.wav
#
# Odvisnosti:
#   py -3.10 -m pip install numpy glfw moderngl librosa imageio imageio-ffmpeg
#
# Kaj dela:
# - Samo Julia množice
# - Offline render (frame -> t = frame/FPS) -> stabilen sync
# - Feedback / trails (ping-pong FBO) za psy “afterimage”
# - Premikanje je ritem-driven:
#     hitrejši BPM => hitreje se spreminja (c-orbit + feedback warp + zoom)
#     kick/pulse => dodatni pospeški
# - Shrani MP4 in na koncu poskusi dodati audio (ffmpeg iz PATH ali imageio-ffmpeg)

# run in terminal with: py -3.11 working_julia_4.py audio\Shpongle-TheSixthRevelation[Visualization].wav

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


def frame_is_too_flat(img_uint8_hwc, step=32, var_thresh=35.0):
    """
    Detects near-uniform frames (e.g. only background color).
    Uses a cheap downsample + luminance variance.
    """
    sm = img_uint8_hwc[::step, ::step, :].astype(np.float32)
    lum = 0.2126 * sm[..., 0] + 0.7152 * sm[..., 1] + 0.0722 * sm[..., 2]
    return float(lum.var()) < var_thresh

# =========================
# AUDIO ANALYZER (offline + global trend detection)
# =========================
class AudioAnalyzer:
    def __init__(self, filename: str):
        import librosa
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import savgol_filter

        self.filename = filename
        self.y, self.sr = librosa.load(filename, sr=None, mono=True)
        self.duration = float(librosa.get_duration(y=self.y, sr=self.sr))

        self.hop = 512

        # --- Energy + spectrum ---
        self.rms = librosa.feature.rms(y=self.y, hop_length=self.hop)[0].astype(np.float32)
        self.spec = np.abs(librosa.stft(self.y, hop_length=self.hop)).astype(np.float32)
        self.freqs = librosa.fft_frequencies(sr=self.sr).astype(np.float32)

        # --- Onset envelope ---
        oe = librosa.onset.onset_strength(y=self.y, sr=self.sr, hop_length=self.hop)
        oe = oe.astype(np.float32)
        oe -= oe.min()
        if oe.max() > 1e-9:
            oe /= oe.max()
        self.onset_env = oe

        # --- Beat & tempo ---
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

        # --- Loudness normalization ---
        self.rms_p90 = float(np.percentile(self.rms, 90))
        self.rms_p70 = float(np.percentile(self.rms, 70))

        # ============================
        # GLOBAL TREND ANALYSIS
        # ============================

        def detect_trends(onset_env, sr, hop, window_sec=10.0):
            hop_t = hop / sr
            w = int(window_sec / hop_t)
            if w % 2 == 0:
                w += 1
            w = max(w, 3)

            # smooth onset
            sm = savgol_filter(onset_env, w, 3, mode="mirror")

            # adaptive thresholds
            p33 = np.percentile(sm, 33)
            p66 = np.percentile(sm, 66)

            # long-term trend
            trend = gaussian_filter1d(sm, sigma=w / 6.0)
            avg = np.convolve(trend, np.ones(w) / w, mode="same")

            labels = np.zeros_like(avg, dtype=np.uint8)
            labels[avg >= p33] = 1   # medium
            labels[avg >= p66] = 2   # high

            print(f"[Trends] Low/Med={p33:.3f}  Med/High={p66:.3f}")
            return labels

        self.trend_labels = detect_trends(self.onset_env, self.sr, self.hop)

        # ============================
        # EMA smoothing state
        # ============================
        self._state = {
            "volume": 0.0,
            "low": 0.0,
            "mid": 0.0,
            "high": 0.0,
            "beat": 0.0,
        }

        self.alpha_feat = 0.18
        self.alpha_beat = 0.30

    # --------------------
    # helpers
    # --------------------
    @staticmethod
    def _ema(prev, x, a):
        return prev * (1.0 - a) + x * a

    def _idx(self, t):
        idx = int(t * self.sr / self.hop)
        return int(np.clip(idx, 0, len(self.rms) - 1))

    # --------------------
    # trend at time
    # --------------------
    def trend_at(self, t: float) -> int:
        idx = self._idx(t)
        return int(self.trend_labels[min(idx, len(self.trend_labels) - 1)])

    # --------------------
    # beat pulse
    # --------------------
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

    # --------------------
    # audio features
    # --------------------
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

        # loudness normalization
        vol = np.clip(rms / (self.rms_p70 + 1e-9), 0.0, 2.0)
        vol = float(np.sqrt(np.clip(vol, 0.0, 1.0)))

        def soft_norm(x, k):
            v = np.clip(x / (k + 1e-9), 0.0, 3.0)
            return float(np.sqrt(np.clip(v, 0.0, 1.0)))

        low_n = soft_norm(low, 50.0)
        mid_n = soft_norm(mid, 40.0)
        high_n = soft_norm(high, 30.0)

        onset = float(self.onset_env[idx]) if idx < len(self.onset_env) else 0.0
        gate = 1.0 if rms > self.rms_p90 else 0.0
        beat_raw = float(np.clip(0.60 * onset + 0.40 * gate, 0.0, 1.0))

        # EMA smoothing
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
            "trend": self.trend_at(t),   # 0 = low, 1 = medium, 2 = high
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

# PASS 1: render current Julia to curr_tex
FRACTAL_FS = r"""
#version 330
uniform vec2  u_resolution;
uniform float u_time;
uniform float u_volume;
uniform float u_low;
uniform float u_mid;
uniform float u_high;
uniform float u_beat;
uniform float u_pulse;

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

// returns (t, escaped, glow)
vec3 juliaSmooth(vec2 z, vec2 c, int max_iter){
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
            return vec3(t, 1.0, glow);
        }
    }
    float glow = exp(-2.2 * trap);
    return vec3(1.0, 0.0, glow);
}

vec3 shade(vec3 r, vec3 palHSV){
    float t = r.x;
    float escaped = r.y;
    float glow = r.z;

    float kick = saturate(u_pulse);
    float beat = saturate(u_beat);

    // --- base palette dynamics (same as before) ---
    float hue = fract(palHSV.x + 0.62*t + 0.05*beat + 0.06*sin(0.06*u_time + 2.0*u_high));
    float sat = saturate(palHSV.y + 0.25*u_mid + 0.12*kick);
    float val = saturate(palHSV.z + 0.30*u_volume + 0.10*kick);

    // ---- CONTRAST LOCK START ----
    float hue_fr = hue;

    // try a background hue offset, but enforce minimum distance
    float hue_bg = fract(hue_fr + 0.33 + 0.10*u_mid);

    float dh = abs(hue_bg - hue_fr);
    dh = min(dh, 1.0 - dh);                 // circular hue distance [0..0.5]

    // if too close -> push BG toward complementary
    if(dh < 0.22){
        hue_bg = fract(hue_fr + 0.50);
    }

    vec3 fr = hsv2rgb(vec3(hue_fr, sat, val));
    vec3 bg = hsv2rgb(vec3(hue_bg, saturate(sat*0.85), saturate(val*0.90)));

    // luminance check
    float Lfr = dot(fr, vec3(0.2126, 0.7152, 0.0722));
    float Lbg = dot(bg, vec3(0.2126, 0.7152, 0.0722));
    float dL  = abs(Lfr - Lbg);

    // if luminance too similar -> force separation:
    // make background darker and/or fractal brighter
    if(dL < 0.22){
        float dir = (Lfr > Lbg) ? 1.0 : -1.0;
        // darken bg strongly
        bg *= (dir > 0.0) ? 0.55 : 0.70;
        // brighten fr a bit
        fr = mix(fr, vec3(1.0), 0.18);
    }

    // outline color adapts to bg luminance: bright bg -> black outline, dark bg -> white outline
    float Lbg2 = dot(bg, vec3(0.2126, 0.7152, 0.0722));
    vec3 outlineCol = (Lbg2 > 0.55) ? vec3(0.0) : vec3(1.0);

    // ---- CONTRAST LOCK END ----

    vec3 ink = vec3(0.02, 0.02, 0.03);
    float edge = pow(saturate(1.0 - t), 2.1);

    float outline = smoothstep(0.02, 0.20, edge) * smoothstep(1.0, 0.65, t);

    float strobe = 0.25 + 0.75*kick;
    vec3 glowCol = hsv2rgb(vec3(fract(hue_fr + 0.12), 1.0, 1.0)) * (0.25 + 0.75*strobe);

    vec3 col = mix(ink, ink + 0.06*bg, 0.35);

    if (escaped > 0.5) {
        col = fr + glowCol * edge * (0.20 + 0.70*glow);
        col = mix(col, outlineCol, 0.35 * outline);
    } else {
        col = col + 0.05*bg*(0.25 + 0.75*strobe);
    }

    return col;
}



void main(){
    vec2 uv = (v_uv - 0.5) * 2.0;
    uv.x *= u_resolution.x / u_resolution.y;

    float scale = exp(u_logZoom);

    // tiny drift only
    float driftAmp = 0.008 * scale;
    vec2 drift = driftAmp * vec2(
        sin(0.10*u_time + 2.0*u_mid),
        cos(0.09*u_time + 2.0*u_low)
    );

    vec2 z = uv * scale + u_center + drift;

    float zoomFactor = saturate((-u_logZoom - 0.15) / 4.0);
    int max_iter = int(260.0 + 520.0*zoomFactor + 120.0*u_volume);

    vec3 r = juliaSmooth(z, u_c, max_iter);
    vec3 col = shade(r, u_pal);
    fragColor = vec4(col, 1.0);
}
"""

# PASS 2: feedback / trails
FEEDBACK_FS = r"""
#version 330
uniform sampler2D u_prev;
uniform sampler2D u_curr;

uniform float u_decay;    // 0.94..0.99
uniform float u_amount;   // 0.10..0.50
uniform float u_kick;     // 0..1
uniform float u_motion;   // 0.4..3.0
uniform vec2  u_res;

in vec2 v_uv;
out vec4 fragColor;

// --- no rotation function needed anymore ---

void main(){
    vec2 uv = v_uv;

    vec2 p = uv - 0.5; // center coordinates
    float m = clamp(u_motion, 0.4, 3.0);

    // micro-zoom scale with motion + kick
    float z = 1.0 - (0.003 + 0.010*u_kick) * m;
    p = p * z;  // scale only, no rotation

    // swirl grows with radius + motion
    float r = length(p);
    p += (0.0018 + 0.0060*u_kick) * m * vec2(-p.y, p.x) * (0.6 + 1.2*r);

    vec2 uv_prev = p + 0.5;

    vec3 prev = texture(u_prev, uv_prev).rgb * u_decay;
    vec3 curr = texture(u_curr, uv).rgb;

    vec3 outc = mix(prev, curr, u_amount);
    fragColor = vec4(outc, 1.0);
}
"""

# PASS 3: copy to screen
COPY_FS = r"""
#version 330
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 fragColor;
void main(){
    fragColor = texture(u_tex, v_uv);
}
"""


# =========================
# Safe uniform setter
# =========================
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


# =========================
# Video writer + mux audio
# =========================
class VideoWriter:
    def __init__(self, out_path, fps=60, crf=18):
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
# Julia presets + scenes
# =========================
JULIA_PRESETS = [
    (0.285, 0.01),
    (-0.8, 0.156),
    (-0.7269, 0.1889),
    (-0.4, 0.6),
    (0.355, 0.355),
    (-0.70176, -0.3842),
    (-0.835, -0.2321),
    (0.37, -0.1),
    (-0.54, 0.54),
    (-0.5, 0.974),
        # new ones below
    (-0.123, 0.745),       # tall spiral, good for zoom
    (0.272, -0.654),       # stretched swirl
    (-0.788, 0.15),        # tight dendrite, high detail
    (-0.835, 0.232),       # mirrored dendrite
    (0.45, -0.142),        # soft spiral, subtle orbit
    (-0.6, 0.6),           # diagonal swirl
    (0.32, 0.45),          # open cloud with tiny spirals
    (-0.701, -0.701),      # symmetric spiral, good infinite zoom
    (-0.5, 0.0),           # “classic” Julia shape
    (0.0, 0.8),            # extreme imaginary, long tendrils
        # extra chaotic / alien shapes
    (-0.21, 0.72),        # insect-like, very organic
    (-0.33, 0.61),        # thick tendrils, good glow
    (0.18, 0.56),         # airy nebula, slow morph
    (0.41, 0.27),         # asymmetric spiral arms
    (-0.62, 0.42),        # jagged filaments, aggressive

    # deep-zoom friendly
    (-0.745, 0.186),      # classic filament boundary
    (-0.79, 0.13),        # high-density fractal edge
    (-0.68, 0.32),        # broken spiral, great motion
    (-0.57, -0.48),      # chaotic bloom
    (-0.92, 0.04),        # very sharp boundary (psy peak)

    # weird / psychedelic
    (0.23, -0.51),        # twisted ribbon
    (0.51, -0.28),        # warped mirror spiral
    (-0.41, -0.67),      # dark void + glow edges
    (-0.29, -0.74),      # long symmetric horns
    (0.06, -0.82),        # extreme imaginary chaos
]

def preset_ok(c, max_inside=0.78):
    """
    Rough test: če je preveč 'inside' pikslov -> pogosto velika praznina.
    max_inside: nižje = strožje (0.70 bolj strogo, 0.82 bolj tolerantno)
    """
    cx, cy = float(c[0]), float(c[1])
    c = (cx, cy)

    # približen aspect shaderja (pri 1920x1080 je ~1.777)
    aspect = 1920 / 1080

    # testiramo na "wide" zoomu, kjer se praznine najhitreje vidijo
    logZoom = math.log(1.6)  # podobno kot tvoj wide range 1.2..2.0
    scale = math.exp(logZoom)

    inside = 0
    total = 44 * 25
    max_iter = 70

    for j in range(25):
        v = (j / 24) * 2.0 - 1.0
        for i in range(44):
            u = (i / 43) * 2.0 - 1.0
            zx = (u * aspect) * scale
            zy = v * scale

            it = 0
            while it < max_iter:
                zx2 = zx*zx - zy*zy + cx
                zy2 = 2.0*zx*zy + cy
                zx, zy = zx2, zy2
                if zx*zx + zy*zy > 256.0:
                    break
                it += 1

            if it >= max_iter:
                inside += 1

    inside_ratio = inside / total
    return inside_ratio <= max_inside

GOOD_PRESETS = [c for c in JULIA_PRESETS if preset_ok(c, max_inside=0.78)]


def rand_scene(rng: np.random.Generator):
    cx, cy = GOOD_PRESETS[int(rng.integers(0, len(GOOD_PRESETS)))]
    c0 = np.array([cx, cy], dtype=np.float32)

    # center near origin -> stable composition
    center = np.array([rng.uniform(-0.22, 0.22), rng.uniform(-0.22, 0.22)], dtype=np.float32)

    # zoom not too deep so you always "see something"
    logZoom0 = float(rng.uniform(math.log(1.2), math.log(2.0)))   # wide
    logZoom1 = float(rng.uniform(math.log(0.06), math.log(0.22))) # deeper

    pal = np.array([rng.uniform(0.0, 1.0), rng.uniform(0.65, 1.0), rng.uniform(0.60, 1.0)], dtype=np.float32)

    return {"c0": c0, "center": center, "logZoom0": logZoom0, "logZoom1": logZoom1, "pal": pal}


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


# =========================
# MAIN
# =========================
def main():
    flat_count = 0
    audio_file = sys.argv[1] if len(sys.argv) > 1 else None
    if not audio_file or not os.path.isfile(audio_file):
        print("Uporaba: py -3.10 fractal_julia_feedback_rhythm_record.py audio/psy.wav")
        sys.exit(1)

    audio = AudioAnalyzer(audio_file)
    print("Audio:", audio_file)
    print("Duration (s):", audio.duration, "Tempo (bpm):", round(audio.tempo, 2))

    FPS = 60
    W, H = 1920, 1080

    base = os.path.splitext(os.path.basename(audio_file))[0]
    out_noaudio = f"{base}_julia_fb_rhythm_noaudio.mp4"
    out_final = f"{base}_julia_fb_rhythm.mp4"

    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
    window = glfw.create_window(W, H, "Julia Feedback (Rhythm) Render", None, None)
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

    # float textures for FBO ping-pong (fast + smooth trails)
    curr_tex = ctx.texture((W, H), components=3, dtype="f1")
    curr_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

    fb_tex_a = ctx.texture((W, H), components=3, dtype="f1")
    fb_tex_b = ctx.texture((W, H), components=3, dtype="f1")
    for t in (fb_tex_a, fb_tex_b):
        t.filter = (moderngl.LINEAR, moderngl.LINEAR)

    fbo_curr = ctx.framebuffer(color_attachments=[curr_tex])
    fbo_a = ctx.framebuffer(color_attachments=[fb_tex_a])
    fbo_b = ctx.framebuffer(color_attachments=[fb_tex_b])

    # init feedback to black
    fbo_a.use(); ctx.clear(0.0, 0.0, 0.0, 1.0)
    fbo_b.use(); ctx.clear(0.0, 0.0, 0.0, 1.0)

    # deterministic RNG
    seed = abs(hash(os.path.abspath(audio_file))) % (2**32)
    rng = np.random.default_rng(seed)

    sceneA = rand_scene(rng)
    sceneB = rand_scene(rng)

    beat_times = audio.beat_times
    beats_per_scene = 16
    next_scene_beat = beats_per_scene

    scene_t = 1.0
    mix = 1.0

    writer = VideoWriter(out_noaudio, fps=FPS, crf=18)
    total_frames = int(math.ceil(audio.duration * FPS))
    start = time.time()

    prev_fb = fb_tex_a
    next_fb = fb_tex_b
    prev_fbo = fbo_a
    next_fbo = fbo_b

    try:
        for frame in range(total_frames):
            if glfw.window_should_close(window):
                break
            glfw.poll_events()
            t = frame / FPS
            f = audio.features_at(t)
            trend = audio.trend_at(t)     # 0 = low, 1 = medium, 2 = high
            is_low = (trend == 0)
            is_mid = (trend == 1)
            is_high = (trend == 2)

            # -------------------------------
            # Tempo & beat
            # -------------------------------
            tempo = float(f["tempo"])
            tempo_norm = max(0.6, min(2.2, tempo / 120.0))
            kick = float(f["pulse"])

            # -------------------------------
            # Motion by trend
            # -------------------------------
            if is_high:
                motion = 0.2 + 0.2 * kick
            elif is_mid:
                motion = (0.55 + 0.65 * kick + 0.25 * f["volume"] + 0.20 * f["high"]) * tempo_norm
                motion = float(np.clip(motion, 0.2, 2.5))
            elif is_low:
                motion = (0.35 + 0.55 * kick + 0.25 * f["volume"] + 0.15 * f["high"]) * tempo_norm
                motion = float(np.clip(motion, 0.2, 1.5))

            # -------------------------------
            # Scene blending
            # -------------------------------
            energy = 0.55 * f["volume"] + 0.25 * f["mid"] + 0.20 * kick
            trans_seconds = float(np.clip(3.2 - 1.4 * energy, 1.8, 3.8))
            mix = min(1.0, mix + (1.0 / (trans_seconds * FPS)))
            m = smoothstep(mix)
            scene_seconds = beats_per_scene * (60.0 / max(1e-6, audio.tempo))
            base_dt = 1.0 / (scene_seconds * FPS)
            scene_t = min(1.0, scene_t + base_dt * (0.85 + 0.55 * tempo_norm + 1.10 * kick))
            ss = smoothstep(scene_t)

            # -------------------------------
            # Beat index for scene switching
            # -------------------------------
            if len(beat_times) > 2:
                beat_count = int(np.searchsorted(beat_times, t))
            else:
                beat_count = int(t * (audio.tempo / 60.0))
            strong_kick = (kick > 0.90) and (f["volume"] > 0.30)
            if ((beat_count >= next_scene_beat and mix >= 0.999) or (strong_kick and mix >= 0.999)):
                sceneA = sceneB
                sceneB = rand_scene(rng)
                scene_t = 0.0
                mix = 0.0
                beats_per_scene = 16 if rng.random() < 0.75 else 24
                next_scene_beat = beat_count + beats_per_scene

            # -------------------------------
            # Scene interpolation
            # -------------------------------
            c0 = (1.0 - m) * sceneA["c0"] + m * sceneB["c0"]
            center = (1.0 - m) * sceneA["center"] + m * sceneB["center"]
            base_zoom = (1.0 - m) * (sceneA["logZoom0"] + (sceneA["logZoom1"] - sceneA["logZoom0"]) * ss) + \
                        m * (sceneB["logZoom0"] + (sceneB["logZoom1"] - sceneB["logZoom0"]) * ss)
            pal = (1.0 - m) * sceneA["pal"] + m * sceneB["pal"]

            # -------------------------------
            # Julia C, center, and logZoom by trend
            # -------------------------------
            if frame == 0:
                logZoom = float(base_zoom)  # initialize first frame

            if is_high:
                # High trend: viral infinite zoom
                # High trend: keep scene c0 but make it evolve (no dead fixed attractor)
                orb = (0.010 + 0.025 * f["high"] + 0.020 * kick)
                spd = (0.6 * tempo_norm + 1.2 * motion)
                c = c0 + np.array([orb * math.cos(spd * t), orb * math.sin(spd * t)], dtype=np.float32)

                # center stays stable, no hard reset to (0,0)
                # center = center  (leave as computed)

                # Continuous zoom, now volume-dependent
                base_speed = 0.012
                zoom_speed = base_speed + 0.030 * f["volume"] + 0.010 * kick  # stronger effect of volume
                logZoom -= zoom_speed
                logZoom = max(logZoom, math.log(0.06))

            elif is_mid:
                # Medium trend: gentle orbit around c0
                orb = (0.015 + 0.030 * f["high"] + 0.020 * kick)
                spd = (0.5 * tempo_norm + 1.0 * motion)
                c = c0 + np.array([
                    orb * math.cos(spd * t),
                    orb * math.sin(spd * t)
                ], dtype=np.float32)

                # Center drift
                scale = math.exp(float(logZoom))
                center += np.array([
                    0.008 * (f["mid"] - 0.5),
                    0.008 * (f["low"] - 0.5)
                ], dtype=np.float32) * float(scale)

                # -------------------------------
                # Add rotation for mid trend
                # -------------------------------
                # rotation angle based on normalized tempo (tempo_norm)
                angle = 0.03 * tempo_norm  # tweak 0.03 for faster/slower rotation
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                x, y = center
                center = np.array([
                    cos_a * x - sin_a * y,
                    sin_a * x + cos_a * y
                ], dtype=np.float32)

                # Follow scene zoom
                logZoom = float(base_zoom)

            else:
                # Low trend: slower orbit, more drift
                orb = (0.010 + 0.020 * f["high"] + 0.015 * kick)
                spd = (0.3 * tempo_norm + 0.8 * motion)
                c = c0 + np.array([
                    orb * math.cos(spd * t),
                    orb * math.sin(spd * t)
                ], dtype=np.float32)

                scale = math.exp(float(logZoom))
                center += np.array([
                    0.014 * (f["mid"] - 0.5),
                    0.014 * (f["low"] - 0.5)
                ], dtype=np.float32) * float(scale)

                logZoom = float(base_zoom)  # follow scene zoom




            # ---------- PASS 1 ----------
            fbo_curr.use()
            ctx.viewport = (0, 0, W, H)
            ctx.clear(0.0, 0.0, 0.0, 1.0)
            U_fr.set("u_time", float(t))
            U_fb.set("u_motion", float(motion))
            U_fr.set("u_resolution", (float(W), float(H)))
            U_fr.set("u_volume", float(f["volume"]))
            U_fr.set("u_low", float(f["low"]))
            U_fr.set("u_mid", float(f["mid"]))
            U_fr.set("u_high", float(f["high"]))
            U_fr.set("u_beat", float(f["beat"]))
            U_fr.set("u_pulse", float(kick))
            U_fr.set("u_c", (float(c[0]), float(c[1])))
            U_fr.set("u_center", (float(center[0]), float(center[1])))
            U_fr.set("u_logZoom", float(logZoom))
            U_fr.set("u_pal", (float(pal[0]), float(pal[1]), float(pal[2])))
            vao_fr.render(moderngl.TRIANGLE_STRIP)
            # ---------- PASS 2 ----------
            next_fbo.use()
            ctx.viewport = (0, 0, W, H)
            prev_fb.use(location=0)
            curr_tex.use(location=1)
            U_fb.set("u_prev", 0)
            U_fb.set("u_curr", 1)
            U_fb.set("u_res", (float(W), float(H)))
            U_fb.set("u_kick", float(kick))
            U_fb.set("u_motion", float(motion))
            if is_high:
                decay = 0.985
                amount = 0.12
            else:
                decay = 0.975 - 0.010 * min(1.0, tempo_norm - 1.0) - 0.012 * kick
                decay = float(np.clip(decay, 0.940, 0.985))
                amount = 0.10 + 0.18 * min(1.0, tempo_norm - 0.6) + 0.28 * kick
                amount = float(np.clip(amount, 0.10, 0.50))
            U_fb.set("u_decay", decay)
            U_fb.set("u_amount", amount)
            vao_fb.render(moderngl.TRIANGLE_STRIP)
            # ---------- PASS 3 ----------
            ctx.screen.use()
            ctx.viewport = (0, 0, W, H)
            next_fb.use(location=0)
            U_cp.set("u_tex", 0)
            vao_cp.render(moderngl.TRIANGLE_STRIP)
            glfw.swap_buffers(window)
            raw = next_fbo.read(components=3, alignment=1)
            img = np.frombuffer(raw, dtype=np.uint8).reshape((H, W, 3))
            img = np.flipud(img)
            # --- Anti-void safeguard: never allow "only background" frames ---
            if frame_is_too_flat(img, step=32, var_thresh=35.0):
                flat_count += 1
            else:
                flat_count = 0

            # if flat for a short streak -> recover
            if flat_count >= 6:  # ~0.1s at 60fps
                # 1) back out of deep zoom (increase logZoom => less deep)
                logZoom = max(logZoom, float(base_zoom) - 0.35)

                # 2) nudge center to escape uniform region
                center += rng.normal(0.0, 0.06, size=2).astype(np.float32) * math.exp(float(logZoom))

                # 3) optionally refresh destination scene if it keeps happening
                sceneB = rand_scene(rng)
                mix = 0.0
                scene_t = 0.0

                flat_count = 0

            writer.add_frame(img)
            prev_fb, next_fb = next_fb, prev_fb
            prev_fbo, next_fbo = next_fbo, prev_fbo
            
            if frame % (FPS * 2) == 0:
                elapsed = time.time() - start
                pct = 100.0 * frame / max(1, total_frames)
                print(f"\rRender: {pct:6.2f}%  frame {frame}/{total_frames}  motion {motion:4.2f}  elapsed {elapsed:6.1f}s", end="")

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
