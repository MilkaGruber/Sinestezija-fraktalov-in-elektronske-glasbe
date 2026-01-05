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

import os
import sys
import time
import math
import shutil
import subprocess
import numpy as np
import glfw
import moderngl


# =========================
# AUDIO ANALYZER (offline)
# =========================
class AudioAnalyzer:
    def __init__(self, filename: str):
        import librosa

        self.filename = filename
        self.y, self.sr = librosa.load(filename, sr=None, mono=True)
        self.duration = float(librosa.get_duration(y=self.y, sr=self.sr))

        self.hop = 512
        self.rms = librosa.feature.rms(y=self.y, hop_length=self.hop)[0].astype(np.float32)
        self.spec = np.abs(librosa.stft(self.y, hop_length=self.hop)).astype(np.float32)
        self.freqs = librosa.fft_frequencies(sr=self.sr).astype(np.float32)

        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr, hop_length=self.hop)
        oe = onset_env.astype(np.float32)
        oe -= oe.min()
        if oe.max() > 1e-9:
            oe /= oe.max()
        self.onset_env = oe

        tempo, beat_frames = librosa.beat.beat_track(
            y=self.y, sr=self.sr, hop_length=self.hop, units="frames"
        )

        # tempo can be ndarray -> avoid numpy warning
        if tempo is None:
            self.tempo = 140.0
        else:
            tempo_arr = np.asarray(tempo).reshape(-1)
            self.tempo = float(tempo_arr[0]) if tempo_arr.size else 140.0

        if beat_frames is None or len(beat_frames) == 0:
            self.beat_times = np.array([0.0], dtype=np.float32)
        else:
            beat_frames = np.asarray(beat_frames, dtype=np.int32)
            self.beat_times = (beat_frames * self.hop / self.sr).astype(np.float32)

        self.rms_p90 = float(np.percentile(self.rms, 90))
        self.rms_p70 = float(np.percentile(self.rms, 70))

        # smoothing
        self._state = {"volume": 0.0, "low": 0.0, "mid": 0.0, "high": 0.0, "beat": 0.0}
        self.alpha_feat = 0.18
        self.alpha_beat = 0.30

    @staticmethod
    def _ema(prev, x, a):
        return prev * (1.0 - a) + x * a

    def _idx(self, t):
        idx = int(t * self.sr / self.hop)
        return int(np.clip(idx, 0, len(self.rms) - 1))

    def beat_pulse(self, t: float, sigma: float = 0.040) -> float:
        """Gaussian pulse around nearest beat time."""
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

    // keep beat/pulse used
    float kick = saturate(u_pulse);
    float beat = saturate(u_beat);

    float hue = fract(palHSV.x + 0.62*t + 0.05*beat + 0.06*sin(0.06*u_time + 2.0*u_high));
    float sat = saturate(palHSV.y + 0.25*u_mid + 0.12*kick);
    float val = saturate(palHSV.z + 0.30*u_volume + 0.10*kick);

    vec3 bg = hsv2rgb(vec3(hue, sat, val));
    vec3 ink = vec3(0.02, 0.02, 0.03);
    float edge = pow(saturate(1.0 - t), 2.1);

    float strobe = 0.25 + 0.75*kick;
    vec3 glowCol = hsv2rgb(vec3(fract(hue + 0.12), 1.0, 1.0)) * (0.25 + 0.75*strobe);

    vec3 col = mix(ink, ink + 0.06*bg, 0.35);

    if(escaped > 0.5){
        col = bg + glowCol * edge * (0.22 + 0.75*glow);
    } else {
        col = col + 0.08*bg*(0.25 + 0.75*strobe);
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

vec2 rot2(vec2 p, float a){
    float c = cos(a), s = sin(a);
    return vec2(c*p.x - s*p.y, s*p.x + c*p.y);
}

void main(){
    vec2 uv = v_uv;

    vec2 p = (uv - 0.5);

    float m = clamp(u_motion, 0.4, 3.0);

    // rotation + micro-zoom scale with motion + kick
    float a = (0.010 + 0.035*u_kick) * m;
    float z = 1.0 - (0.003 + 0.010*u_kick) * m;

    p = rot2(p, a) * z;

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
    (-0.12, 0.74),
]

def rand_scene(rng: np.random.Generator):
    cx, cy = JULIA_PRESETS[int(rng.integers(0, len(JULIA_PRESETS)))]
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

            # ---- RYTHM-DRIVEN MOTION FACTOR ----
            tempo = float(f["tempo"])
            tempo_norm = max(0.6, min(2.2, tempo / 120.0))  # 120 BPM => 1.0
            kick = float(f["pulse"])
            motion = tempo_norm * (0.55 + 0.65 * kick + 0.25 * f["volume"] + 0.20 * f["high"])
            motion = float(np.clip(motion, 0.4, 3.0))

            # beat index for scene switching
            if len(beat_times) > 2:
                beat_count = int(np.searchsorted(beat_times, t))
            else:
                beat_count = int(t * (audio.tempo / 60.0))

            strong_kick = (kick > 0.90) and (f["volume"] > 0.30)

            # change destination occasionally (not too often)
            if ((beat_count >= next_scene_beat and mix >= 0.999) or (strong_kick and mix >= 0.999)):
                sceneA = sceneB
                sceneB = rand_scene(rng)
                scene_t = 0.0
                mix = 0.0
                beats_per_scene = 16 if rng.random() < 0.75 else 24
                next_scene_beat = beat_count + beats_per_scene

            # smooth crossfade
            energy = 0.55*f["volume"] + 0.25*f["mid"] + 0.20*kick
            trans_seconds = float(np.clip(3.2 - 1.4*energy, 1.8, 3.8))
            mix = min(1.0, mix + (1.0 / (trans_seconds * FPS)))
            m = smoothstep(mix)

            # within-scene zoom progression (tempo + kick accelerates zoom)
            scene_seconds = beats_per_scene * (60.0 / max(1e-6, audio.tempo))
            base_dt = 1.0 / (scene_seconds * FPS)
            scene_t = min(1.0, scene_t + base_dt * (0.85 + 0.55 * tempo_norm + 1.10 * kick))
            ss = smoothstep(scene_t)

            # scene interpolate
            c0 = (1.0 - m) * sceneA["c0"] + m * sceneB["c0"]
            center = (1.0 - m) * sceneA["center"] + m * sceneB["center"]
            logZoom = (1.0 - m) * (sceneA["logZoom0"] + (sceneA["logZoom1"] - sceneA["logZoom0"]) * ss) + \
                      m * (sceneB["logZoom0"] + (sceneB["logZoom1"] - sceneB["logZoom0"]) * ss)
            pal = (1.0 - m) * sceneA["pal"] + m * sceneB["pal"]

            # ---- RYTHM-DRIVEN Julia evolution: c(t) orbit ----
            orb = (0.020 + 0.060 * f["high"] + 0.030 * kick)           # radius
            spd = (0.7 * tempo_norm + 1.4 * motion)                    # rad/s
            c = c0 + np.array([orb * math.cos(spd * t), orb * math.sin(spd * t)], dtype=np.float32)

            # tiny center nudging scaled by zoom (still subtle)
            scale = math.exp(float(logZoom))
            center = center + np.array([0.012*(f["mid"]-0.5), 0.012*(f["low"]-0.5)], dtype=np.float32) * float(scale)

            # ---------- PASS 1: render fractal to curr_tex ----------
            fbo_curr.use()
            ctx.viewport = (0, 0, W, H)
            ctx.clear(0.0, 0.0, 0.0, 1.0)

            U_fr.set("u_time", float(t))
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

            # ---------- PASS 2: feedback mix prev + curr -> next_fb ----------
            next_fbo.use()
            ctx.viewport = (0, 0, W, H)

            prev_fb.use(location=0)
            curr_tex.use(location=1)
            U_fb.set("u_prev", 0)
            U_fb.set("u_curr", 1)
            U_fb.set("u_res", (float(W), float(H)))
            U_fb.set("u_kick", float(kick))
            U_fb.set("u_motion", float(motion))

            # Rythm-driven trails params:
            # faster tempo => less decay (more motion); kick => more injection
            decay = 0.975 - 0.010 * min(1.0, tempo_norm - 1.0) - 0.012 * kick
            decay = float(np.clip(decay, 0.940, 0.985))

            amount = 0.10 + 0.18 * min(1.0, tempo_norm - 0.6) + 0.28 * kick
            amount = float(np.clip(amount, 0.10, 0.50))

            U_fb.set("u_decay", float(decay))
            U_fb.set("u_amount", float(amount))

            vao_fb.render(moderngl.TRIANGLE_STRIP)

            # ---------- PASS 3: preview (copy next_fb to screen) ----------
            ctx.screen.use()
            ctx.viewport = (0, 0, W, H)
            next_fb.use(location=0)
            U_cp.set("u_tex", 0)
            vao_cp.render(moderngl.TRIANGLE_STRIP)
            glfw.swap_buffers(window)

            # Read frame from next_fbo for video
            raw = next_fbo.read(components=3, alignment=1)
            img = np.frombuffer(raw, dtype=np.uint8).reshape((H, W, 3))
            img = np.flipud(img)
            writer.add_frame(img)

            # swap feedback buffers
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
