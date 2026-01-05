# fractal_audio_gpu.py

import sys
import time
import math
import numpy as np
import glfw
import moderngl


class AudioAnalyzer:
    def __init__(self, filename=None):
        self.enabled = filename is not None
        self.sr = 44100
        self.duration = 0.0

        if self.enabled:
            import librosa
            self.y, self.sr = librosa.load(filename, sr=None, mono=True)
            self.duration = librosa.get_duration(y=self.y, sr=self.sr)

            self.hop = 512
            self.rms = librosa.feature.rms(y=self.y, hop_length=self.hop)[0]
            self.spec = np.abs(librosa.stft(self.y, hop_length=self.hop))
            self.freqs = librosa.fft_frequencies(sr=self.sr)
        else:
            self.duration = 9999.0

    def features_at(self, t):
        if not self.enabled:
            # fallback animacija
            return {
                "volume": 0.5 + 0.5 * math.sin(t * 1.2),
                "low": 0.5,
                "mid": 0.5,
                "high": 0.5,
                "beat": int(math.sin(t * 2.0) > 0.9),
            }

        idx = int(t * self.sr / self.hop)
        idx = np.clip(idx, 0, len(self.rms) - 1)

        rms = self.rms[idx]
        spec = self.spec[:, idx]

        low = spec[self.freqs < 200].mean() if np.any(self.freqs < 200) else 0
        mid = spec[(self.freqs > 200) & (self.freqs < 2000)].mean()
        high = spec[self.freqs > 2000].mean()

        return {
            "volume": float(np.clip(rms * 5, 0, 1)),
            "low": float(np.clip(low / 50, 0, 1)),
            "mid": float(np.clip(mid / 50, 0, 1)),
            "high": float(np.clip(high / 50, 0, 1)),
            "beat": int(rms > np.percentile(self.rms, 90)),
        }


# =========================
# SHADERS
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

FRAGMENT_SHADER = """
#version 330

uniform vec2 u_resolution;
uniform float u_time;
uniform float u_volume;
uniform float u_low;
uniform float u_mid;
uniform float u_high;
uniform int u_beat;

in vec2 v_uv;
out vec4 fragColor;

void main() {

    vec2 uv = (v_uv - 0.5) * 2.0;
    uv.x *= u_resolution.x / u_resolution.y;

    float zoom = 0.8 + pow(u_volume, 0.5) * 1.0;
    uv *= zoom;

    vec2 c = vec2(
        cos(u_time * 0.04) * (0.55 + u_low * 0.3),
        sin(u_time * 0.03) * (0.55 + u_high * 0.3)
    );

    vec2 z = uv;
    int max_iter = int(220.0 + u_mid * 500.0);
    int i;

    for (i = 0; i < max_iter; i++) {
        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        if (dot(z, z) > 4.0) break;
    }

    float mu = float(i);
    if (i < max_iter) {
        float log_zn = log(dot(z, z)) / 2.0;
        float nu = log(log_zn / log(2.0)) / log(2.0);
        mu = float(i) + 1.0 - nu;
    }
    float t = mu / float(max_iter);


    vec3 colA = vec3(0.15, 0.35, 0.25); // temno zelena
    vec3 colB = vec3(0.20, 0.55, 0.45); // turkizna
    vec3 colC = vec3(0.15, 0.30, 0.55); // modra

    float k = smoothstep(0.0, 1.0, t);
    vec3 col = mix(colA, colB, k);
    col = mix(col, colC, smoothstep(0.4, 1.0, k));


    float glow = 1.0 + 0.2 * float(u_beat);
    col *= glow;

    float fog = exp(-t * 2.5);
    vec3 bg = vec3(0.03, 0.06, 0.08); // temno, nevtralno ozadje
    col = mix(bg, col, fog);

    fragColor = vec4(col, 1.0);
}
"""




# =========================
# MAIN
# =========================
def main():
    audio_file = sys.argv[1] if len(sys.argv) > 1 else None
    audio = AudioAnalyzer(audio_file)
    print(audio_file)
    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    width, height = 1280, 720
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(width, height, "Audio Reactive Julia (GPU)", None, None)
    glfw.make_context_current(window)

    ctx = moderngl.create_context()
    prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)

    quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
    vbo = ctx.buffer(quad.tobytes())
    vao = ctx.simple_vertex_array(prog, vbo, "in_pos")

    start_time = time.time()

    while not glfw.window_should_close(window):
        glfw.poll_events()

        t = time.time() - start_time
        f = audio.features_at(t)

        prog["u_time"].value = t
        prog["u_resolution"].value = (width, height)
        prog["u_volume"].value = f["volume"]
        prog["u_low"].value = f["low"]
        prog["u_mid"].value = f["mid"]
        prog["u_high"].value = f["high"]
        prog["u_beat"].value = f["beat"]

        ctx.clear()
        vao.render(moderngl.TRIANGLE_STRIP)
        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()
