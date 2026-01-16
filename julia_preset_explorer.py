import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import math

#### Use this explorer to find c, orbit:

# ============================================================
# Julia 
# ============================================================

@njit(fastmath=True)
def julia_image(w, h, cx, cy, center_x, center_y, logZoom, max_iter=600):
    img = np.zeros((h, w), np.float32)

    scale = math.exp(logZoom) * 0.65
    aspect = w / h

    for j in range(h):
        v = (j / (h - 1)) * 2.0 - 1.0
        for i in range(w):
            u = (i / (w - 1)) * 2.0 - 1.0

            zx = center_x + u * scale * aspect
            zy = center_y + v * scale

            for it in range(max_iter):
                r2 = zx*zx + zy*zy
                if r2 > 256.0:
                    img[j, i] = it / max_iter
                    break

                zx, zy = zx*zx - zy*zy + cx, 2.0*zx*zy + cy

    return img


# ============================================================
# Orbit helper
# ============================================================

def build_orbit_circle(C, P, N):
    v = P - C
    r = abs(v)
    a0 = math.atan2(v.imag, v.real)

    return [
        C + complex(r * math.cos(a0 + 2*math.pi*i/N),
                    r * math.sin(a0 + 2*math.pi*i/N))
        for i in range(N)
    ]


# ============================================================
# Complex pixel conversion (for plotting center)
# ============================================================

def complex_to_pixel(zx, zy, center, logZoom, w, h):
    scale = math.exp(logZoom) * 0.65
    aspect = w / h

    u = (zx - center[0]) / (scale * aspect)
    v = (zy - center[1]) / scale

    x = (u + 1) * 0.5 * (w - 1)
    y = (v + 1) * 0.5 * (h - 1)

    return x, y


# ============================================================
# Main viewer for individual scenes
# ============================================================

def draw_grid(ax, center, logZoom, w, h, spacing=0.1):
    scale = math.exp(logZoom) * 0.65
    aspect = w / h

    xmin = center[0] - scale * aspect
    xmax = center[0] + scale * aspect
    ymin = center[1] - scale
    ymax = center[1] + scale

    # vertical grid lines
    x = math.floor(xmin / spacing) * spacing
    while x <= xmax:
        u = (x - center[0]) / (scale * aspect)
        px = (u + 1) * 0.5 * (w - 1)
        ax.axvline(px, color="white", alpha=0.15)
        x += spacing

    # horizontal grid lines
    y = math.floor(ymin / spacing) * spacing
    while y <= ymax:
        v = (y - center[1]) / scale
        py = (v + 1) * 0.5 * (h - 1)
        ax.axhline(py, color="white", alpha=0.15)
        y += spacing


def show_preset(c, center, zoom, orbit=None, size=(900, 600), N_orbit=128):

    w, h = size
    h2 = h // 2

    # pick orbit c
    if orbit:
        C = complex(*orbit["C"]) 
        P = complex(*orbit["P"])
        orbit_pts = build_orbit_circle(C, P, N_orbit)
        oc = orbit_pts[len(orbit_pts)//2]
        cx, cy = P.real, P.imag
    else:
        cx, cy = c

    print("c =", (cx, cy))

    img0 = julia_image(w, h2, cx, cy, center[0], center[1], zoom[0])
    img1 = julia_image(w, h2, cx, cy, center[0], center[1], zoom[1])

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    ax[0].imshow(img0, cmap="inferno")
    draw_grid(ax[0], center, zoom[0], w, h2, spacing=0.1)
    ax[0].set_title(f"logZoom = {zoom[0]}")
    px, py = complex_to_pixel(center[0], center[1], center, zoom[0], w, h2)
    ax[0].scatter(px, py, c="cyan", s=80, marker="+")
    ax[0].text(px+10, py+10, "center", color="cyan")
    ax[0].axis("off")

    ax[1].imshow(img1, cmap="inferno")
    draw_grid(ax[1], center, zoom[1], w, h2, spacing=0.1)
    ax[1].set_title(f"logZoom = {zoom[1]}")
    px, py = complex_to_pixel(center[0], center[1], center, zoom[1], w, h2)
    ax[1].scatter(px, py, c="cyan", s=80, marker="+")
    ax[1].text(px+10, py+10, "center", color="cyan")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()


def show_all_presets_grid(presets, size=(1080, 1920), N_orbit=128, cmap="inferno"):
    """
    Show all presets in a single figure, 6 plots per row:
    left 3 plots: orbit points with zoom0: how the picture looks zoomed out
    right 3 plots: orbit points with zoom1: how the picture looks zoomed in
    """
    import matplotlib.pyplot as plt

    n_presets = len(presets)
    n_cols = 6  # 3 orbit points left, 3 orbit points right
    fig, axes = plt.subplots(n_presets, n_cols, figsize=(4*n_cols, 3*n_presets))

    # Ensure axes is 2D even if n_presets = 1
    if n_presets == 1:
        axes = axes.reshape(1, n_cols)

    for row, preset in enumerate(presets):
        c = preset["c"]
        center = preset["center"]
        zoom = preset["zoom"]
        orbit = preset.get("orbit", None)

        w, h = size
        h2 = h // 2

        # Orbit points if any
        if orbit:
            C = complex(*orbit["C"])
            P = complex(*orbit["P"])
            orbit_pts = build_orbit_circle(C, P, N_orbit)
            # pick 3 evenly spaced points from the orbit
            idxs = np.linspace(0, len(orbit_pts)-1, 3, dtype=int)
            points = [orbit_pts[i] for i in idxs]
        else:
            points = [complex(*c)]*3 # stays the same

        # Left column: zoom0
        for col, pt in enumerate(points):
            ax = axes[row, col]
            img = julia_image(w, h2, pt.real, pt.imag, center[0], center[1], zoom[0])
            ax.imshow(img, cmap=cmap)
            px, py = complex_to_pixel(center[0], center[1], center, zoom[0], w, h2)
            ax.scatter(px, py, c="cyan", s=50, marker="+")
            ax.axis("off")

        # Right column: zoom1
        for col, pt in enumerate(points):
            ax = axes[row, col+3]  # right half
            img = julia_image(w, h2, pt.real, pt.imag, center[0], center[1], zoom[1])
            ax.imshow(img, cmap=cmap)
            px, py = complex_to_pixel(center[0], center[1], center, zoom[1], w, h2)
            ax.scatter(px, py, c="cyan", s=50, marker="+")
            ax.axis("off")

    plt.tight_layout()
    plt.show()


############################# FOUND PRESETS  ##########################################
JULIA_PRESETS1 = [
    # julia_preset3.png
    {
        "c": (-0.35068, 0.64535),
        "center": (-0.36505707942695353, 0.3695188606158817),
        "zoom": (0.002, 0.15),
        "orbit": {"C": (-0.32803, 0.65185), "P": (-0.33345, 0.68443)}
    },

    # julia_preset4.png
    {
        "c": (-0.35068, 0.64535),
        "center": (-0.36505707942695353, 0.3695188606158817),
        "zoom": (0.0000000005, 0.000002),
        "orbit": {"C": (-0.32803, 0.65185), "P": (-0.33345, 0.68443)}
    },

    # julia_preset5.png
    {
        "c": (-0.35068, 0.64535),
        "center": (-0.36505707942695353, 0.3695188606158817),
        "zoom": (-1.5, -2),
        "orbit": {"C": (-0.32803, 0.65185), "P": (-0.33345, 0.68443)}
    },

    # julia_preset5.png (alternate)
    {
        "c": (-0.35568, 0.64535),
        "center": (-0.36505707942695353, 0.3695188606158817),
        "zoom": (-1.5, -2),
        "orbit": {"C": (-0.32803, 0.65185), "P": (-0.33345, 0.68443)}
    },

    # julia_preset6.png
    {
        "c": (0.28958, -0.1211),
        "center": (-0.5, 0.2),
        "zoom": (-1.5, -2),
        "orbit": {"C": (0.28958, -0.01211), "P": (0.28873, -0.01339)}
    },

    # julia_preset6.png (alternate)
    {
        "c": (0.28958, -0.1211),
        "center": (-0.7, 0.2),
        "zoom": (-1, -2),
        "orbit": {"C": (0.28671, -0.01576), "P": (0.29, -0.02)}
    },

    # julia_preset4.png (final)
    {
        "c": (-0.35068, 0.64535),
        "center": (-0.36505707942695353, 0.3695188606158817),
        "zoom": (-1.5, -2),
        "orbit": {"C": (-0.32803, 0.65185), "P": (-0.33345, 0.68443)}
    }
]

JULIA_PRESETS = [ # the presets used in the first yt upload: https://www.youtube.com/shorts/yTTk5nESxlk
    {
        "c": (-0.36152, 0.6383),
        "center": (-0.36505707942695353, 0.3695188606158817),
        "zoom": (0.002, 0.15),  # before -kind of ok (0.002, 0.08 first component of zoom - > how zoomed in we are when we begin, second component - target zoom for pulasting
        "orbit": {
            "C": (-0.36194, 0.62536),
            "P": (-0.36371, 0.63076)
        }
    },
    {
        "c": (-0.35068, 0.64535),
        "center": (-0.36505707942695353, 0.3695188606158817),
        "zoom": (-1.5, -2),  # before -kind of ok (0.002, 0.08 first component of zoom - > how zoomed in we are when we begin, second component - target zoom for pulasting
        "orbit": {
            "C": (-0.32803, 0.65185),
            "P": (-0.33345, 0.68443)
        }
    },
    {
        "c": (-0.35568, 0.64535),
        "center": (-0.38, 0.36),
        "zoom": (-1.5, -2),  # before -kind of ok (0.002, 0.08 first component of zoom - > how zoomed in we are when we begin, second component - target zoom for pulasting
        "orbit": {
            "C": (-0.32803, 0.65185),
            "P": (-0.33345, 0.68443)
        }
    }, 

    {
        "c": (0.28958, -0.1211),
        "center": (-0.7, 0.2),
        "zoom": (-1.5, -2),  # before -kind of ok (0.002, 0.08 first component of zoom - > how zoomed in we are when we begin, second component - target zoom for pulasting
        "orbit": {
            "C": (0.28671, -0.01576),
            "P": (0.28873, -0.01339)
        }
    }, 
    
]
show_all_presets_grid(JULIA_PRESETS)


# =========================
# OLD JULIA PRESETS
# =========================
# JULIA_PRESETS = [
#     (0.285, 0.01), (-0.8, 0.156), (-0.7269, 0.1889), (-0.4, 0.6),
#     (0.355, 0.355), (-0.70176, -0.3842), (-0.835, -0.2321), (0.37, -0.1),
#     (-0.54, 0.54), (-0.5, 0.974), (-0.123, 0.745), (0.272, -0.654),
#     (-0.788, 0.15), (-0.835, 0.232), (0.45, -0.142), (-0.6, 0.6),
#     (0.32, 0.45), (-0.701, -0.701), (-0.5, 0.0), (0.0, 0.8),
#     (-0.21, 0.72), (-0.33, 0.61), (0.18, 0.56), (0.41, 0.27),
#     (-0.62, 0.42), (-0.745, 0.186), (-0.79, 0.13), (-0.68, 0.32),
#     (-0.57, -0.48), (-0.92, 0.04), (0.23, -0.51), (0.51, -0.28),
#     (-0.41, -0.67), (-0.29, -0.74), (0.06, -0.82),
# ]
