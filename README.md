# Sinestezija-fraktalov-in-elektronske-glasbe (Julia Fractal Audio Visualizer)

This project is a visualizer that generates Julia set fractal videos which are in sync with music.
The animation reacts to audio beats, global energy trends, pulse and onsets. 

Resulting animations can be found at: www.youtube.com/@DancingFractals


## Features

- **Audio analysis**
  - RMS volume, spectral bands (low, mid, high)
  - Onsets, beat detection, pulse calculation
  - Trend detection for energy trends (low, mid, high)
- **Dynamic fractal visualization**
  - Julia set fractals with smooth zoom and orbiting effects
  - Beat-synced zoom pulses
  - Color palette transitions based on audio energy tiers

## Dependencies

- Python 3.11
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [librosa](https://librosa.org/) (audio analysis)
- [glfw](https://www.glfw.org/) (OpenGL context/window)
- [ModernGL](https://moderngl.readthedocs.io/) (OpenGL wrapper)
- [imageio](https://imageio.readthedocs.io/) + [imageio-ffmpeg](https://pypi.org/project/imageio-ffmpeg/) (video writing)

## Usage
1. Download your track using `download.py` and convert it with `convert.py`.
2. For shorter clips, use `take_fragment_track.py`.
3. Run `music_analysis.ipynb` to analyze your track and set `TRACK_PRESETS` in `main_julia.py`.
4. Decide on `JULIA_PRESETS` in `julia_preset_explorer.py` (this is like a "slide show" of different scenes for the animation) and import them to `main_julia.py`,
5. Run the visualizer:
   ```bash
   python3.11 main_julia.py <audio_file>
