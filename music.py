import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np

music_file = "audio_samples/MIJUxMKDSL-Metatron(BlissInc.Remix).wav"

y, sr = librosa.load(music_file, sr=None)  # sr=None keeps the original sample rate
print(f"Audio duration: {len(y)/sr:.2f} seconds, Sample rate: {sr} Hz")

## 3️⃣ Plot the waveform
#plt.figure(figsize=(14, 4))
#librosa.display.waveshow(y, sr=sr)
#plt.title('Waveform')
#plt.xlabel('Time (s)')
#plt.ylabel('Amplitude')
#plt.show()
#

## 4️⃣ Compute and plot the spectrogram (STFT)
#D = np.abs(librosa.stft(y))
#plt.figure(figsize=(14, 6))
#librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
#                         sr=sr, y_axis='log', x_axis='time')
#plt.colorbar(format='%+2.0f dB')
#plt.title('Spectrogram (log scale)')
#plt.show()

tempo, beats = librosa.beat.beat_track(y, sr=sr)

# If tempo is an array, take the first element
if isinstance(tempo, np.ndarray):
    tempo = tempo[0]

print(f"Estimated tempo: {tempo:.2f} BPM")
## 6️⃣ Optional: plot chroma features (pitch content)
#chroma = librosa.feature.chroma_stft(y, sr=sr)
#plt.figure(figsize=(14, 4))
#librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
#plt.colorbar()
#plt.title('Chroma Feature (Pitch Class)')
#plt.show()