from pydub import AudioSegment
import os
from pydub.utils import which

######################################################################################
######################### script for converting to wav format   ######################
######################################################################################

# Tell pydub exactly where ffmpeg is:
AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffmpeg = r"C:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg\bin\ffprobe.exe"

# Force ffmpeg/ffprobe availability
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

music_folder = r"C:/Users/Milka/Documents/FMF_2025_26/MatematikaZRacunalnikom/Sinestezija-fraktalov-in-elektronske-glasbe/music"
output_folder = r"C:/Users/Milka/Documents/FMF_2025_26/MatematikaZRacunalnikom/Sinestezija-fraktalov-in-elektronske-glasbe/audio"

supported_formats = (".mp3", ".m4a", ".flac", ".ogg", ".wav", ".aac", ".wma")

for filename in os.listdir(music_folder):
    if filename.lower().endswith(supported_formats):

        input_path = os.path.join(music_folder, filename)

        # Change extension to .wav
        name_without_ext = os.path.splitext(filename)[0]
        safe_name = name_without_ext.replace(" ", "")
        output_path = os.path.join(output_folder, safe_name + ".wav")

        print(f"Converting: {filename}")

        try:
            audio = AudioSegment.from_file(input_path)
            audio.export(output_path, format="wav")
        except Exception as e:
            print(f"Failed on {filename}: {e}")

print("All conversions finished!")

