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

audio_file = r"C:/Users/Milka/Documents/FMF_2025_26/MatematikaZRacunalnikom/Sinestezija-fraktalov-in-elektronske-glasbe/music/MIJU x MKDSL - Metatron (Bliss Inc. Remix) [HEAD008].m4a"
output_file = r"music/MIJUxMKDSL-Metatron(BlissInc.Remix).wav"

audio = AudioSegment.from_file(audio_file)
audio.export(output_file, format="wav")

print("Conversion successful!")

