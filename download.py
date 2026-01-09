from pytubefix import YouTube
from pytubefix.cli import on_progress
import os

# fractal video, psychill: https://www.youtube.com/watch?v=EFzr4CYN8o8&list=RDEFzr4CYN8o8&start_radio=1

######################################################################################
################### script for downloading music from youtube (m4a format) ###########
######################################################################################

#url = 'https://www.youtube.com/watch?v=D-Q_BDCJRiY&list=RDD-Q_BDCJRiY' # MIJU
#url = 'https://www.youtube.com/watch?v=j-JlQPIqdj4&list=RDj-JlQPIqdj4'  # trippy psychill Shpongle
#url = 'https://www.youtube.com/watch?v=5fSaq4c-Y28&t=328s' # The Mystery of the Yeti Part 2 - whole album, take: 31.22 - 46.27
#url = 'https://www.youtube.com/watch?v=gMsAz7HVuFA' # cel komad Total Eclipse - Freefallin Upwards
#url = 'https://www.youtube.com/watch?v=JkyaQnLEjto&list=RDD-Q_BDCJRiY&index=12'
#url = 'https://www.youtube.com/watch?v=OlTWpTdqpRc'
#url = 'https://www.youtube.com/watch?v=nK1RPXUQK_U&list=RDnK1RPXUQK_U' # Zen Baboon - Odd lion
#url = 'https://www.youtube.com/watch?v=lMzyg9p35UM&list=RDlMzyg9p35UM' # Hair! - Hashish (Encapsulate Remix) 
#url = 'https://www.youtube.com/watch?v=u6uZIWZ_yLc&list=PLWF8VW76sjDuzZVp21s4XsM9Wx8yxw-Zs&index=4' # Topology of a Fat Rabbit
#url = 'https://www.youtube.com/watch?v=xDOTusFU_ko' # Zen Racoon - Novo Ace (Panama Mix)
#url = 'https://www.youtube.com/watch?v=G9vjPdv9ilQ' # Zen Racoon - Novotov
#url = 'https://www.youtube.com/watch?v=F8f80VUySMg&list=RDF8f80VUySMg'
#url = 'https://www.youtube.com/watch?v=s6q9HtooQWE&list=RDF8f80VUySMg&index=4'
url = 'https://www.youtube.com/watch?v=fcAZSy-W0gA&list=RDF8f80VUySMg&index=28' # Shpongle - Sixth Revelation
yt = YouTube(url, on_progress_callback=on_progress)

def best_audio_itag():
    """Return the itag of the highest bitrate audio stream."""
    max_audio = 0
    audio_value = None
    for audio_stream in yt.streams.filter(only_audio=True):
        abr = int(audio_stream.abr.replace('kbps', ''))
        if abr > max_audio:
            max_audio = abr
            audio_value = audio_stream.itag
    if audio_value is None:
        raise ValueError("No audio streams found")
    return audio_value

# Get best audio
audio_itag = best_audio_itag()

save_folder = "music"
os.makedirs(save_folder, exist_ok=True)

# Download audio
yt.streams.get_by_itag(audio_itag).download(output_path=save_folder)
print("Audio downloaded successfully!")
