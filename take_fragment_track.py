from pydub import AudioSegment


def take_fragment(file_name, min_start, sec_start, min_end, sec_end):
# Load the WAV file
    audio = AudioSegment.from_wav(file_name)

    start_ms = (min_start * 60 + sec_start) * 1000   
    end_ms   = (min_end * 60 + sec_end) * 1000   

    # Slice the segment
    segment = audio[start_ms:end_ms]

    # Export the fragment
    new_file_name = file_name[:-4] + '_shorter_segment.wav'
    print(new_file_name)
    segment.export(new_file_name, format="wav")

shpongle = 'audio\Shpongle-TheSixthRevelation[Visualization].wav'

take_fragment(shpongle, 4, 13, 5, 30)