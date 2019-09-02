import librosa
import numpy as np
import soundfile

from utils import normalize

sampling_rate = 16000
top_db = 20
reduced_ratios = []

for i in range(10):
    audiopath = '../audios/audio_{}.wav'.format(i)
    print(audiopath)
    y, sr = librosa.load(audiopath)
    # Trim the beginning and ending silence
    yt, index = librosa.effects.trim(y, top_db=top_db)
    yt = normalize(yt)

    reduced_ratios.append(len(yt) / len(y))

    # Print the durations
    print(librosa.get_duration(y), librosa.get_duration(yt))
    print(len(y), len(yt))
    target = '../audios/trimed_{}.wav'.format(i)
    soundfile.write(target, yt, sampling_rate)

print('\nreduced_ratio: ' + str(100 - 100 * np.mean(reduced_ratios)))
