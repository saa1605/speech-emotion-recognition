import librosa.core
import librosa.display
import librosa.feature
import librosa.util
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

data_home = '../../data/audios'

# Read CSV file with information about every recording
audio_df = pd.read_csv('df_iemocap.csv')
print(audio_df.head())
indexNames = audio_df[audio_df['emotion'] == 'xxx'].index

# Delete these row indexes from dataFrame
audio_df.drop(indexNames, inplace=True)
audio_df.reset_index(drop=True, inplace=True)

# size of audios
size = 100506

# Get Average Duration of audios
start = np.array(audio_df['start_time'])
end = np.array(audio_df['end_time'])
durations = end - start
avg_duration = np.mean(durations)

mel_spectrograms = np.zeros((7532, 128, 197))
mel_spectrograms_db = np.zeros((7532, 128, 197))
labels = [None] * 7532


mel_spectrograms = np.zeros((7532, 128, 197))
mel_spectrograms_db = np.zeros((7532, 128, 197))


def foo(data, x):
    global audio_df, size
    array1, array2, labels = data
    filename = audio_df['wav_file'][x]
    offset = audio_df['start_time'][x]
    a, sr = librosa.load(
        f'{data_home}/Session{filename[4]}/{filename[:-5]}.wav', offset=offset, duration=avg_duration)
    if len(a) != size:
        a = np.pad(a, (0, size - len(a)), 'constant')
    mel = np.abs((librosa.feature.melspectrogram(a)))
    mel_db = librosa.power_to_db(mel, ref=np.max)
    array1[x] = mel
    array2[x] = mel_db
    labels[x] = audio_df['emotion'][x]
    return 0


def main(mel_spectrograms, mel_spectrograms_db, labels):
    inputs = list(range(10))
    if __name__ == '__main__':
        Parallel(n_jobs=8, verbose=5)(delayed(foo)((mel_spectrograms, mel_spectrograms_db, labels), i)
                                      for i in inputs)


main(mel_spectrograms, mel_spectrograms_db, labels)

np.save('mel_spectrograms', mel_spectrograms)
np.save('mel_spectrograms_db', mel_spectrograms_db)
