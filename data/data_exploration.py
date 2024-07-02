import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from IPython.display import Audio


class DataVisualization():
    def __init__(self) -> None:
        pass

    def show_waveplot(y, sr, ax):
        librosa.display.waveshow(y, sr=sr, ax=ax[0])
    
    def show_spectrogram(y, sr, ax):
        stft = np.abs(librosa.stft(y))
        db = librosa.amplitude_to_db(stft, ref=np.max)
        librosa.display.specshow(db, y_axis="linear", x_axis="time", sr=sr, ax=ax[1])
        plt.show()


def main(file_path):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=True)
    y, sr = librosa.load(file_path)
    data_viz = DataVisualization()
    data_viz.show_waveplot(y, sr, ax)
    data_viz.show_spectrogram(y, sr, ax)
    Audio(file_path)
