import numpy as np
import librosa


class GetFeatures():
    def __init__(self):
        pass

    def get_mean_zero_cross_rate(self, y):
        return np.mean(librosa.feature.zero_crossing_rate(y=y), axis=1)

    def get_mean_rms(self, y):
        """ 
        Compute root-mean-square (RMS) value for each frame
        spectrogram will give a more accurate representation of energy over time because its frames can be windowed 
        """
        # magphase separates a complex-valued spectrogram D into its magnitude (S) and phase (P) components, so that D = S * P
        S, phase = librosa.magphase(librosa.stft(y))
        rms = librosa.feature.rms(S=S)
        return np.mean(rms, axis=1)

    def get_chroma_stft(self, y, sr):
        """
        Compute a chromagram from a waveform or power spectrogram.
        Use energy (magnitude) spectrum 
        """
        S = np.abs(librosa.stft(y))
        chroma = librosa.feature.chroma_stft(S=S, sr=sr)
        return np.mean(chroma, axis=1)

    def get_chroma_cqt(self, y, sr):
        """ Constant-Q chromagram """
        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
        return np.mean(chroma_cq, axis=1)

    def get_chroma_chroma_cens(self, y, sr):
        """ Chroma Energy Normalized """
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        return np.mean(chroma_cens, axis=1)

    def get_chroma_vqt(self, y, sr):
        chroma_vq = librosa.feature.chroma_vqt(y=y, sr=sr, intervals='ji5')
        return np.mean(chroma_vq, axis=1)

    def get_mel_scaled_spectrogram(self, y, sr):
        s = librosa.feature.melspectrogram(y=y, sr=sr)
        return np.mean(s, axis=1)

    def get_mfcc(self, y, sr):
        mcff = librosa.feature.mfcc(y=y, sr=sr)
        return np.mean(mcff, axis=1)

    def get_spectral_centroid(self, y, sr):
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        return np.mean(cent, axis=1)

    def get_spectral_bandwidth(self, y, sr):
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        return np.mean(spec_bw, axis=1)

    def get_spectral_contrast(self, y, sr):
        S = np.abs(librosa.stft(y))
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        return np.mean(contrast, axis=1)

    def get_spectral_flatness(self, y):
        flatness = librosa.feature.spectral_flatness(y=y)
        return np.mean(flatness, axis=1)

    def get_spectral_rolloff(self, y, sr):
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        return np.mean(rolloff, axis=1)

    def extract_all_features(self, y, sr):
        features = np.array([])
        features = np.hstack((features, self.get_mean_zero_cross_rate(y)))
        features = np.hstack((features, self.get_mean_rms(y)))
        features = np.hstack((features, self.get_chroma_stft(y, sr)))
        features = np.hstack((features, self.get_chroma_cqt(y, sr)))
        features = np.hstack((features, self.get_chroma_chroma_cens(y, sr)))
        features = np.hstack((features, self.get_chroma_vqt(y, sr)))
        features = np.hstack((features, self.get_mel_scaled_spectrogram(y, sr)))
        features = np.hstack((features, self.get_mfcc(y, sr)))
        features = np.hstack((features, self.get_spectral_centroid(y, sr)))
        features = np.hstack((features, self.get_spectral_bandwidth(y, sr)))
        features = np.hstack((features, self.get_spectral_contrast(y, sr)))
        features = np.hstack((features, self.get_spectral_flatness(y)))
        features = np.hstack((features, self.get_spectral_rolloff(y, sr)))
        return features
