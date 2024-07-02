import numpy as np
import librosa


class DataAugmentation():
    def __init__(self):
        pass

    def add_white_noise(self, y):
        wn = np.random.randn(len(y))
        data_wn = y + 0.005*wn
        return data_wn

    def compress_audio(self, y):
        y_fast = librosa.effects.time_stretch(y, rate=1.5)
        return y_fast

    def stretch_audio(self, y):
        y_slow = librosa.effects.time_stretch(y, rate=0.75)
        return y_slow

    def increase_pitch(self, y, sr):
        y_third = librosa.effects.pitch_shift(y, sr=sr, n_steps=4)
        return y_third

    def decrease_pitch(self, y, sr):
        y_tritone = librosa.effects.pitch_shift(y, sr=sr, n_steps=-6)
        return y_tritone

    def shift_audio(self, y):
        shift_range = int(np.random.uniform(low=-10, high=10) * 1000)
        return np.roll(y, shift_range)

    def get_all_features(self, file_path):
        """
        Extract all features of original and augmented audio data
        """
        y, sr = librosa.load(file_path)

        # original audio
        feature_mat = self.extract_all_features(y, sr)

        # with white noise
        wn_y = self.add_white_noise(y)
        feature_mat = np.vstack((feature_mat, self.extract_all_features(wn_y, sr)))

        # compress_audio (sped up audio)
        sped_up_y = self.compress_audio(y)
        feature_mat = np.vstack((feature_mat, self.extract_all_features(sped_up_y, sr)))

        # stretch_audio (slowed down audio)
        slowed_down_y = self.stretch_audio(y)
        feature_mat = np.vstack((feature_mat, self.extract_all_features(slowed_down_y, sr)))

        # increase_pitch
        higher_pitch = self.increase_pitch(y, sr)
        feature_mat = np.vstack((feature_mat, self.extract_all_features(higher_pitch, sr)))

        # decrease_pitch
        lower_pitch = self.decrease_pitch(y, sr)
        feature_mat = np.vstack((feature_mat, self.extract_all_features(lower_pitch, sr)))

        # shift_audio
        shifted_audio = self.shift_audio(y)
        feature_mat = np.vstack((feature_mat, self.extract_all_features(shifted_audio, sr)))

        return feature_mat
