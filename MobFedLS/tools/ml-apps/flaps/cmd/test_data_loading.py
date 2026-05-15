import os
import librosa
import numpy as np

def load_musdb_track(track_path):
    stems = ['mixture.wav', 'drums.wav', 'bass.wav', 'vocals.wav', 'other.wav']
    audio_data = {}
    for stem in stems:
        path = os.path.join(track_path, stem)
        if os.path.exists(path):
            # Load only 10 seconds for testing
            y, sr = librosa.load(path, sr=22050, duration=10)
            audio_data[stem.replace('.wav', '')] = y
        else:
            print(f"Missing {stem} in {track_path}")
    return audio_data, 22050

def get_spectrogram(y):
    S = librosa.stft(y, n_fft=2048, hop_length=512)
    return np.abs(S)

if __name__ == "__main__":
    base_dir = "musdb18hq/train"
    tracks = os.listdir(base_dir)
    if tracks:
        track_path = os.path.join(base_dir, tracks[0])
        print(f"Loading track: {tracks[0]}")
        audio_data, sr = load_musdb_track(track_path)
        for name, y in audio_data.items():
            spec = get_spectrogram(y)
            print(f"{name} spectrogram shape: {spec.shape}")
    else:
        print("No tracks found.")
