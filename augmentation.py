import os
import torch
import numpy as np
import soundfile as sf
import torchaudio
import librosa
import librosa.display
import matplotlib.pyplot as plt

#from pydub import AudioSegment
#from pydub.utils import make_chunks
from audiomentations import Compose, AddGaussianSNR, PitchShift, HighPassFilter, TimeMask, TimeStretch, Reverse
#from plot_helper import _plot_signal_and_augmented_signal


def create_chunks(audio_file, chunk_length_ms=3000):
    my_audio = AudioSegment.from_file(audio_file, format="wav")
    chunks = make_chunks(my_audio, chunk_length_ms)

    return chunks

#-----------------raw waveform augmentation-------------------------
def add_random_noise(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    #casting augmented data back to its formal format
    augmented_data = augmented_data.astype(type(data[0]))

    return augmented_data

def shift_time(data, sampling_rate, shift_max, shift_direction):
    #shift audio left/right for a random second
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift

    augmented_data = np.roll(data, shift)
    #set silence for tail/head
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0

    return augmented_data

def time_stretch(data, stretch_factor):
    return librosa.effects.time_stretch(data, rate= stretch_factor) 

def tune_pitch(data, sampling_rate, pitch_factor):
    return librosa.effects.pitch_shift(data, sr= sampling_rate, n_steps= pitch_factor)


#-------------------------spectogram augmentation---------------------
"""
def mixspeech(original_melspecs, original_labels, alpha=1.0):
    indices = torch.randperm(original_melspecs.size(0))

    lam = np.random.beta(alpha, alpha) 

    augmented_melspecs = original_melspecs * lam + original_melspecs[indices] * (1 - lam)
    augmented_labels = [(original_labels * lam), (original_labels[indices] * (1 - lam))]

    return augmented_melspecs, augmented_labels
"""
def mixspeech(original_melspecs, original_labels, alpha=1.0):
    
    # Randomly select two indices
    idx1, idx2 = np.random.choice(len(original_melspecs), size=2, replace=False)

    # Get the two spectrograms
    melspec1 = original_melspecs[idx1]
    melspec2 = original_melspecs[idx2]

    # Resizing melspec2 to match the shape of melspec1
    melspec2 = melspec2[:222, :]

    # Generate mixing coefficient
    lam = np.random.beta(alpha, alpha)

    # Mix up the spectrograms
    augmented_melspec = melspec1 * lam + melspec2 * (1 - lam)

    # Mix up the labels
    augmented_labels = original_labels[idx1] * lam + original_labels[idx2] * (1 - lam)

    return augmented_melspec, augmented_labels

#------------------------audiomentations augmentation------------------
"""
augment = Compose([
    #AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
    #PitchShift(min_semitones= -1, max_semitones= 1, p= 0.3),
    #HighPassFilter(min_cutoff_freq= 2000, max_cutoff_freq= 4000, p= 0.2),
    Reverse(p= 0.9),
    TimeMask(min_band_part=0.1, max_band_part=0.15, fade=True, p= 1)
    #TimeStretch(min_rate=0.8, max_rate=1.25, leave_length_unchanged=True, p= 0.9)
])
"""
augment_0 = Reverse(p=1)

augment_1 = TimeMask(min_band_part=0.1, max_band_part=0.15, p=1)

augment_2 = AddGaussianSNR(
            min_snr_db = 20.0, 
            max_snr_db = 40.0, 
            p=1.0
)
 

def plot_waveform(waveform, sample_rate, title = "Waveform"):
  waveform = waveform.numpy()
  num_channels, num_frames = waveform.shape
  time = np.arange(0, num_frames) / sample_rate

  fig, axes = plt.subplots(num_channels, 1)
  
  if num_channels == 1:
    axes = [axes]
  for ch in range(num_channels):
    axes[ch].plot(time, waveform[ch])
    axes[ch].grid(True)

    if num_channels > 1:
      axes[ch].set_ylabel(f"Channel: {ch+1}")
  plt.suptitle(title)
  plt.show(block = False)


if __name__ == "__main__":

    base_path = os.getcwd()
    dataset_path = os.path.join(base_path, "dataset/Speech_data_forSSC_copy/Speech_data_forSSC_2.0/Speech_data/Neutral_speech/Amazon+Google_voices/dataset_1/wav/mixed_chunk_sentence")

    dataset_list = os.path.join(base_path, "dataset/Speech_data_forSSC_copy/Speech_data_forSSC_2.0/Speech_data/Neutral_speech/Amazon+Google_voices", "train_list_one_hot_chunk_sentence_edited.txt")

    data = open(dataset_list).read().splitlines()
    #print(data)

    audio = []
    for idx, d in enumerate(data):
        file_name = os.path.join(dataset_path, d.split()[1])
        audio.append(file_name)

    #print(audio)

    """
    chunks = []
    for au in audio:
        speaker = au.split("/")[-2]
        file = (au.split("/")[-1]).split(".")[0]
        print("chunking...")
        print(speaker, au)
        chunks = create_chunks(au)

        for j, chunk in enumerate(chunks):
            export_path = os.path.join(dataset_path, speaker)
            chunk.export(f"{export_path}/chunk_{file}_{j}.wav", format= "wav")
    """
    #pass
    
    ##Adding white noise, time stretch, pitch shift, time_shift
    wave, sr = sf.read(audio[5])
    wave_1, sr = sf.read(audio[4])
    augmented_wave = add_random_noise(wave, 0.1)
    #augmented_wave = time_stretch(wave, 0.8)
    #augmented_wave = tune_pitch(wave, sr, 1.5)
    #augmented_wave = shift_time(wave, sr, 0.1, 'right')
    #sf.write("augmented.wav", augmented_wave, sr)
    ##plot to compare original to augmented waveform
    _plot_signal_and_augmented_signal(wave, augmented_wave, sr)

    wave_torch, sr_torch = torchaudio.load(audio[5])

    ##torchaudio augmentation
    effects = [["lowpass", "-1", "300"],
           ["speed", "0.8"],
           ["rate", f"{sr}"],
           ["reverb", "-w"]]
    
    waveform2, sample_rate2 = torchaudio.sox_effects.apply_effects_tensor(wave_torch, sr, effects)

    plot_waveform(wave_torch, sr_torch, title="Original")
    plot_waveform(waveform2, sample_rate2, title="Effect Applied")

    #_plot_signal_and_augmented_signal(wave, waveform2, sr)

    ##audiomentations
    augmented_wave_audiomentations = augment(wave, sr)
    #sf.write("augmented_audiomentations.wav", augmented_wave_audiomentations, sr)
    _plot_signal_and_augmented_signal(wave, augmented_wave_audiomentations, sr)

    ##mixspeech, specswap
    original_melspec_1 = librosa.feature.melspectrogram(y = wave,
                                                  sr = sr, 
                                                  n_fft = 512, 
                                                  hop_length = 256, 
                                                  n_mels = 40).T

    original_melspec_2 = librosa.feature.melspectrogram(y = wave_1,
                                                  sr = sr, 
                                                  n_fft = 512, 
                                                  hop_length = 256, 
                                                  n_mels = 40).T

    melspecs = [original_melspec_1, original_melspec_2]
    labels = [0, 1]

    augment_spec, mixed_labels = mixspeech(melspecs, labels) 

    #print("mixed_labels: ", mixed_labels)

    # Plotting the first spectrogram
    plt.figure(figsize=(10, 4))
    plt.subplot(3, 1, 1)
    librosa.display.specshow(librosa.power_to_db(original_melspec_1, ref=np.max), sr=sr, hop_length=256, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram 1')
    plt.tight_layout()

    # Plotting the second spectrogram
    plt.subplot(3, 1, 2)
    librosa.display.specshow(librosa.power_to_db(original_melspec_2, ref=np.max), sr=sr, hop_length=256, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram 2')
    plt.tight_layout()

    plt.subplot(3, 1, 3)
    librosa.display.specshow(librosa.power_to_db(augment_spec, ref=np.max), sr=sr, hop_length=256, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram 2')
    plt.tight_layout()

    plt.show()
