import math
import os
import random

import torch

from argparse import Namespace

from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize
from scipy.io.wavfile import read
from torch.utils.data import Dataset

MAX_WAV_VALUE = 32768.0


def spectral_normalize(magnitudes: torch.Tensor, clip_val: float = 1e-5):
    output = torch.log(torch.clamp(magnitudes, min=clip_val))
    return output


def get_mel_spectrogram(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False, **kwargs
) -> torch.Tensor:
    mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel_basis = torch.from_numpy(mel).float().to(y.device)
    hann_window = torch.hann_window(win_size).to(y.device)
    pad_value = int((n_fft - hop_size) / 2)
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad_value, pad_value), mode="reflect").squeeze(1)
    spectrogram = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )
    spectrogram = torch.sqrt(spectrogram.pow(2).sum(-1) + 1e-9)
    mel_spectrogram = torch.matmul(mel_basis, spectrogram)
    normalized_mel_spectrogram = spectral_normalize(mel_spectrogram)
    return normalized_mel_spectrogram


def get_dataset_filelist(args: Namespace) -> tuple[list, list]:
    with open(args.input_training_file, "r", encoding="utf-8") as f:
        training_files = [i[:-1] for i in f.readlines()]
    with open(args.input_validation_file, "r", encoding="utf-8") as f:
        validation_files = [i[:-1] for i in f.readlines()]
    return training_files, validation_files


class MelDataset(Dataset):
    def __init__(
        self,
        training_files,
        segment_size,
        n_fft,
        num_mels,
        hop_size,
        win_size,
        sampling_rate,
        fmin,
        fmax,
        seed,
        split=True,
        device=None,
        fmax_loss=None,
        fine_tuning=False,
        input_mels_dir=None,
        **kwargs,
    ):
        random.seed(seed)
        self.fmin = fmin
        self.fmax = fmax
        self.split = split
        self.n_fft = n_fft
        self.device = device
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmax_loss = fmax_loss
        self.fine_tuning = fine_tuning
        self.audio_files = training_files
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.base_mels_path = input_mels_dir
        self.frames_per_sec = math.ceil(segment_size / hop_size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]:
        filename = self.audio_files[index]
        sampling_rate, audio = read(filename)
        audio = audio / MAX_WAV_VALUE
        if not self.fine_tuning:
            audio = normalize(audio) * 0.95
        if sampling_rate != self.sampling_rate:
            raise ValueError(f"{sampling_rate} SR doesn't match target {self.sampling_rate} SR")
        audio = torch.FloatTensor(audio).unsqueeze(0)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start : audio_start + self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), "constant")

            mel = get_mel_spectrogram(
                audio,
                self.n_fft,
                self.num_mels,
                self.sampling_rate,
                self.hop_size,
                self.win_size,
                self.fmin,
                self.fmax,
                center=False,
            )
        else:
            mel = torch.load(
                os.path.join(self.base_mels_path, os.path.splitext(os.path.split(filename)[-1])[0] + ".pt")
            )
            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                if audio.size(1) >= self.segment_size:
                    mel, audio = self.cut_mel_audio(mel, audio)
                else:
                    mel, audio = self.pad_mel_audio(mel, audio)

        mel_loss = get_mel_spectrogram(
            audio,
            self.n_fft,
            self.num_mels,
            self.sampling_rate,
            self.hop_size,
            self.win_size,
            self.fmin,
            self.fmax_loss,
            center=False,
        )

        return mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze()

    def cut_mel_audio(self, mel: torch.Tensor, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mel_start = random.randint(0, mel.size(2) - self.frames_per_sec - 1)
        mel = mel[:, :, mel_start : mel_start + self.frames_per_sec]
        audio = audio[:, mel_start * self.hop_size : (mel_start + self.frames_per_sec) * self.hop_size]
        return mel, audio

    def pad_mel_audio(self, mel: torch.Tensor, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mel = torch.nn.functional.pad(mel, (0, self.frames_per_sec - mel.size(2)), "constant")
        audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), "constant")
        return mel, audio

    def __len__(self) -> int:
        return len(self.audio_files)
