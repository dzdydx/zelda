import logging
import random
import time
import h5py
from torch.utils.data import Dataset
import io
import av
import numpy as np
import torch
import torchaudio
import torch.nn as nn


def decode_mp3(mp3_arr):
    """
    decodes an array if uint8 representing an mp3 file
    :rtype: np.array
    """
    container = av.open(io.BytesIO(mp3_arr.tobytes()))
    stream = next(s for s in container.streams if s.type == "audio")
    # print(stream)
    a = []
    for i, packet in enumerate(container.demux(stream)):
        for frame in packet.decode():
            a.append(frame.to_ndarray().reshape(-1))
    waveform = np.concatenate(a)
    if waveform.dtype != "float32":
        raise RuntimeError("Unexpected wave type")
    return waveform


def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    if len(x) <= audio_length:
        return np.concatenate(
            (x, np.zeros(audio_length - len(x), dtype=np.float32)), axis=0
        )
    else:
        return x[0:audio_length]


class AudiosetDataset(Dataset):
    def __init__(
        self,
        h5_path,
        in_mem=False,
        sample_rate=32000,
        classes_num=527,
        clip_length=10,
        n_mels=128,
        win_length=800,
        hopsize=320,
        n_fft=1024,
        htk=False,
        fmin=0.0,
        fmax=None,
        norm=1,
        fmin_aug_range=1,
        fmax_aug_range=1000,
        augment=False,
        mixup=0.0,
        cum_weights=None,  # TODO: add this to config
        freq_mask=0,
        time_mask=0,
    ):
        self.sample_rate = sample_rate
        self.clip_length = clip_length * sample_rate
        self.classes_num = classes_num
        self.h5_path = h5_path
        if in_mem:
            print("\nPreloading in memory\n")
            with open(h5_path, "rb") as f:
                self.h5_file = io.BytesIO(f.read())

        with h5py.File(h5_path, "r") as f:
            self.length = len(f["audio_name"])
            print(f"Dataset from {h5_path} with length {self.length}.")

        self.h5_file = None

        # STFT extractor
        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.htk = htk
        self.fmin = fmin
        if fmax is None:
            fmax = sample_rate // 2 - fmax_aug_range // 2
            print(f"Warning: FMAX is None setting to {fmax} ")
        self.fmax = fmax
        self.norm = norm
        self.hopsize = hopsize
        self.window = torch.hann_window(win_length, periodic=False)
        assert (
            fmin_aug_range >= 1
        ), f"fmin_aug_range={fmin_aug_range} should be >=1; 1 means no augmentation"
        assert (
            fmin_aug_range >= 1
        ), f"fmax_aug_range={fmax_aug_range} should be >=1; 1 means no augmentation"
        self.fmin_aug_range = fmin_aug_range
        self.fmax_aug_range = fmax_aug_range
        self.preemphasis_coefficient = torch.as_tensor([[[-0.97, 1]]])

        # augmentation
        self.augment = augment
        if augment:
            self.mixup = mixup
            self.cum_weights = cum_weights
            self.freq_mask = freq_mask
            self.time_mask = time_mask

            if self.freq_mask > 0:
                self.freqm = torchaudio.transforms.FrequencyMasking(
                    freq_mask, iid_masks=True
                )
            if self.time_mask > 0:
                self.timem = torchaudio.transforms.TimeMasking(
                    time_mask, iid_masks=True
                )

    def open_hdf5(self):
        self.h5_file = h5py.File(self.h5_path, "r")

    def resample(self, waveform):
        """Resample.
        Args:
          waveform: (clip_samples,)
        Returns:
          (resampled_clip_samples,)
        """
        if self.sample_rate == 32000:
            return waveform
        elif self.sample_rate == 16000:
            return waveform[0::2]
        elif self.sample_rate == 8000:
            return waveform[0::4]
        else:
            raise Exception("Incorrect sample rate!")

    def get_logmel(self, waveform):
        """Get log mel spectrogram of a waveform.
        Args:
          waveform: (clip_samples,)
        Returns:
          logmel: (1, frames_num, mel_bins)
        """

        x = nn.functional.conv1d(waveform, self.preemphasis_coefficient).squeeze(1)
        x = torch.stft(
            x,
            self.n_fft,
            hop_length=self.hopsize,
            win_length=self.win_length,
            center=True,
            normalized=False,
            window=self.window,
            return_complex=False,
        )
        x = (x**2).sum(dim=-1)  # power mag
        fmin = self.fmin + torch.randint(self.fmin_aug_range, (1,)).detach()
        fmax = (
            self.fmax
            + self.fmax_aug_range // 2
            - torch.randint(self.fmax_aug_range, (1,)).detach()
        )
        # don't augment eval data
        if not self.augment:
            fmin = self.fmin
            fmax = self.fmax

        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
            self.n_mels,
            self.n_fft,
            self.sample_rate,
            fmin,
            fmax,
            vtln_low=100.0,
            vtln_high=-500.0,
            vtln_warp_factor=1.0,
        )
        mel_basis = torch.as_tensor(
            torch.nn.functional.pad(mel_basis, (0, 1), mode="constant", value=0),
            device=x.device,
        )
        with torch.cuda.amp.autocast(enabled=False):
            melspec = torch.matmul(mel_basis, x)

        melspec = (melspec + 0.00001).log()
        melspec = (melspec + 4.5) / 5.0  # fast normalization

        return melspec

    def __len__(self):
        return self.length

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    def __getitem__(self, index):
        if self.h5_file is None:
            self.open_hdf5()

        audio_name = self.h5_file["audio_name"][index].decode()
        waveform = decode_mp3(self.h5_file["mp3"][index])
        waveform = pad_or_truncate(waveform, self.clip_length)
        waveform = self.resample(waveform)

        target = self.h5_file["target"][index]
        target = np.unpackbits(target, axis=-1, count=self.classes_num).astype(
            np.float32
        )

        waveform = torch.from_numpy(waveform.reshape(1, -1))
        target = torch.from_numpy(target)

        if self.augment:
            if random.random() < self.mixup:
                logmel = self.get_logmel(waveform)
            else:
                # Use weighted sampler to choose mixup sample
                mix_sample_idx = random.choices(
                    range(len(self.data)), cum_weights=self.cum_weights
                )[0]

                label2 = self.h5_file["target"][mix_sample_idx]
                label2 = np.unpackbits(label2, axis=-1, count=self.classes_num).astype(
                    np.float32
                )
                label2 = torch.from_numpy(label2)

                waveform2 = decode_mp3(self.h5_file["mp3"][mix_sample_idx])
                waveform2 = pad_or_truncate(waveform2, self.clip_length)
                waveform2 = self.resample(waveform2)
                waveform2 = torch.from_numpy(waveform2.reshape(1, -1))

                logmel2 = self.get_logmel(waveform2)

                # Do mixup
                mix_lambda = np.random.beta(0.2, 0.2)
                logmel = mix_lambda * logmel + (1 - mix_lambda) * logmel2
                target = mix_lambda * target + (1 - mix_lambda) * label2

            if self.freq_mask > 0:
                logmel = self.freqm(logmel)

            if self.freq_mask > 0:
                logmel = self.timem(logmel)
        else:
            logmel = self.get_logmel(waveform)

        return logmel, target, audio_name
