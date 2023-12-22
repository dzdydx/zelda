from pathlib import Path
import random
from torch.utils.data import Dataset
import io
import av
import numpy as np
import torch
import torchaudio
import torch.nn as nn
import csv
import pandas as pd

def make_index_dict(label_vocab):
    index_lookup = {}
    with open(label_vocab, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['label']] = row['idx']
            line_count += 1
    return index_lookup

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


class ESC50Dataset(Dataset):
    def __init__(
        self,
        meta_csv,
        root_dir,
        label_vocab,
        sample_rate=32000,
        classes_num=50,
        clip_length=5,
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
        freq_mask=0,
        time_mask=0,
    ):
        self.sample_rate = sample_rate
        self.clip_length = clip_length * sample_rate
        self.classes_num = classes_num
        self.root_dir = root_dir
        self.index_dict = make_index_dict(label_vocab)
        
        # Read meta csv
        self.data = pd.read_csv(meta_csv)

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
            return_complex=True,
        )
        x = torch.view_as_real(x)
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
        # melspec = (melspec + 4.5) / 5.0  # fast normalization

        return melspec

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # Read item meta from self.data dataframe
        data_item = self.data.iloc[index]
        filepath = Path(self.root_dir) / data_item['filename']
        label = data_item['esc_category']
        audio_name = data_item['filename']

        waveform, _ = torchaudio.load(filepath)
        waveform = self.resample(waveform.squeeze(0))
        waveform = pad_or_truncate(waveform, self.clip_length)
        waveform = torch.from_numpy(waveform.reshape(1, -1))

        target = torch.zeros(self.classes_num)
        target[int(self.index_dict[label])] = 1.0

        logmel = self.get_logmel(waveform)

        if self.augment:
            if random.random() > self.mixup:
                mix_sample_idx = random.choices(range(len(self.data)))[0]

                label2 = self.data.iloc[mix_sample_idx]['esc_category']
                target2 = torch.zeros(self.classes_num)
                target2[int(self.index_dict[label2])] = 1.0

                filepath2 = Path(self.root_dir) / self.data.iloc[mix_sample_idx]['filename']
                waveform2, _ = torchaudio.load(filepath2)
                waveform2 = self.resample(waveform2.squeeze(0))
                waveform2 = pad_or_truncate(waveform2, self.clip_length)
                waveform2 = torch.from_numpy(waveform2.reshape(1, -1))

                logmel2 = self.get_logmel(waveform2)

                # Do mixup
                mix_lambda = np.random.beta(0.2, 0.2)
                logmel = mix_lambda * logmel + (1 - mix_lambda) * logmel2
                target = mix_lambda * target + (1 - mix_lambda) * target2

            if self.freq_mask > 0:
                logmel = self.freqm(logmel)

            if self.freq_mask > 0:
                logmel = self.timem(logmel)

        return logmel, target, audio_name
