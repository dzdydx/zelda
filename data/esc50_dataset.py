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
import torch
import torchaudio

def get_text_desc(fg_label, bg_label, snr_db):
    """
    Get text description of the audio mix based on foreground sound event label (fg_label),
    background sound scene label (bg_label), and signal-to-noise ratio (snr_db).
    snr_db: [-15, -10, -5, 0, 5]
    """
    # Replace _ with space
    fg_label = fg_label.replace("_", " ")
    bg_label = bg_label.replace("_", " ")

    # Determine the noise level description based on SNR
    if snr_db == 5:
        noise_level_desc = "quiet"
    elif snr_db == 0:
        noise_level_desc = "moderately noisy"
    elif snr_db == -5:
        noise_level_desc = "noisy"
    elif snr_db == -10:
        noise_level_desc = "very noisy"
    elif snr_db == -15:
        noise_level_desc = "extremely noisy"
    else:
        raise ValueError("Invalid SNR value. Must be one of [-15, -10, -5, 0, 5].")

    # Construct the text prompt
    prompt = f"Sound of {fg_label} in {noise_level_desc} {bg_label}."

    return prompt

def mix_audios(bg_file, fg_file, snr_db):
    """
    Mix background and foreground audio files with given SNR
    Same as mix_audios, but using torchaudio
    """
    bg, bg_sr = torchaudio.load(bg_file)
    fg, fg_sr = torchaudio.load(fg_file)

    # bg from TAU is stereo, convert to mono
    if bg.shape[0] > 1:
        bg = torch.mean(bg, dim=0, keepdim=True)

    # Resample to 32kHz, fit model input
    bg = torchaudio.transforms.Resample(orig_freq=bg_sr, new_freq=32000)(bg)
    fg = torchaudio.transforms.Resample(orig_freq=fg_sr, new_freq=32000)(fg)

    # Trim bg to fg length if bg is longer than fg
    if bg.shape[1] > fg.shape[1]:
        bg = bg[:, :fg.shape[1]]
    else:
        # Otherwise, repeat bg to fg length
        bg = torch.cat([bg] * int(np.ceil(fg.shape[1] / bg.shape[1])), dim=1)[:, :fg.shape[1]]

    # Get the initial energy for reference
    fg_energy = torch.mean(fg ** 2)
    bg_energy = torch.mean(bg ** 2)

    # Calculates the gain to be applied to the noise
    # to achieve the given SNR
    gain = torch.sqrt(10.0 ** (-snr_db/10) * fg_energy / bg_energy)

    # Assumes signal and noise to be decorrelated
    # and calculate (a, b) such that energy of 
    # a*signal + b*noise matches the energy of the input signal
    a = torch.sqrt(1 / (1 + gain**2))
    b = torch.sqrt(gain**2 / (1 + gain**2))

    return a * fg + b * bg

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
    """Pad or truncate audio to specific length."""
    if len(x) <= audio_length:
        return torch.cat(
            (x, torch.zeros(audio_length - len(x), dtype=torch.float32)), dim=0
        )
    else:
        return x[0:audio_length]


class ESC50Dataset(Dataset):
    def __init__(
        self,
        meta_csv,
        root_dir,
        label_vocab,
        mix_noise=False,
        tau_root=None,
        tau_meta=None,
        tau_scene_label=None,
        snr=0,
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

        # add noise from TAU
        self.mix_noise = mix_noise
        self.tau_root = tau_root
        self.snr = snr
        self.scene_label = tau_scene_label
        if mix_noise:
            self.tau_meta = pd.read_csv(tau_meta, sep="\t")


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
        text = ""
        # Get text description
        if self.scene_label:
            text = get_text_desc(label, self.scene_label, self.snr)

        if self.mix_noise:
            # randomly sample 1 file from TAU with given scene label
            tau_sample = self.tau_meta[self.tau_meta['scene_label'] == self.scene_label].sample(1)
            noise_filepath = Path(self.tau_root) / tau_sample['filename'].values[0]
            # mix noise
            waveform = mix_audios(noise_filepath, filepath, self.snr)
            waveform = pad_or_truncate(waveform.squeeze(0), self.clip_length)
            waveform = waveform.reshape(1, -1)
            # Get text description
            text = get_text_desc(label, self.scene_label, self.snr)
        else:
            waveform, _ = torchaudio.load(filepath)
            waveform = pad_or_truncate(waveform.squeeze(0), self.clip_length)
            waveform = waveform.reshape(1, -1)

        target = torch.zeros(self.classes_num)
        target[int(self.index_dict[label])] = 1.0
        logmel = self.get_logmel(waveform)

        if self.augment:
            # Temporarily disable mixup, because mixing TAU bg files makes it
            # difficult to keep amp consistent

            # if random.random() > self.mixup:
            #     mix_sample_idx = random.choices(range(len(self.data)))[0]

            #     label2 = self.data.iloc[mix_sample_idx]['esc_category']
            #     target2 = torch.zeros(self.classes_num)
            #     target2[int(self.index_dict[label2])] = 1.0

            #     filepath2 = Path(self.root_dir) / self.data.iloc[mix_sample_idx]['filename']
            #     waveform2, _ = torchaudio.load(filepath2)
            #     waveform2 = self.resample(waveform2.squeeze(0))
            #     waveform2 = pad_or_truncate(waveform2, self.clip_length)
            #     waveform2 = torch.from_numpy(waveform2.reshape(1, -1))

            #     logmel2 = self.get_logmel(waveform2)

            #     # Do mixup
            #     mix_lambda = np.random.beta(0.2, 0.2)
            #     logmel = mix_lambda * logmel + (1 - mix_lambda) * logmel2
            #     target = mix_lambda * target + (1 - mix_lambda) * target2

            if self.freq_mask > 0:
                logmel = self.freqm(logmel)

            if self.freq_mask > 0:
                logmel = self.timem(logmel)

        return logmel, target, text, audio_name
