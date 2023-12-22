import lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from .audioset_dataset import AudiosetDataset
from .esc50_dataset import ESC50Dataset
from omegaconf import OmegaConf


def process_config(cfg, dataset="audioset", split="train"):
    base_cfg_dict = OmegaConf.to_container(cfg.preprocess_args, resolve=True)
    new_cfg = base_cfg_dict.copy()
    if dataset == "audioset":
        if split == "train":
            new_cfg["h5_path"] = cfg.train_h5
        elif split == "val":
            new_cfg["h5_path"] = cfg.val_h5
            new_cfg["augment"] = False
        elif split == "test":
            new_cfg["h5_path"] = cfg.test_h5
            new_cfg["augment"] = False
        else:
            raise ValueError(
                f"Split must be one of 'train', 'val', 'test'. Got {split} instead."
            )
    elif dataset == "esc50":
        new_cfg["root_dir"] = cfg.root_dir
        new_cfg["label_vocab"] = cfg.label_vocab
        if split == "train":
            new_cfg["meta_csv"] = cfg.train_csv
        elif split == "val":
            new_cfg["meta_csv"] = cfg.val_csv
            new_cfg["augment"] = False
        elif split == "test":
            new_cfg["meta_csv"] = cfg.test_csv
            new_cfg["augment"] = False
        else:
            raise ValueError(
                f"Split must be one of 'train', 'val', 'test'. Got {split} instead."
            )

    return new_cfg


class DInterface(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()

        self.args = args

        if args.dataset == "audioset":
            self.data_module = AudiosetDataset
        elif args.dataset == "esc50":
            self.data_module = ESC50Dataset

        self.train_cfg = process_config(args, dataset=args.dataset, split="train")
        self.val_cfg = process_config(args, dataset=args.dataset, split="val")
        self.test_cfg = process_config(args, dataset=args.dataset, split="test")

    def setup(self, stage):
        if self.args.dataset == "audioset":
            if stage == "fit":
                self.trainset = self.data_module(**self.train_cfg)
                if self.val_h5:
                    self.valset = self.data_module(**self.val_cfg)

            if stage == "test":
                self.testset = self.data_module(**self.test_cfg)

            if stage == "predict":
                self.predict_set = self.data_module(**self.test_cfg)
                
        elif self.args.dataset == "esc50":
            if stage == "fit":
                self.trainset = self.data_module(**self.train_cfg)
                if self.args.val_csv:
                    self.valset = self.data_module(**self.val_cfg)

            if stage == "test":
                self.testset = self.data_module(**self.test_cfg)

            if stage == "predict":
                self.predict_set = self.data_module(**self.test_cfg)

    def train_dataloader(self):
        if len(self.trainset) > 50000 and self.train_cfg.get("cum_weights"):
            samples_weight = np.loadtxt(
                self.train_cfg.get("cum_weights"),
                delimiter=",",
            )
            print(
                f"Using weighted sampler. Weight file: {self.train_cfg.get('cum_weights')}"
            )
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            return DataLoader(
                self.trainset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                sampler=sampler,
            )
        else:
            return DataLoader(
                self.trainset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=True,
            )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_set,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
        )
