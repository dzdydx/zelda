import torch
import torchmetrics
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MultilabelAveragePrecision,
    MultilabelAUROC,
)
import lightning as pl
from torch.nn import functional as F
import torch.optim.lr_scheduler as lr_schedulers
from .helpers.ramp import exp_warmup_linear_down, cosine_cycle

# from ontology_audio_tagging import (
#     ontology_binary_cross_entropy,
#     ontology_mean_average_precision,
# )


class MInterface(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.num_classes = args.num_classes
        self.args = args
        self.save_hyperparameters()

        if args.classification_type == "multilabel":
            self.metric = torchmetrics.MetricCollection(
                {
                    "mAP": MultilabelAveragePrecision(num_classes=args.num_classes),
                    "aucroc": MultilabelAUROC(num_classes=args.num_classes),
                }
            )
        elif args.classification_type == "multiclass":
            self.metric = torchmetrics.MetricCollection(
                {
                    "top1_acc": MulticlassAccuracy(num_classes=args.num_classes),
                    "aucroc": MulticlassAUROC(num_classes=args.num_classes),
                }
            )

        self.load_model(args)

    def load_model(self, args):
        if args.name == "panns":
            from .panns import Cnn14

            model = Cnn14(
                classes_num=args.num_classes,
            )
        elif args.name == "passt":
            from .passt import PaSST

            model = PaSST(
                stride=args.stride,
                num_classes=args.num_classes,
                distilled=args.distilled,
                s_patchout_t=20,
                s_patchout_f=4,
            )
        else:
            raise NotImplementedError(f"Unsupported model name {args.name}!")

        if args.get("pretrain_model", None):
            print("Loading pretrained model from", args.pretrain_model)
            checkpoint = torch.load(args.pretrain_model)
            if args.name == "panns":
                model.load_state_dict(checkpoint["model"])
            elif args.name == "passt":
                model.load_state_dict(checkpoint)

        self.model = model

    def forward(self, x):
        return self.model(x)

    def on_train_start(self) -> None:
        # log hyperparams
        self.logger.log_hyperparams(
            self.hparams,
            {
                "top1_acc": 0,
                "aucroc": 0,
            },
        )
        return super().on_train_start()

    def training_step(self, batch):
        x, target, text, _ = batch
        y_hat = self.model(x)

        if self.args.classification_type == "multilabel":
            y_hat = torch.sigmoid(y_hat)
            loss = F.binary_cross_entropy(y_hat, target)
            # obce_loss = ontology_binary_cross_entropy(y_hat, target)
            # total_loss = (bce_loss + obce_loss) / 2
            # self.log("obce_loss", obce_loss, on_step=True, on_epoch=True, logger=True)
            self.log(
                "train_loss",
                loss,
                on_step=True,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )
        elif self.args.classification_type == "multiclass":
            loss = F.cross_entropy(y_hat, torch.argmax(target, axis=1))
            self.log(
                "train_loss",
                loss,
                on_step=True,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )
        else:
            raise NotImplementedError(
                "Only multilabel and multiclass classification are supported"
            )

        return loss

    def validation_step(self, batch, batch_idx):
        x, target, text, audio_name = batch
        y_hat = self.model(x)

        if self.args.classification_type == "multilabel":
            y_hat = torch.sigmoid(y_hat)
            loss = F.binary_cross_entropy(y_hat, target)
            self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True)

            self.log_dict(self.metric(y_hat, target), prog_bar=True, logger=True)
        elif self.args.classification_type == "multiclass":
            loss = F.cross_entropy(y_hat, torch.argmax(target, axis=1))
            self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True)

            # y_hat = F.softmax(y_hat, dim=1)
            self.log_dict(
                self.metric(y_hat, torch.argmax(target, axis=1)),
                prog_bar=True,
                logger=True,
            )
        else:
            raise NotImplementedError(
                "Only multilabel and multiclass classification are supported"
            )

        return {"target": target, "prediction": y_hat, "audio_name": audio_name}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        x, target, _, audio_name = batch
        y_hat = self.model(x)

        if self.args.classification_type == "multilabel":
            y_hat = torch.sigmoid(y_hat)
        elif self.args.classification_type == "multiclass":
            y_hat = F.softmax(y_hat, dim=1)
        else:
            raise NotImplementedError(
                "Only multilabel and multiclass classification are supported"
            )

        return {"target": target, "prediction": y_hat, "audio_name": audio_name}

    def configure_optimizers(self):
        if self.args.weight_decay > 0:
            weight_decay = self.args.weight_decay
        else:
            weight_decay = 0.0

        if self.args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.args.lr, weight_decay=weight_decay
            )
        else:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.args.lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )

        if self.args.lr_scheduler is None:
            return optimizer
        else:
            if self.args.lr_scheduler == "step":
                scheduler = lr_schedulers.StepLR(
                    optimizer,
                    step_size=self.args.lr_decay_steps,
                    gamma=self.args.lr_decay_rate,
                )
            elif self.args.lr_scheduler == "cosine":
                scheduler = lr_schedulers.CosineAnnealingLR(
                    optimizer,
                    T_max=self.args.lr_decay_steps,
                    eta_min=self.args.lr_decay_min_lr,
                )
            elif self.args.lr_scheduler == "multistep":
                scheduler = lr_schedulers.MultiStepLR(
                    optimizer, [1, 2, 4, 7], gamma=0.5, last_epoch=-1
                )
            elif self.args.lr_scheduler == "cos_ann_warm":
                scheduler = lr_schedulers.CosineAnnealingWarmRestarts(
                    optimizer, T_0=4, T_mult=3, eta_min=1e-6, last_epoch=-1
                )
            elif self.args.lr_scheduler == "exp_warm":
                scheduler = lr_schedulers.LambdaLR(
                    optimizer,
                    exp_warmup_linear_down(
                        warmup=self.args.warmup_epochs,
                        rampdown_length=self.args.rampdown_length,
                        start_rampdown=self.args.start_rampdown,
                        last_value=self.args.last_lr_value,
                    ),
                )
            elif self.args.lr_scheduler == "cos_cycle":
                scheduler = lr_schedulers.LambdaLR(optimizer, cosine_cycle(5, 50, 0.01))
            else:
                raise ValueError("Invalid lr_scheduler type!")
            return [optimizer], [scheduler]

    # def log_scores(self, name, outputs):
    #     target = torch.stack([output["target"] for output in outputs]).detach().cpu().numpy()
    #     prediction = torch.stack([output["prediction"] for output in outputs]).detach().cpu().numpy()

    #     if self.args.classification_type == "multilabel":
    #         (
    #             omap_average,
    #             omap_on_different_coarse_level,
    #             omap_on_different_coarse_level_details,
    #         ) = ontology_mean_average_precision(prediction, target)
    #         mAP = MeanAveragePrecision()(prediction, target)
    #         aucroc = AUCROC()(prediction, target)

    #         self.log(f"{name}_OmAP", omap_average, on_step=False, on_epoch=True, logger=True)
    #         self.log(f"{name}_mAP", mAP, on_step=False, on_epoch=True, logger=True)
    #         self.log(f"{name}_aucroc", aucroc, on_step=False, on_epoch=True, logger=True)
    #     elif self.args.classification_type == "multiclass":
    #         top1_acc = Top1Accuracy()(prediction, target)
    #         aucroc = AUCROC()(prediction, target)
    #         self.log(f"{name}_top1_acc", top1_acc, on_step=False, on_epoch=True, logger=True)
    #         self.log(f"{name}_aucroc", aucroc, on_step=False, on_epoch=True, logger=True)
    #     else:
    #         raise NotImplementedError(
    #             "Only multilabel and multiclass classification are supported"
    #         )

    # def on_validation_epoch_end(self, outputs):
    #     self.log_scores("val", outputs)

    # def on_test_epoch_end(self, outputs):
    #     self.log_scores("test", outputs)

    # def on_validation_epoch_end(self):
    #     # Make the Progress Bar leave there
    #     self.print("")
