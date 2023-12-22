from argparse import ArgumentParser
import lightning as pl
from omegaconf import OmegaConf
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch.callbacks as plc
from models import MInterface
from data import DInterface
from utils.helpers import save_prediction

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--config", required=True, help="path to configuration file.")

    args = parser.parse_args()

    return args

def load_callbacks():
    callbacks = []

    # callbacks.append(plc.EarlyStopping(
    #     monitor="top1_acc",
    #     mode="max",
    #     patience=20,
    #     min_delta=0.001,
    #     check_on_train_epoch_end=False,
    #     verbose=False,
    # ))

    callbacks.append(plc.ModelCheckpoint(
        monitor="top1_acc",
        filename='best-{epoch:02d}-{top1_acc:.3f}',
        save_top_k=1,
        mode="max",
        save_last=True
    ))

    callbacks.append(plc.LearningRateMonitor(
            logging_interval='step'))
    
    return callbacks

def main(args):
    cfg = OmegaConf.load(args.config)

    data_module = DInterface(cfg.data)
    model = MInterface(cfg.model)
    logger = TensorBoardLogger(save_dir=cfg.log_dir, name=cfg.model.name)
    callbacks = load_callbacks()

    trainer = pl.Trainer(**cfg.trainer, logger=logger, accelerator='gpu', callbacks=callbacks)

    if cfg.mode == "train":
        trainer.fit(model, data_module, ckpt_path=cfg.get("ckpt_path", None))
        result = trainer.test(model, data_module)
    elif cfg.mode == "eval":
        result = trainer.test(model, data_module, ckpt_path=cfg.get("ckpt_path", None))
    elif cfg.mode == "predict":
        result = trainer.predict(model, data_module, ckpt_path=cfg.get("ckpt_path", None))
    elif cfg.mode == "tune":
        trainer.tune(model, data_module)

    save_prediction(result, save_dir=logger.log_dir, mode=cfg.mode)
if __name__ == "__main__":
    args = parse_args()
    main(args)