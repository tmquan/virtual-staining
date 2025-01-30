import os
import sys
import contextlib
import io
import warnings
warnings.simplefilter("ignore", category=Warning)

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
print(rlimit)
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, rlimit[1]))

import hydra
from omegaconf import DictConfig, OmegaConf

from typing import Any, Callable, Dict, Optional, OrderedDict, Tuple, List
from lightning.pytorch import seed_everything, Trainer, LightningModule

from datamodule import PairedDataModule
from litmodelgan import LightningModule


class SuppressOutput:
    def __init__(self):
        self._original_stdout = sys.stdout  # Save the original stdout

    def __enter__(self):
        # Redirect stdout to a string stream
        sys.stdout = io.StringIO()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Get the output and restore original stdout
        output = sys.stdout.getvalue()
        sys.stdout = self._original_stdout
        
        # Filter out lines containing '[Warning]'
        for line in output.splitlines():
            if '[Warning]' not in line:
                print(line)  # Print only lines that do not contain the warning
                
@hydra.main(version_base=None, config_path="./conf")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)  # resolve all str interpolation
    seed_everything(42)
    datamodule = PairedDataModule(
        root_folder=cfg.data.root_folder,
        batch_size=cfg.data.batch_size,
        img_shape=cfg.data.img_shape,
        train_folders=cfg.data.train_folders,
        val_folders=cfg.data.val_folders,
        test_folders=cfg.data.test_folders,
        train_samples=cfg.data.train_samples,
        val_samples=cfg.data.val_samples,
        test_samples=cfg.data.test_samples
    )

    model = LightningModule(model_cfg=cfg.model, train_cfg=cfg.train,)
    callbacks = [hydra.utils.instantiate(c) for c in cfg.callbacks]
    logger = [hydra.utils.instantiate(c) for c in cfg.logger]

    trainer = Trainer(callbacks=callbacks, logger=logger, **cfg.trainer)

    print(f"Is testing: {cfg.test}")
    if cfg.test is None:
        trainer.fit(
            model,
            # datamodule=datamodule,
            train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader(),
            # ckpt_path=cfg.resume_from_checkpoint
        )
    else:
        trainer.test(
            model,
            dataloaders=datamodule.test_dataloader(),
            # ckpt_path=cfg.resume_from_checkpoint
        )


if __name__ == "__main__":
    main()

