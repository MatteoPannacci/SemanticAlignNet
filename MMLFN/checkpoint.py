from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import os

# @title Implementation ModelCheckpoint
class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.dirpath:
            raise ValueError("dirpath must be specified in ModelCheckpoint")
        file_path = os.path.join(os.path.dirname(self.dirpath), "config.txt")
        with open(file_path, 'w') as f:
            f.write(self.config)
            f.write('\n')

        return super().on_train_start(trainer, pl_module)