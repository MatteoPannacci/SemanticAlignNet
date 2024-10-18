from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import os

# # @title Implementation ModelCheckpoint
# class MyModelCheckpoint(ModelCheckpoint):
#     def __init__(self, config, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.config = config

#     def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
#         if not self.dirpath:
#             raise ValueError("dirpath must be specified in ModelCheckpoint")
#         file_path = os.path.join(os.path.dirname(self.dirpath), "config.txt")
#         with open(file_path, 'w') as f:
#             f.write(self.config)
#             f.write('\n')

#         return super().on_train_start(trainer, pl_module)

class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.dirpath:
            raise ValueError("dirpath must be specified in ModelCheckpoint")
        
        # Save the config file once at the beginning of training
        config_file_path = os.path.join(os.path.dirname(self.dirpath), "config.txt")
        with open(config_file_path, 'w') as f:
            f.write(self.config)
            f.write('\n')

        super().on_train_start(trainer, pl_module)

    def on_save_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: dict) -> dict:
        # Set a new directory for each epoch
        epoch_dir = os.path.join(self.dirpath, f"epoch_{trainer.current_epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        # Update the dirpath to the new epoch-specific directory
        self.dirpath = epoch_dir
        
        return super().on_save_checkpoint(trainer, pl_module, checkpoint)
