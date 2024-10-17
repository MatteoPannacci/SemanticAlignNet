### Imports
import pytorch_lightning as pl
import abc
from .utils import top_k_rank_accuracy
import torch


###



# @title Multi-Branch Model

class MultiBranchModel(pl.LightningModule):

    def __init__(self, loss, learning_rate):
        super(MultiBranchModel, self).__init__()

        self.learning_rate = learning_rate

        self.loss = loss

        self.branch_grd = None
        self.branch_grd_seg = None
        self.branch_grd_dep = None
        self.branch_sat = None
        self.branch_sat_seg = None

        self.grd_features_train = []
        self.sat_features_train = []

        self.grd_features_val =  []
        self.sat_features_val =  []

        self.grd_features_test = []
        self.sat_features_test = []


    @abc.abstractmethod
    def forward(self, batch):
        return


    @abc.abstractmethod
    def compute_loss(self, embeddings, grd_emb, sat_emb):
        return


    def training_step(self, batch, batch_idx):
        embeddings, grd_emb, sat_emb = self(batch)

        loss = self.compute_loss(embeddings, grd_emb, sat_emb)
        top_1 = top_k_rank_accuracy(grd_emb, sat_emb, k=1)

        self.log('train_top1', top_1, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):
        embeddings, grd_emb, sat_emb = self(batch)

        self.grd_features_val.append(grd_emb)
        self.sat_features_val.append(sat_emb)
        loss = self.compute_loss(embeddings, grd_emb, sat_emb)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)


    def on_validation_epoch_end(self):

      grd_features_val = torch.cat(self.grd_features_val, dim=0)
      sat_features_val = torch.cat(self.sat_features_val, dim=0)

      num_samples = grd_features_val.shape[0]
      percent1 = int(0.01*num_samples)

      top_1 = top_k_rank_accuracy(grd_features_val, sat_features_val, k=1)
      top_5 = top_k_rank_accuracy(grd_features_val, sat_features_val, k=5)
      top_10 = top_k_rank_accuracy(grd_features_val, sat_features_val, k=10)
      top_percent1 = top_k_rank_accuracy(grd_features_val, sat_features_val, k=percent1)

      top_1_inverse = top_k_rank_accuracy(grd_features_val, sat_features_val, k=1, inverse=True)
      top_5_inverse = top_k_rank_accuracy(grd_features_val, sat_features_val, k=5, inverse=True)
      top_10_inverse = top_k_rank_accuracy(grd_features_val, sat_features_val, k=10, inverse=True)
      top_percent1_inverse = top_k_rank_accuracy(grd_features_val, sat_features_val, k=percent1, inverse=True)

      self.log('val_top1', top_1, on_step=False, on_epoch=True, prog_bar=True)
      self.log('val_top5', top_5, on_step=False, on_epoch=True, prog_bar=True)
      self.log('val_top10', top_10, on_step=False, on_epoch=True, prog_bar=True)
      self.log('val_top1%', top_percent1, on_step=False, on_epoch=True, prog_bar=True)

      self.log('val_top1_inv', top_1_inverse, on_step=False, on_epoch=True, prog_bar=True)
      self.log('val_top5_inv', top_5_inverse, on_step=False, on_epoch=True, prog_bar=True)
      self.log('val_top10_inv', top_10_inverse, on_step=False, on_epoch=True, prog_bar=True)
      self.log('val_top1%_inv', top_percent1_inverse, on_step=False, on_epoch=True, prog_bar=True)

      self.grd_features_val.clear()
      self.sat_features_val.clear()
      del grd_features_val, sat_features_val

      return top_1, top_5, top_10, top_percent1, top_1_inverse, top_5_inverse, top_10_inverse, top_percent1_inverse


    def test_step(self, batch, batch_idx):
        embeddings, grd_emb, sat_emb = self(batch)

        self.grd_features_test.append(grd_emb)
        self.sat_features_test.append(sat_emb)
        loss = self.compute_loss(embeddings, grd_emb, sat_emb)
        #self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):

      grd_features_test = torch.cat(self.grd_features_test, dim = 0)
      sat_features_test = torch.cat(self.sat_features_test, dim = 0)

      num_samples = grd_features_test.shape[0]
      percent1 = int(0.01*num_samples)

      top_1 = top_k_rank_accuracy(grd_features_test, sat_features_test, k=1)
      top_5 = top_k_rank_accuracy(grd_features_test, sat_features_test, k=5)
      top_10 = top_k_rank_accuracy(grd_features_test, sat_features_test, k=10)
      top_percent1 = top_k_rank_accuracy(grd_features_test, sat_features_test, k=percent1)

      top_1_inverse = top_k_rank_accuracy(grd_features_test, sat_features_test, k=1, inverse=True)
      top_5_inverse = top_k_rank_accuracy(grd_features_test, sat_features_test, k=5, inverse=True)
      top_10_inverse = top_k_rank_accuracy(grd_features_test, sat_features_test, k=10, inverse=True)
      top_percent1_inverse = top_k_rank_accuracy(grd_features_test, sat_features_test, k=percent1, inverse=True)

      self.log('test_top1', top_1, on_step=False, on_epoch=True, prog_bar=True)
      self.log('test_top5', top_5, on_step=False, on_epoch=True, prog_bar=True)
      self.log('test_top10', top_10, on_step=False, on_epoch=True, prog_bar=True)
      self.log('test_top1%', top_percent1, on_step=False, on_epoch=True, prog_bar=True)

      self.log('test_top1_inv', top_1_inverse, on_step=False, on_epoch=True, prog_bar=True)
      self.log('test_top5_inv', top_5_inverse, on_step=False, on_epoch=True, prog_bar=True)
      self.log('test_top10_inv', top_10_inverse, on_step=False, on_epoch=True, prog_bar=True)
      self.log('test_top1%_inv', top_percent1_inverse, on_step=False, on_epoch=True, prog_bar=True)

      self.grd_features_test.clear()
      self.sat_features_test.clear()
      del grd_features_test, sat_features_test

      return top_1, top_5, top_10, top_percent1, top_1_inverse, top_5_inverse, top_10_inverse, top_percent1_inverse


    def configure_optimizers(self):
      optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
      return optimizer

    @abc.abstractmethod
    def __repr__(self):
        return
