### Import
from generic_model import MultiBranchModel
from .losses import InfoNCE
from torch import nn


###



# @title Dual Model

class DualModel(MultiBranchModel):

    def __init__(
        self,
        model_grd,
        model_sat,
        loss = InfoNCE(loss_function=nn.CrossEntropyLoss()),
        learning_rate = 0.0001,
    ):
        super(DualModel, self).__init__(loss, learning_rate)

        # create branches
        self.branch_grd = model_grd
        self.branch_sat = model_sat

        # check output dimension
        if self.branch_grd.output_dim != self.branch_sat.output_dim:
          raise ValueError("Mismatching output dimensions for the branches!")
        self.output_dim = self.branch_grd.output_dim


    def forward(self, batch):
        # elaborate the batch
        grd_imgs, _, _, sat_imgs, _ = batch

        # apply the models
        grd_emb = self.branch_grd(grd_imgs['imgs'])
        sat_emb = self.branch_sat(sat_imgs['imgs'])

        return (grd_emb, sat_emb), grd_emb, sat_emb


    def compute_loss(self, embeddings, grd_emb, sat_emb):
        loss = self.loss(grd_emb, sat_emb)
        return loss


    def __repr__(self):
        return f"DualModel(model_grd={self.branch_grd}, model_sat={self.branch_sat}, loss={self.loss})"
