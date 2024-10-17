### Import
from generic_model import MultiBranchModel
from .losses import InfoNCE
from torch import nn
import torch

###



#@title Triple Model ground with multiple losses
class TripleModel_grd(MultiBranchModel):

    def __init__(
        self,
        model_grd,
        model_sat,
        model_grd_seg,
        loss=InfoNCE(loss_function=nn.CrossEntropyLoss()),
        fully_concat=True,
        multiple_losses=False,
        auxilary_loss_weight = 0.5,
        learning_rate = 0.0001,
    ):
        super(TripleModel_grd, self).__init__(loss, learning_rate)

        # create branches
        self.branch_grd = model_grd
        self.branch_grd_seg = model_grd_seg
        self.branch_sat = model_sat

        # check output dimension
        grd_output_dim = self.branch_grd.output_dim + self.branch_grd_seg.output_dim
        if self.branch_sat.output_dim != grd_output_dim:
            raise ValueError("Mismatching output dimensions for the branches!")
        self.output_dim = self.branch_sat.output_dim

        self.fully_concat = fully_concat
        self.multiple_losses = multiple_losses

        # to make concatenation learnable
        if fully_concat:
            self.fc = nn.Linear(self.output_dim, self.output_dim)

        self.auxilary_loss_weight = auxilary_loss_weight #used for multiple losses only

        self.loss = loss


    def forward(self, batch):
        # elaborate the batch
        grd_imgs, grd_seg, _, sat_imgs, _ = batch

        # apply the models
        grd_rgb_emb = self.branch_grd(grd_imgs['imgs'])
        grd_seg_emb = self.branch_grd_seg(grd_seg['imgs'])
        sat_emb = self.branch_sat(sat_imgs['imgs'])

        # compute the total embeddings
        grd_emb = torch.cat((grd_rgb_emb, grd_seg_emb), dim=1)
        if self.fully_concat:
            grd_emb = self.fc(grd_emb)

        return (grd_rgb_emb, grd_seg_emb, sat_emb), grd_emb, sat_emb


    def compute_loss(self, embeddings, grd_emb, sat_emb):
        loss = self.loss(grd_emb, sat_emb)
        if self.multiple_losses:
            loss += self.auxilary_loss_weight*self.loss(embeddings[0], embeddings[1])

        return loss

    def __repr__(self):
        return f"""TripleModel_grd(model_grd={self.branch_grd}, model_sat={self.branch_grd_seg},
        model_sat_seg={self.branch_sat}, loss={self.loss}, fully_concat={self.fully_concat},
        multiple_losses={self.multiple_losses}, auxilary_loss_weight = {self.auxilary_loss_weight})"""
