### Import
from models.generic_model import MultiBranchModel
from losses import InfoNCE
from torch import nn
import torch

###



# @title Quintuple Model

class QuintupleModel(MultiBranchModel):
    def __init__(
        self,
        model_grd,
        model_grd_seg,
        model_grd_dep,
        model_sat,
        model_sat_seg,
        loss=InfoNCE(loss_function=nn.CrossEntropyLoss()),
        multiple_losses=False,
        fully_concat_grd=True,
        fully_concat_sat=True,
        auxilary_loss_weight = 0.5,
        learning_rate = 0.0001,
    ):
        super(QuintupleModel, self).__init__(loss, learning_rate)

        # create branches
        self.branch_grd = model_grd
        self.branch_grd_seg = model_grd_seg
        self.branch_grd_dep = model_grd_dep
        self.branch_sat = model_sat
        self.branch_sat_seg = model_sat_seg

        # check output dimension
        grd_output_dim = self.branch_grd.output_dim + self.branch_grd_seg.output_dim + self.branch_grd_dep.output_dim
        sat_output_dim = self.branch_sat.output_dim + self.branch_sat_seg.output_dim
        if grd_output_dim != sat_output_dim:
          raise ValueError("Mismatching output dimensions for the branches!")
        self.output_dim = grd_output_dim

        self.fully_concat_grd = fully_concat_grd
        self.fully_concat_sat = fully_concat_sat

        self.multiple_losses = multiple_losses

        # To make concatenations learnable
        if self.fully_concat_grd:
          self.fc_grd = nn.Linear(self.output_dim, self.output_dim)

        if self.fully_concat_sat:
          self.fc_sat = nn.Linear(self.output_dim, self.output_dim)

        self.auxilary_loss_weight = auxilary_loss_weight


    def forward(self, batch):
        # elaborate the batch
        grd_img, grd_seg, grd_dep, sat_img, sat_seg = batch

        # apply the models
        grd_rgb_emb = self.branch_grd(grd_img['imgs'])
        grd_seg_emb = self.branch_grd_seg(grd_seg['imgs'])
        grd_dep_emb = self.branch_grd_dep(grd_dep['imgs'])
        sat_rgb_emb = self.branch_sat(sat_img['imgs'])
        sat_seg_emb = self.branch_sat_seg(sat_seg['imgs'])

        # compute the total embeddings
        grd_emb = torch.cat((grd_rgb_emb, grd_seg_emb, grd_dep_emb), dim=1)
        sat_emb = torch.cat((sat_rgb_emb, sat_seg_emb), dim=1)
        if self.fully_concat_grd:
          grd_emb = self.fc_grd(grd_emb)
        if self.fully_concat_sat:
          sat_emb = self.fc_sat(sat_emb)

        return (grd_rgb_emb, grd_seg_emb, grd_dep_emb, sat_rgb_emb, sat_seg_emb), grd_emb, sat_emb


    def compute_loss(self, embeddings, grd_emb, sat_emb):
        loss = self.loss(grd_emb, sat_emb)
        if self.multiple_losses:
            loss += self.auxilary_loss_weight*self.loss(embeddings[0], embeddings[1])
            loss += self.auxilary_loss_weight*self.loss(embeddings[0], embeddings[2])
            loss += self.auxilary_loss_weight*self.loss(embeddings[1], embeddings[2])
            loss += self.auxilary_loss_weight*self.loss(embeddings[3], embeddings[4])

        return loss


    def __repr__(self):
        return f"""QuintupleModel(model_grd={self.branch_grd}, model_grd_seg={self.branch_grd_seg},
        model_grd_dep={self.branch_grd_dep}, model_sat={self.branch_sat}, model_sat_seg={self.branch_sat_seg},
        loss={self.loss}, multiple_losses={self.multiple_losses},
        fully_concat_grd={self.fully_concat_grd}, fully_concat_sat={self.fully_concat_sat}, auxilary_loss_weight = {self.auxilary_loss_weight})"""
