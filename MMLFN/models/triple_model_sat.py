### Import


###



#@title Triple Model satellite using multiple losses

class TripleModel_sat(MultiBranchModel):

    def __init__(
        self,
        model_grd,
        model_sat,
        model_sat_seg,
        loss=InfoNCE(loss_function=nn.CrossEntropyLoss()),
        fully_concat=True,
        multiple_losses=False,
        auxilary_loss_weight = 0.5,
        learning_rate = 0.0001,
    ):
        super(TripleModel_sat, self).__init__(loss, learning_rate)

        # create branches
        self.branch_grd = model_grd
        self.branch_sat = model_sat
        self.branch_sat_seg = model_sat_seg

        # check output dimension
        sat_output_dim = self.branch_sat.output_dim + self.branch_sat_seg.output_dim
        if self.branch_grd.output_dim != sat_output_dim:
            raise ValueError("Mismatching output dimensions for the branches!")
        self.output_dim = self.branch_grd.output_dim

        self.fully_concat = fully_concat
        self.multiple_losses = multiple_losses

        # To make concatenation learnable
        if fully_concat:
            self.fc = nn.Linear(self.output_dim, self.output_dim)

        self.auxilary_loss_weight = auxilary_loss_weight #used for multiple losses only



    def forward(self, batch):
        # elaborate the batch
        grd_imgs, _, _, sat_imgs, sat_seg = batch

        # apply the models
        grd_emb = self.branch_grd(grd_imgs['imgs'])
        sat_rgb_emb = self.branch_sat(sat_imgs['imgs'])
        sat_seg_emb = self.branch_sat_seg(sat_seg['imgs'])

        # compute the total embeddings
        sat_emb = torch.cat((sat_rgb_emb, sat_seg_emb), dim=1)
        if self.fully_concat:
            sat_emb = self.fc(sat_emb)

        return (grd_emb, sat_rgb_emb, sat_seg_emb), grd_emb, sat_emb


    def compute_loss(self, embeddings, grd_emb, sat_emb):
        loss = self.loss(grd_emb, sat_emb)
        if self.multiple_losses:
            loss += self.auxilary_loss_weight*self.loss(embeddings[1], embeddings[2])

        return loss


    def __repr__(self):
        return f"""TripleModel_sat(model_grd={self.branch_grd}, model_sat={self.branch_sat},
        model_sat_seg={self.branch_sat_seg}, loss={self.loss}, fully_concat={self.fully_concat},
        multiple_losses={self.multiple_losses}, auxilary_loss_weight = {self.auxilary_loss_weight})"""