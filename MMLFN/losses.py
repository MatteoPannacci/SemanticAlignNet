### Imports


###


#@title Implementation SoftMarginTripletLoss

class TripletLoss(pl.LightningModule):

    def __init__(self, loss_weight = 1.0, symmetric = True):
        super().__init__()
        self.loss_weight = loss_weight
        self.symmetric = symmetric

    def forward(self, image_features1, image_features2):

        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)

        dist_array = 2.0 - 2.0 * torch.matmul(image_features2, image_features1.T)
        n = len(image_features1)
        pos_dist = torch.diag(dist_array)
        pair_n = n * (n - 1.0)

        triplet_dist_g2s = pos_dist - dist_array
        loss_g2s = torch.sum(torch.log(1.0 + torch.exp(triplet_dist_g2s * self.loss_weight)))/pair_n

        if not self.symmetric:
            return loss_g2s

        else:
            triplet_dist_s2g = torch.unsqueeze(pos_dist, 1) - dist_array
            loss_s2g = torch.sum(torch.log(1.0 + torch.exp(triplet_dist_s2g * self.loss_weight)))/pair_n
            return (loss_g2s + loss_s2g) / 2.0
        


#@title InfoNCELoss
class InfoNCE(pl.LightningModule):
    def __init__(self, loss_function, logit_scale=3.0, learnable_logit_scale=True, symmetric=True):
        super().__init__()

        self.loss_function = loss_function
        self.symmetric = symmetric
        if learnable_logit_scale:
            self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        else:
            self.logit_scale = logit_scale

    def forward(self, image_features1, image_features2):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)

        logits_per_image1 = self.logit_scale * image_features1 @ image_features2.T
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=image_features1.device)
        loss_g2s = self.loss_function(logits_per_image1, labels)

        if not self.symmetric:
            return loss_g2s

        else:
            logits_per_image2 = logits_per_image1.T
            loss_s2g = self.loss_function(logits_per_image2, labels)
            return (loss_g2s + loss_s2g) / 2.0
        


#@title Combined Loss
class CombinedLoss(pl.LightningModule):
    def __init__(self, loss_function, logit_scale=3.0, loss_weight=0.5, symmetric=True):
        super().__init__()

        # Initialize InfoNCE Loss
        self.info_nce_loss = InfoNCE(loss_function, logit_scale, symmetric = symmetric)

        # Initialize SoftMarginTripletLoss
        self.triplet_loss = TripletLoss(10.0, symmetric = symmetric)

        self.loss_weight = loss_weight
        self.symmetric = symmetric

    def forward(self, image_features1, image_features2):

        # Compute InfoNCE loss
        info_nce_loss_value = self.info_nce_loss(image_features1, image_features2)

        # Compute Triplet loss
        triplet_loss_value = self.triplet_loss(image_features1, image_features2)

        # Combine the losses
        combined_loss = (1-self.loss_weight)*info_nce_loss_value + self.loss_weight * triplet_loss_value

        return combined_loss