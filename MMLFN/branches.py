### Imports
import pytorch_lightning as pl
from torch import nn
import torch
import torchvision.models as models
###



#@title VGG16
class VGGBranch(pl.LightningModule):
    def __init__(self, output_dim=1000, use_embeddings=False, freeze_layers=None,reset = False):
        super(VGGBranch, self).__init__()
        self.freeze_layers = freeze_layers
        self.output_dim = output_dim
        self.use_embeddings = use_embeddings
        self.reset = reset

        self.vgg16 = models.vgg16(pretrained=True)

        if self.use_embeddings:
            self.vgg16.classifier[-1] = nn.Linear(in_features=4096, out_features = self.output_dim, bias=True)
        else:
            self.vgg16.classifier = nn.Flatten()

        # Optionally freeze layers
        if self.freeze_layers:
            self.freeze_vgg_layers(self.freeze_layers)

    def forward(self, x):
        # Forward pass through feature extractor
        x = self.vgg16(x)
        return x

    def freeze_vgg_layers(self, num_layers):
        """
        Freeze the first `num_layers` layers of the VGG16 model.
        """
        layers = [layer for layer in self.vgg16.features]
        for i in range(num_layers):
            for param in layers[i].parameters():
                param.requires_grad = False
        if self.reset:
            for layer in layers[num_layers:]:
                for param in layer.parameters():
                    if param.dim() > 1:  # Apply Xavier initialization only to weight tensors
                        torch.nn.init.xavier_uniform_(param.data)

    def __repr__(self):
        return f"VGGBranch(freeze_layers={self.freeze_layers})"
    


# @title Resnet

class ResNetBranch(pl.LightningModule):
    def __init__(
        self,
        output_dim = 1000,
        use_embeddings=False,
        resnet_version=50,
        input_images = 1,
        freeze_layers = None,
        reset = False,
        conv_only=False
    ):
        super(ResNetBranch, self).__init__()
        self.output_dim = output_dim
        self.use_embeddings = use_embeddings
        self.resnet_version = resnet_version
        self.reset = reset
        self.freeze_layers = freeze_layers
        self.conv_only = conv_only
        self.input_images = input_images

        if resnet_version == 50:
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif resnet_version == 101:
            self.resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        elif resnet_version == 152:
            self.resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        else:
            raise ValueError("Unsupported ResNet version. Choose between 50, 101, or 152.")


        # Generalization to accept multiple input images
        if input_images > 1:
            weight = self.resnet.conv1.weight
            weight = list(weight for i in range(input_images))
            weight = torch.cat(weight, dim = 1)
            weight = torch.nn.Parameter(weight)
            self.resnet.conv1.weight = weight


        if not self.conv_only:
            if self.use_embeddings:
                self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.output_dim)
            else:
                self.resnet.fc = nn.Flatten()
        elif self.conv_only:
            # could remove last layers of resnet?
            pass

        if self.freeze_layers:
            self.freeze_resnet_layers(self.freeze_layers)

    def forward(self, x, featuremaps=False):
        # To print the featuremap we need to return the last conv layer output
        if featuremaps:
            for name, layer in list(self.resnet.named_children())[:-2]:
                x = layer(x)
                if x.shape[2]*x.shape[1]<128*128: #magic number: dimension 16 for visualization
                  break
            return x
        else:
            return self.resnet(x)

    def freeze_resnet_layers(self, num_layers):
        """
        Freeze the first `num_layers` layers of the ResNet model.
        If `reset` is True, reset the weights of the remaining layers.
        """
        layers = list(self.resnet.children())[:-2]  # Exclude avgpool and fc layers
        for i, layer in enumerate(layers[:num_layers]):
            for param in layer.parameters():
                param.requires_grad = False

        if self.reset:
            for layer in layers[num_layers:]:
                for param in layer.parameters():
                    if param.dim() > 1:  # Apply Xavier initialization only to weight tensors
                        torch.nn.init.xavier_uniform_(param.data)

    def __repr__(self):
        return (f"ResNetBranch(output_dim={self.output_dim}, use_embeddings={self.use_embeddings}, "
                f"resnet_version={self.resnet_version}, input_images={self.input_images}, "
                f"freeze_layers={self.freeze_layers}, reset={self.reset}, conv_only={self.conv_only})")
    


#@title ConvNeXt-B
class ConvNeXtBranch(pl.LightningModule):
    def __init__(self, output_dim=1000, use_embeddings=False, model_type="Tiny", freeze_layers = None, reset = False):
        super(ConvNeXtBranch, self).__init__()
        self.output_dim = output_dim
        self.use_embeddings = use_embeddings
        self.model_type = model_type
        self.freeze_layers = freeze_layers
        self.reset = reset

              # Initialize ConvNeXt model based on the specified model type
        if self.model_type == "Tiny":
            self.convNeXt = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
            in_features = 768
        elif self.model_type == "Small":  # Add this block for Small model
            self.convNeXt = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
            in_features = 768  # Adjusted input features for Small model
        elif self.model_type == "Base":
            self.convNeXt = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
            in_features = 1024
        elif self.model_type == "Large":
            self.convNeXt = models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT)
            in_features = 1536
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. Choose from 'Tiny', 'Small', 'Base', or 'Large'.")

         # Optionally freeze layers
        if self.freeze_layers:
            self.freeze_convnext_layers(self.freeze_layers)

        # Define the embedding layer with appropriate in_features
        if self.use_embeddings:
            # Replace the classifier with an identity layer or remove the last layer
            self.convNeXt.classifier[-1] = nn.Linear(in_features=in_features, out_features=self.output_dim, bias=True)
        else:
            self.convNeXt.classifier = nn.Flatten()

    def forward(self, x):
        # Forward pass through ConvNeXt model
        x = self.convNeXt(x)
        return x

    def freeze_convnext_layers(self, num_layers):
        """
        Freeze the first `num_layers` layers of the ConvNeXt model.
        """
        layers = [layer for layer in self.convNeXt.features]
        for i in range(min(num_layers, len(layers))):
            for param in layers[i].parameters():
                param.requires_grad = False
        if self.reset:
            for layer in layers[num_layers:]:
                for param in layer.parameters():
                    if param.dim() > 1:  # Apply Xavier initialization only to weight tensors
                        torch.nn.init.xavier_uniform_(param.data)

    def __repr__(self):
        return f"ConvNeXtBranch(output_dim={self.output_dim}, model_type={self.model_type}, reset = {self.reset})"