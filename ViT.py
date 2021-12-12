from torch import nn

from Architecture import *
from Architecture import PatchEmbedding
from device import default_device


class ViT(nn.Module):
    def __init__(self, hidden_size,
                 H, W, num_msa_heads,
                 patch_size, num_encoders,
                 mlp_expansion=2,
                 in_channels=3,
                 mlp_p_out=0.5,
                 num_classes=10,
                 device=default_device):
        super().__init__()
        self.to(device)
        self.hidden_size = hidden_size
        self.H = H
        self.W = W
        self.patch_size = patch_size
        self.num_msa_heads = num_msa_heads
        self.num_classes = num_classes
        self.mlp_expansion = mlp_expansion
        self.mlp_p_out = mlp_p_out
        self.in_channels = in_channels
        self.num_encoders = num_encoders
        self.vit = self.make_vit()

    def make_vit(self):
        layers = [PatchEmbedding(self.hidden_size, self.H, self.W, self.patch_size, self.in_channels)]
        for i in range(self.num_encoders):
            layers.append(TransformerEncoder(self.hidden_size, self.num_msa_heads, self.mlp_expansion, self.mlp_p_out).to(default_device))
        layers.append(MLPHead(self.num_classes, self.hidden_size).to(default_device))
        return nn.Sequential(*layers)

    def forward(self, X):
        return self.vit(X)
