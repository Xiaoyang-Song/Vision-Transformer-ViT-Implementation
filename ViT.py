from Architecture import *


class ViT(nn.Module):
    def __init__(self, hidden_size,
                 H, W, num_msa_heads,
                 patch_size, mlp_expansion,
                 num_encoders,
                 in_channels=3,
                 mlp_p_out=0.5,
                 num_classes=10):
        self.D = hidden_size
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
            layers.append(TransformerEncoder(self.hidden_size, self.num_msa_heads, self.mlp_expansion, self.mlp_p_out))
        layers.append(MLPHead(self.num_classes, self.num_msa_heads))
        return nn.Sequential(*layers)

    def forward(self, X):
        return self.vit(X)
