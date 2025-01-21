import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TransformerEncoderBlock(nn.Module):
    """
    A single Transformer encoder block: Multi-Head Self-Attention + MLP.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, seq_len, embed_dim)
        # 1) Multi-head self-attention
        x_norm = self.norm1(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + x_attn  # Residual connection
        
        # 2) MLP
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        x = x + x_mlp  # Residual connection
        
        return x


class ViTExtractor(BaseFeaturesExtractor):
    """
    A Vision Transformer (ViT) feature extractor for observations of shape (C, H, W).
    Produces a final feature vector of size `features_dim`.
    """
    def __init__(
        self,
        observation_space,
        features_dim=128,
        patch_size=10, # 5
        embed_dim=64,
        num_heads=4,
        num_layers=4, # 4 
        mlp_ratio=4.0,
        dropout=0.0,
        use_cls_token=True
    ):
        super().__init__(observation_space, features_dim)
        
        self.C, self.H, self.W = observation_space.shape  # e.g. (5, 50, 50)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token

        assert self.H % patch_size == 0, "Observation height not divisible by patch_size"
        assert self.W % patch_size == 0, "Observation width not divisible by patch_size"
        
        # Number of patches in the spatial grid
        self.num_patches_h = self.H // patch_size
        self.num_patches_w = self.W // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Simple linear embedding: each patch is flattened and then linearly projected
        self.patch_embed = nn.Conv2d(
            in_channels=self.C,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        # After this convolution, the shape is (B, embed_dim, H/patch_size, W/patch_size).
        # We'll then flatten to (B, num_patches, embed_dim).

        self.seq_length = self.num_patches + (1 if use_cls_token else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim))

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, features_dim),
            nn.ReLU()
        )

        self._init_weights()

        with torch.no_grad():
            sample_input = torch.zeros((1, self.C, self.H, self.W))
            sample_features = self.forward(sample_input)
            assert sample_features.shape == (1, features_dim), (
                f"Expected (1, {features_dim}), got {sample_features.shape}"
            )

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        :param observations: shape (B, C, H, W)
        :return: a feature tensor of shape (B, features_dim)
        """
        B = observations.shape[0]

        # 1) Patch embedding => (B, embed_dim, H/patch_size, W/patch_size)
        x = self.patch_embed(observations)
        
        # 2) Flatten spatial dimensions => (B, embed_dim, num_patches)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        
        # 3) Transpose => (B, num_patches, embed_dim)
        x = x.transpose(1, 2)

        # 4) If we use a CLS token, prepend it
        if self.use_cls_token:
            # cls_token is (1, 1, embed_dim), expand over batch => (B, 1, embed_dim)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)  # => (B, 1 + num_patches, embed_dim)

        # 5) Add learnable positional embeddings
        x = x + self.pos_embed[:, : x.shape[1], :]  # broadcast over batch

        # 6) Pass through each Transformer encoder block
        for block in self.blocks:
            x = block(x)

        # 7) Final layer norm
        x = self.norm(x)

        # 8) Pool the output (e.g., take the CLS token if available, or mean pool)
        if self.use_cls_token:
            # x[:, 0] is the CLS token
            x = x[:, 0]  # => (B, embed_dim)
        else:
            # Alternatively: mean pool
            x = x.mean(dim=1)  # => (B, embed_dim)

        # 9) Final projection to features_dim
        out = self.fc(x)  # => (B, features_dim)
        return out
