import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from vit_model import TransformerEncoderLayer, PositionalEncoding

class CNNFeatureExtractor(nn.Module):
    """CNN backbone for feature extraction."""
    
    def __init__(self, backbone='resnet18', feature_dim=512, pretrained=True):
        super().__init__()
        
        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            # Remove the final fully connected layer
            self.features = nn.Sequential(*list(resnet.children())[:-2])
            self.feature_channels = 512
        elif backbone == 'efficientnet':
            efficientnet = models.efficientnet_b0(pretrained=pretrained)
            self.features = efficientnet.features
            self.feature_channels = efficientnet.classifier[1].in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        # Adaptive pooling to get fixed size features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
        self.feature_proj = nn.Conv2d(self.feature_channels, feature_dim, 1)
        
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.feature_proj(x)
        
        # Flatten spatial dimensions
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(1, 2)  # (B, H*W, C)
        return x, H, W

class HybridCNNTransformer(nn.Module):
    """Hybrid model combining CNN features with Transformer encoder."""
    
    def __init__(self, num_classes=10, backbone='resnet18', 
                 embed_dim=512, depth=6, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.cnn_backbone = CNNFeatureExtractor(backbone, embed_dim)
        num_patches = 14 * 14  # From adaptive pooling
        
        self.pos_encoding = PositionalEncoding(num_patches, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x, return_attention=False):
        # Extract CNN features
        x, H, W = self.cnn_backbone(x)
        
        # Add positional encoding and CLS token
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through transformer encoder
        attention_weights = []
        for layer in self.encoder_layers:
            x, attn = layer(x, return_attention)
            if return_attention:
                attention_weights.append(attn)
        
        # Use CLS token for classification
        x = self.norm(x)
        cls_token = x[:, 0]
        logits = self.head(cls_token)
        
        if return_attention:
            return logits, attention_weights, H, W
        return logits

if __name__ == "__main__":
    # Test hybrid model
    model = HybridCNNTransformer(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    logits, attn_weights, H, W = model(x, return_attention=True)
    print(f"Output shape: {logits.shape}")
    print(f"Feature map size: {H}x{W}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")