import torch
import torch.nn as nn
from transformers import ViTModel

class FoodViT(nn.Module):
    def __init__(self, num_ingredients, num_nutrients, feature_dim=0):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.dropout = nn.Dropout(0.2)
        vit_embed_dim = self.vit.config.hidden_size

        if feature_dim > 0:
            self.feature_proj = nn.Linear(feature_dim, 128)
            combined_dim = vit_embed_dim + 128
        else:
            self.feature_proj = None
            combined_dim = vit_embed_dim

        self.ingredients_head = nn.Linear(combined_dim, num_ingredients)
        self.nutrients_head = nn.Linear(combined_dim, num_nutrients)

    def forward(self, x, extra_features=None):
        outputs = self.vit(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0]
        cls_token = self.dropout(cls_token)

        if self.feature_proj is not None and extra_features is not None:
            feat_proj = torch.relu(self.feature_proj(extra_features))
            combined = torch.cat([cls_token, feat_proj], dim=1)
        else:
            combined = cls_token

        ingredients = torch.sigmoid(self.ingredients_head(combined))
        nutrients = self.nutrients_head(combined)
        return ingredients, nutrients
