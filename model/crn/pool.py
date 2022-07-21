import torch
import torch.nn as nn

class TemporalPooling(nn.Module):
    def __init__(self, d_model, dropout_ratio):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(2048, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
            nn.Dropout(dropout_ratio)
        )

    def forward(self, x):
        # b x w x d_model -> b x d_model
        x = torch.mean(x, dim=1)
        # b x d_model -> b x 2
        x = self.mlp(x)

        return x