import numpy as np
import torch
import torch.nn as nn

class LinearProbe(nn.Module):
    def __init__(self, pretrained_model, hidden_dim, num_classes, use_batch_norm=False):
        super().__init__()
        self.encoder = pretrained_model

        # Freeze the encoder's parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(hidden_dim, num_classes)
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_dim)
        else:
            self.batch_norm = nn.Identity()


    def forward(self, x):
        
        with torch.no_grad():
            x = self.pretrained_model(x)
            x = x.mean(dim=1)

        x = self.batch_norm(x)

        return self.linear(x)
