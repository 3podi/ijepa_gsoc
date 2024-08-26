import numpy as np
import torch
import torch.nn as nn

from src.models.resnet import ResNet50
import src.models.vision_transformer as vit


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
            x = self.encoder(x)
            x = x.mean(dim=1)

        x = self.batch_norm(x)

        return self.linear(x)
    
class AddLinear(nn.Module):
    def __init__(self, encoder, hidden_dim, num_classes, use_batch_norm=False):
        super().__init__()
        self.encoder = encoder
        self.linear = nn.Linear(hidden_dim,num_classes)

        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_dim)
        else:
            self.batch_norm = nn.Identity()
    
    def forward(self,x):
        
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.batch_norm(x)

        return self.linear(x)

    

def resnet_50(num_classes, classification_head):
    return ResNet50(num_classes=num_classes,classification_head=classification_head)

def vit_model(num_classes, use_batch_norm, img_size, patch_size,model_name):
    encoder = vit.__dict__[model_name](img_size=[img_size],patch_size=patch_size)
    embed_dim = encoder.embed_dim

    return AddLinear(encoder, embed_dim, num_classes, use_batch_norm)





