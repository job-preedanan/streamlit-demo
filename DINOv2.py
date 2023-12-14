import torch
import torch.nn as nn


# Information of DINOv2 model : https://github.com/facebookresearch/dinov2
class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self, num_class, model_size='s', include_top=True):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.model_size = model_size
        self.embedding_size = 384 if model_size == 's' else 768
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        self.transformer = torch.hub.load('facebookresearch/dinov2', 'dinov2_vit' + self.model_size + '14', pretrained=False)
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_class),
        )
        self.include_top = include_top

    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        if self.include_top:
          x = self.classifier(x)
        return x

# usage example
# model = DinoVisionTransformerClassifier(num_class=4)
# model.transformer.requires_grad_(False)    # freeze backbone