"""
The ArcFace model. It should be noted that this network gives unit
vectors back, rather than the raw inputs.
"""
import torch.nn as nn
import torch
from torchvision.transforms import Resize
from insightface.recognition.arcface_torch.backbones import get_model


class ArcFaceModel(nn.Module):
    def __init__(self, weights_location: str, model_type: str):
        """
        Creates an ArcFace network from pretrained weights and network type
        """
        super(ArcFaceModel, self).__init__()
        self.network = get_model(model_type, fp16=True)

        self.network.load_state_dict(torch.load(weights_location))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The network was trained on images that are 112 x 112 and have inputs
        [-1, 1], so there's a scaling that needs to be done before passing things in
        """
        # This is the size given in the ArcFace docs
        resized = Resize((112, 112))(x)
        output = self.network(resized)

        # Normalize into a unit vector
        magnitudes = torch.sqrt(torch.sum(output ** 2, dim=1)).unsqueeze(1)

        return output / magnitudes
