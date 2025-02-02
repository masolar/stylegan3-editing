import torch
from torchvision.transforms import ToTensor, ToPILImage
from typing import Iterable, Optional
from pathlib import Path
import numpy as np
from PIL import Image


def crop_transform(frame: torch.Tensor, alignment_transform: np.ndarray) -> Optional[torch.Tensor]:
    """
    A typed, functional version of the image transform found in video_handler.py

    Arguments:
        frame_path: The path to a frame
        alignment_transform: A transform matrix
    """
    try:
        curr_im = ToPILImage()(frame)
        # curr_im = (frame * 255).permute(1, 2, 0).to(torch.uint8).numpy()
        curr_im = curr_im.transform(
            (1024, 1024), Image.QUAD, alignment_transform, Image.BILINEAR)

        return ToTensor()(curr_im)

    except Exception as e:
        print(e)
        return None
