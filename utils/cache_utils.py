'''
Utility functions related to caching things like images and videos
'''
import torch
from torchvision.io import write_jpeg
from typing import Iterable, Union, Tuple
from pathlib import Path


def cache_images(generator: Iterable[Tuple[torch.Tensor, Union[Path, str, bytes]]]) -> Iterable[Tuple[torch.Tensor, Union[Path, str, bytes]]]:
    '''
    A simple wrapper over a generator that writes an image Tensor to its associated path

    Arguments:
        generator: The generator that produces image/filepath pairs. Images are assumed to be n x H x W with range [0, 1]

    Returns:
        The same pairs after they have been saved
    '''
    for image, image_path in generator:
        # Convert image to the form PyTorch needs for saving
        image = image * 255
        image = image.to(torch.uint8)

        write_jpeg(image, str(image_path))

        yield image, image_path
