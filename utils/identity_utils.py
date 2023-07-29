from typing import Callable, Iterable
import torch


def get_avg_id(images: Iterable[torch.Tensor], identifier: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """
    Gets the average identity found in a series of images. This could
    be useful in finding the average identity of video frames, for instance.

    This function assumes that each tensor is on the same device.
    Arguments:
        images: An iterable of images. It's assumed that each tensor is n x 3 x height x width, where n is the batch size
        identity: A function that takes a tensor image and gives an identity vector for it

    Returns:
        The average identity vector for the images with shape 1 x latent_dim
        This will be a unit vector
    """
    image_iter = iter(images)
    avg_id = identifier(next(image_iter))
    num_vecs = avg_id.shape[0]

    avg_id = avg_id.sum(dim=0)  # Remove the batch dimension only

    for img in image_iter:
        ids = identifier(img)
        num_vecs += ids.shape[0]

        avg_id += ids.sum(dim=0)

    # Get the average vector
    return avg_id.unsqueeze(0) / num_vecs
