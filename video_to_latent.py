"""
This script takes in video frames and converts them to latent vectors
using an encoder
"""
from itertools import takewhile
import pyrallis
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from pathlib import Path
from typing import Iterable, Tuple, Optional
from torchvision.io import write_video
from models.stylegan3.model import SG3Generator
from models.stylegan3.networks_stylegan3 import Generator
from utils.data_utils import ImageAndTransformsDataset
from utils.inference_utils import get_average_image, IMAGE_TRANSFORMS, load_encoder, run_on_batch_senseful
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from utils.video import generate_mp4


@dataclass
class LatentOpts:
    # The path containing video frames to encode
    frames_path: Path
    # The path to output latent vectors and videos
    output_path: Path
    # The path to the inverse transforms file
    inv_trans_path: Path
    # Encoder weights path
    enc_weights_path: Path
    # Number of iters to go per batch
    n_iters_per_batch: int = 3
    # The path to the generator weights
    generator_path: Path = Path('pretrained_models/sg3-r-ffhq-1024.pt')


def compute_latents_and_img(img_trans_pair: Tuple[torch.Tensor, torch.Tensor], encoder: nn.Module, avg_image: torch.Tensor) -> torch.Tensor:
    """
    A function to generate latent vectors from an image

    Arguments:
        img_trans_pairs: An image and transform to turn to latent vectors
        encoder: The encoder for the images
        avg_image: The average image for the encoder

    Returns:
        The calculated images, the unaligned images, and the latent codes for these images
    """
    # Create the vector that will be passed in using the average image
    with torch.no_grad():
        img, trans = img_trans_pair
        img = IMAGE_TRANSFORMS(img)

        y_hat = avg_image.repeat(img.shape[0], 1, 1, 1)
        latent = None

        for iter in range(3):
            x_input = torch.cat([img, y_hat], dim=1)

            is_last_iteration = iter == 2

            res = encoder.forward(x_input,
                                  latent=latent,
                                  landmarks_transform=trans,
                                  return_aligned_and_unaligned=True,
                                  return_latents=True,
                                  resize=False)
            if is_last_iteration:
                _, y_hat, latent = res
            else:
                y_hat, _, latent = res

            y_hat = encoder.face_pool(y_hat)

        return latent


def latents_to_image(generator: SG3Generator, latent: torch.Tensor, landmarks_transform: Optional[torch.Tensor]) -> torch.Tensor:
    '''
    Converts a latent vector to an image tensor with range [0, 1]

    Arguments:
        generator: The network for generating images
        latent: The latent vector to create an image from
        landmarks_transform: The inverse transform from aligned to cropped image

    Returns:
        A tensor representing the output image, with range [0, 1]
    '''
    generator.input.transforms = landmarks_transform

    if len(latent.shape) == 2:
        latent = latent.unsqueeze(0)

    output_image = generator(latent)

    return output_image


@pyrallis.wrap()
def main(opts: LatentOpts):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def device_fn(tensor): return tensor.to(device)

    # Load in the encoder for images
    enc, _ = load_encoder(opts.enc_weights_path)
    # enc = torch.load(opts.enc_weights_path, map_location='cpu')
    # enc.eval()

    # enc.to(device)

    # Load in the data and transforms
    dataset = ImageAndTransformsDataset(
        opts.frames_path, torch.load(opts.inv_trans_path))
    data = tqdm(DataLoader(dataset, batch_size=8))

    # For debugging
    # data = map(lambda entry: entry[1], takewhile(
    #     lambda x: x[0] < 10, enumerate(data)))
    # Get the average image of the encoder
    with torch.no_grad():
        avg_img = get_average_image(enc)

    # Calculate latent vectors for the input images
    output_gen = map(lambda entry: compute_latents_and_img(
        entry, enc, avg_img), map(lambda entry: (device_fn(entry[0]), device_fn(entry[1])), data))

    with torch.no_grad():
        latents = torch.cat([output for output in output_gen])

    # imgs = torch.cat([entry[0] for entry in outputs]).cpu()
    # unaligned = torch.cat([entry[1] for entry in outputs]).cpu()
    # latents = torch.cat([entry[1] for entry in outputs]).cpu()

    # generate_mp4(opts.output_path / 'aligned_vid.mp4', imgs, fps=24)
    # generate_mp4(opts.output_path / 'unaligned_vid.mp4', unaligned, fps=24)
    torch.save(latents, str(opts.output_path / 'latents.pth'))

    # We can remove the previous network to save some memory
    del enc

    # Load in the SG3Generator
    G = SG3Generator(opts.generator_path).decoder

    # Get the mapping and synthesis portions
    synth_net = G.synthesis.to(device)
    synth_net.eval()

    with torch.no_grad():
        # Now we can generate the output videos to show this worked
        gen_frames = torch.cat([latent.cpu() for latent in map(
            lambda latent: latents_to_image(synth_net, latent, None), latents)])

        gen_frames = (gen_frames / 2) + .5

    generate_mp4(str(opts.output_path / 'generated.mp4'), gen_frames)

    orig_frames = torch.cat([image for image, _ in DataLoader(ImageAndTransformsDataset(
        opts.frames_path, torch.load(opts.inv_trans_path)))])

    # For debugging
    # orig_frames = orig_frames[:80, :, :, :]

    combined_generator = zip(orig_frames, gen_frames)
    entry = next(combined_generator)
    combined_generator = map(lambda entry: torch.cat(
        [entry[0], entry[1]], dim=-1), combined_generator)
    print(f'First entry: {entry[0].shape}, second entry: {entry[1].shape}')
    # combined_frames = torch.cat([orig_frames, gen_frames], dim=-1)
    generate_mp4(str(opts.output_path / 'combined.mp4'),
                 combined_generator, size=(2048, 1024))


if __name__ == '__main__':
    main()
