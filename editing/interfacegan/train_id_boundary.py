"""
A script that trains a boundary specifically for a certain
identity. This is necessary as each video has a certain identity,
which we need to capture.
"""
import sys
sys.path.append('.')
sys.path.append('..')
from models.stylegan3.model import SG3Generator
from models.stylegan3.networks_stylegan3 import SynthesisNetwork
from argparse import ArgumentParser
import torch
import torchvision
from models.arcface import ArcFaceModel
import os
from tqdm import tqdm
from typing import List
import numpy as np
from editing.interfacegan.helpers.manipulator import train_boundary
from pathlib import Path

def parse_args(parser: ArgumentParser):
    parser.add_argument('--frames_folder', type=str,
                        help='The path to the video to the frames of the video to generate scores for')
    parser.add_argument('--arcface_weights_path', type=str,
                        help='The path to the ArcFace weights')
    parser.add_argument('--stylegan_pti_checkpoint_path', type=str,
                        help='The path to the stylegan checkpoint trained with PTI')
    parser.add_argument('--img_lat_vecs', type=str, help='The path to the latent vectors for the images in the list')
    parser.add_argument('--num_negative', type=int,
                        help='The number of negative images to generate')
    parser.add_argument('--seed', type=int, help='The seed for random generation', default=0)
    parser.add_argument('--output_path', type=str, help='Where to output the boundary information')

    return parser.parse_args()

def get_avg_video_id(arcface: ArcFaceModel, img_list: List[str], device: torch.device) -> torch.Tensor:
    """
    Given a list of images, computes the average identity for that list as a unit vector

    Arguments:
        arcface: A trained ArcFace model
        image_list: A list of paths to images that will be used
        device: The device to place tensors on

    Returns:
        The average identity of the list of images as a unit vector
    """
    id_vecs = torch.zeros(len(img_list), 512).to(device) # The size of an identity vector in ArcFace 
    for i, img_path in enumerate(tqdm(img_paths)):
        image = torchvision.io.read_image(img_path).to(device).unsqueeze(0).float()
        image = ((image / 255) - .5) / .5 # Needed to convert to what ArcFace was trained on

        with torch.no_grad():
            id_vec = arcface(image)
            id_vecs[i, :] = id_vec
        
    # Get the average identity as a comparison point
    avg_id = torch.mean(id_vecs, dim=0)
    avg_id = avg_id / torch.linalg.norm(avg_id) # Convert to a unit vector id

    return avg_id

def get_id_for_latents(latent_vecs: torch.Tensor, decoder: SG3Generator, arcface: ArcFaceModel, device: torch.device) -> torch.Tensor:
    """
    Computes the id vector after computing the images from
    a set of latent vectors.

    Arguments:
        latent_vecs: The latent vectors to comptue identities for
        decoder: The decoder for the latent vectors
        arcface: The trained arcface model for computing identities of each image
        device: The device to place all tensors on

    Returns:
        A tensor representing the identity vectors for each latent image
    """
    id_vecs = torch.zeros((latent_vecs.shape[0], 512)).to(device)

    for i, latent in enumerate(tqdm(latent_vecs)):
        with torch.no_grad():
            latent = latent.unsqueeze(0).to(device)
            img = decoder(latent)

            id_vec = arcface(img)

            id_vecs[i, :] = id_vec

    return id_vecs

def get_id_similarity(reference_id: torch.Tensor, test_ids: torch.Tensor):
    """
    Gets the cosine similarity between a reference identity and several test identities.
    This code assumes all identities are unit vectors.

    Arguments:
        reference_id: The id vector for the reference image. Should be a 1 x 512 tensor
        test_ids: The matrix of id vectors for the test images. Should be an n x 512 tensor

    Returns:
        A column tensor with the similarity score for each image
    """
    return (reference_id @ test_ids.T).T

if __name__ == '__main__':
    args = parse_args(ArgumentParser())
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Start by loading in ArcFace
    arcface = ArcFaceModel(args.arcface_weights_path, 'r50').to(device).eval()
    
    # Figure out the identity of this particular video. We'll take it to be the average identity vector
    # across the video
    img_paths = sorted(list(map(lambda x: os.path.join(args.frames_folder, x), os.listdir(args.frames_folder))))
    img_paths = img_paths[:5] # For testing

    avg_id = get_avg_video_id(arcface, img_paths, device)
    
    # Load the latent vectors for the image
    pos_lats = np.array(list(np.load(args.img_lat_vecs, allow_pickle=True)[()].values())) # It's an object array, so have to load like this
    pos_lats = torch.tensor(pos_lats)
    #pos_lats = pos_lats[:5] # For testing

    # Load the decoder with PTI for best id recognition
    sg3 = SG3Generator(args.stylegan_pti_checkpoint_path).decoder
    synthesis = sg3.synthesis.to(device)
    mapping = sg3.mapping.to(device)

    id_vecs = get_id_for_latents(pos_lats, synthesis, arcface, device)

    pos_similarities = get_id_similarity(avg_id, id_vecs)
    
    print(f'Average Positive example similarity: {torch.mean(pos_similarities)}')
    
    # Generate a number of likely negative examples
    neg_lats = torch.randn(args.num_negative, sg3.z_dim).to(device) # First, generate the Z vectors
    
    with torch.no_grad():
        neg_lats = mapping(neg_lats, None, truncation_psi=.7)

    neg_id_vecs = get_id_for_latents(neg_lats, synthesis, arcface, device)

    neg_similarities = get_id_similarity(avg_id, neg_id_vecs)

    print(f'Average negative example similarity: {torch.mean(neg_similarities)}')

    # Combine the latent vectors into a single array
    latents = torch.vstack([pos_lats.cpu(), neg_lats.cpu()]).numpy()
    similarities = torch.cat([pos_similarities.cpu(), neg_similarities.cpu()], dim=0).unsqueeze(1).numpy()
    
    boundary = train_boundary(latent_codes=latents[:, 0, :], # The original code only uses the first w
                              scores=similarities,
                              chosen_num_or_ratio=0.02,
                              split_ratio=0.7,
                              invalid_value=None)
    np.save(Path(args.output_path) / f'arcface_boundary.npy', boundary)


