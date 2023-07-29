"""
The same file as before, but saves things in a format that GradCtrl can actually read
"""

from dataclasses import dataclass
import sys
sys.path.append(".")
sys.path.append("..")
import torch
import pyrallis
import numpy as np
from typing import Callable, Iterable, List
from pathlib import Path
import pickle
from configs.paths_config import model_paths
from editing.interfacegan.helpers import anycostgan
from editing.interfacegan.helpers.pose_estimator import PoseEstimator
from editing.interfacegan.helpers.age_estimator import AgeEstimator
from editing.interfacegan.helpers.id_estimator import ArcFace
from models.stylegan3.networks_stylegan3 import Generator, SynthesisNetwork
from editing.interfacegan.helpers.anycostgan import attr_list
from models.stylegan3.model import SG3Generator
from tqdm import tqdm
from torchvision.transforms import Resize
from utils.identity_utils import get_avg_id
import torchvision
import os
import math

@dataclass
class EditConfig:
    # Path to StyleGAN3 generator
    generator_path: Path = Path(model_paths["stylegan3_ffhq"])
    # Number of latents to sample
    n_images: int = 500000
    # Number of latents to sample as negative samples for the id detector
    n_neg_images: int = 3000
    # Truncation psi for sampling
    truncation_psi: float = 0.7
    # Where to save the `npy` files with latents and scores to
    output_path: Path = Path("./latents")
    # The path to the frames of an actual video to get the identity for
    video_frames: Path = Path("./video_frames")
    # The path to the video latent vectors
    video_latents_paths: Path=Path("./video_latents")
    
    # The batch size for the average id compute
    avg_id_batch_size: int = 10

    # The path to the arcface weights
    arcface_weights: Path = Path("./arcface")
    
    # Whether to get logits for the other attributes.
    # Allows you to only get logits for each identity, rather than new
    # logits for each attribute as well.
    # TODO: Split this into two generating files
    attr_logits: bool = False

    num_attr_images: int = 50000


@pyrallis.wrap()
def run(opts: EditConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_fn = lambda x: x.to(device)

    opts.output_path.mkdir(exist_ok=True, parents=True)

    # First, we need to generate our latent vectors to be scored
    
    # Load the arcface network
    arcface = ArcFace({'model_path': opts.arcface_weights}).to(device).eval()

    # For the identity scores, it needs to be in reference to an actual identity, so we'll compute the
    # identity vector for an actual video
    image_frame_paths = list(map(lambda x: os.path.join(opts.video_frames, x), os.listdir(opts.video_frames)))
    num_image_frame_paths = len(image_frame_paths)
    print(num_image_frame_paths)
    actual_frame_generator = map(arcface_transform, 
                                map(device_fn,
                                    batch_function(
                                        map(torchvision.io.read_image, 
                                            image_frame_paths
                                            ), 
                                        opts.avg_id_batch_size
                                    )
                                )
                            )
    
    with torch.no_grad():
        print(f'Calculating average id for video: {opts.video_frames}')
        avg_video_id = get_avg_id(
                                tqdm(
                                    actual_frame_generator, 
                                    total=num_image_frame_paths // opts.avg_id_batch_size
                                ), 
                                arcface
                        )
        
        torch.save(avg_video_id, opts.output_path / 'avg_id.pt')

        # Load the mapping network and stylegan network
        G = SG3Generator(opts.generator_path).decoder
        
        # Get the mapping and synthesis portions
        mapping_net = G.mapping.to(device)
        mapping_fn = lambda x: mapping_net(x, None, truncation_psi=opts.truncation_psi)
        synth_net = G.synthesis.to(device)

        """
        The binary classifier needs things to be negative and positive to work, so
        we subtract the amount ArcFace uses to say whether a face is positive or not

        This should give it negatives and positives to use
        """
        id_learner = lambda x: calc_id_score(x, avg_video_id) - math.cos(math.pi / 3)
        
        # We need a generator that creates random vectors
        # This one creates a generator based on our parameters
        random_latent_generator_creator = lambda x,z_dim: (torch.from_numpy(np.random.randn(z_dim)) for _ in range(x))

        # This is a generator that creates W+ vectors from a W vector
        wplus_latent_generator = map(mapping_fn, map(device_fn, batch_function(random_latent_generator_creator(opts.n_images, G.z_dim), opts.avg_id_batch_size)))

        # This generator reads in the latent vector already computed for the actual video frames
        video_frame_latent_generator = map(device_fn, batch_function((vector for vector in torch.from_numpy(np.array(list(np.load(opts.video_latents_paths, allow_pickle=True)[()].values())))), opts.avg_id_batch_size))
        
        face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        
        # Generate the latent vectors so that we can work with them
        video_frame_latents = torch.cat(list(video_frame_latent_generator))
        wplus_latents = torch.cat(list(wplus_latent_generator))
        
        # Write out the latent vectors
        save_vec(opts.output_path / 'identity_latents.pkl', torch.cat([video_frame_latents, wplus_latents]).cpu().numpy())

        latent_to_image = lambda x: map(face_pool, map(synth_net, x))
        image_to_id = lambda x: map(arcface, map(Resize((112, 112)), x))
        
        id_similarities = lambda x: map(id_learner, image_to_id(latent_to_image(x)))
        #id_similarities = torch.cat(list(tqdm(map(id_learner, image_to_id(latent_to_image(all_latents_generator))))))
        video_frame_similarities = id_similarities(batch_function(video_frame_latents, opts.avg_id_batch_size))
        random_similarities = id_similarities(batch_function(wplus_latents, opts.avg_id_batch_size))
        
        video_frame_similarities = torch.cat(list(tqdm(video_frame_similarities)))
        random_similarities = torch.cat(list(tqdm(random_similarities)))

        print(f'Average similarity of actual frames: {video_frame_similarities.mean()}')
        print(f'Average similarity of random frames: {random_similarities.mean()}')
        
        (opts.output_path / 'logits').mkdir(exist_ok=True)
        save_vec(opts.output_path / 'logits' / 'id,pkl', torch.cat([video_frame_similarities, random_similarities]).cpu().numpy())
        
        # These are just wasting memory at this point
        del video_frame_similarities
        del random_similarities

        # We only train the attributes once. They should work for all videos after that
        if opts.attr_logits:
            random_latent_generator = random_latent_generator_creator(opts.num_attr_images, G.z_dim)
            
            # Load the estimator for the other attributes
            estimator = anycostgan.get_pretrained('attribute-predictor').to('cuda:0')
            estimator.eval().to(device)

            random_latents = torch.cat(list(map(mapping_fn, map(device_fn,batch_function(random_latent_generator, opts.avg_id_batch_size)))))
            
            save_vec(opts.output_path / 'attr_logits.pkl', random_latents.cpu().numpy())

            random_imgs = latent_to_image(batch_function(random_latents, opts.avg_id_batch_size))

            scores = map(estimator, random_imgs)

            scores = torch.cat(list(tqdm(scores))).view(-1, 40, 2)

            for i in range(scores.shape[1]):
                save_vec(opts.output_path / 'logits' / f'logits_{attr_list[i]}.pkl', scores[:, i, 1].unsqueeze(1).cpu().numpy())

        #generate_images(generator_path=opts.generator_path,

     #               truncation_psi=opts.truncation_psi,
 
def combine_generators(gen_1: Iterable[torch.Tensor], gen_2: Iterable[torch.Tensor]) -> Iterable[torch.Tensor]:
    """
    Combines two generators to produce a generator that exhausts both of them

    Arguments:
        gen_1: The first iterable to go over
        gen_2: The second iterable to go over

    Returns:
        A generator that produces the first iterable first, followed by the second
    """
    for item in gen_1:
        yield item
    for item in gen_2:
        yield item

def calc_id_score(test_id: torch.Tensor, ref_id: torch.Tensor) -> torch.Tensor:
    """
    Calculates how close two identities are using the cosine similarity
    between them

    Arguments:
        test_id: The id that will be compared. Should be a batch of unit vectors n x latent_dim
        ref_id: The id to compare to. Will be a 1 x latent_dim tensor

    Returns:
        An n x 1 tensor representing the cosine similarity between the two identities
    """
    return test_id @ ref_id.T

def arcface_transform(img: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of images (n x 3 x height x width) into the format needed by ArcFace. This resizes the images
    to 112 x 112 and changes the pixel values to be between [-1, 1]
    Arguments:
        img: The batch of images to convert. Should be n x 3 x height x width

    Returns:
        A batch of images ready for ArcFace
    """
    resize = Resize((112, 112))
    img = resize(img)
        
    img = ((img / 255.0) - .5) / .5 # I think stylegan outputs into this range anyways, so this function may be obsolete

    return img

def batch_function(images: Iterable[torch.Tensor], batch_size: int) -> Iterable[torch.Tensor]:
    image_batch = []
    image_count = 0
    for image in images:
        image_batch.append(image)
        image_count += 1

        if image_count == batch_size:
            yield torch.stack(image_batch, dim=0)
            image_count = 0
            image_batch = []

    # Handle the last batch
    if len(image_batch) != 0:
        yield torch.stack(image_batch, dim=0)

def generate_image_scores(images: Iterable[torch.Tensor], scorers: List[Callable[[torch.Tensor], torch.Tensor]]) -> List[torch.Tensor]:
    """
    Generates attribute scores for a group of latent vectors by running a series of
    scoring functions on each image.

    Arguments:
        images: An iterable of torch tensors representing images. Should be n x 3 x height x width and have pixels [0, 1]
        scorers: A list of scoring functions for an image

    Returns:
        A list of tensors representing the scores for each scoring function
    """
    # Initialize the tensors that hold the final result
    final_scores = []
    img_iter = iter(images)

    first_img_batch = next(img_iter)
    for scorer in scorers:
        final_scores.append(scorer(first_img_batch))
    
    # Then we can continue to read in image batches
    for img_batch in img_iter:
        for i, scorer in enumerate(scorers):
            torch.cat([final_scores[i], scorer(img_batch)], dim=0) # This should compose tensors along the batch dimension
    
    return final_scores
    # estimator for all attributes
    estimator = anycostgan.get_pretrained('attribute-predictor').to('cuda:0')
    estimator.eval()

    face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

    preds, ws = [], []
    saving_batch_id = 0
    for seed_idx, seed in tqdm(enumerate(range(n_images))):

        z = torch.from_numpy(np.random.RandomState(
            seed).randn(1, G.z_dim)).to(device)
        w = G.mapping(z, None, truncation_psi=truncation_psi)
        ws.append(w.detach().cpu().numpy())

        # if using unaligned generator, before generating the image and predicting attribute scores, align the image
        if generator_path == Path(model_paths["stylegan3_ffhq_unaligned"]):
            w[:, 0] = G.mapping.w_avg

        img = G.synthesis(w, noise_mode="const")
        img = face_pool(img)

        # get attribute scores for the generated image
        logits = estimator(img).view(-1, 40, 2)[0]

        attr_preds = logits.cpu().detach().numpy()[:, 1] # The last bit gives the percentage of it being positive according to the AnyCostGAN docs
        preds.append(attr_preds)

        # get predicted age
        #age = age_estimator.extract_ages(img).cpu().detach().numpy()[0]
        #print(age)
        #ages.append(age)

        # get predicted pose
        #pose = pose_estimator.extract_yaw(img).cpu().detach().numpy()[0]
        #print(pose)
        #poses.append(pose)

        # Get predicted identity
        #id = arcface_estimator(img).cpu().detach()

        #id_pred = torch.dot(random_identity.squeeze(), id.squeeze())
        #ids.append(id_pred)

    save_latents_and_scores(
        preds, ws, output_path)

    print(f'Generated images!')

def save_vec(path: Path, vec: np.ndarray):
    """
    Saves a numpy array as a pkl file at the given path

    Arguments:
        path: The path to save things to
        vec: The vector to save
    """
    with open(path, 'wb') as f:
        pickle.dump(vec, f)

if __name__ == '__main__':
    run()
