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
from typing import List
from pathlib import Path
import pickle
from configs.paths_config import model_paths
from editing.interfacegan.helpers import anycostgan
from editing.interfacegan.helpers.pose_estimator import PoseEstimator
from editing.interfacegan.helpers.age_estimator import AgeEstimator
from editing.interfacegan.helpers.id_estimator import ArcFace
from models.stylegan3.networks_stylegan3 import SynthesisNetwork
from editing.interfacegan.helpers.anycostgan import attr_list
from models.stylegan3.model import SG3Generator
from tqdm import tqdm


@dataclass
class EditConfig:
    # Path to StyleGAN3 generator
    generator_path: Path = Path(model_paths["stylegan3_ffhq"])
    # Number of latents to sample
    n_images: int = 500000
    # Truncation psi for sampling
    truncation_psi: float = 0.7
    # Where to save the `npy` files with latents and scores to
    output_path: Path = Path("./latents")
    # How often to save sample latents/scores to `npy` files
    save_interval: int = 10000


@pyrallis.wrap()
def run(opts: EditConfig):
    generate_images(generator_path=opts.generator_path,
                    n_images=opts.n_images,
                    truncation_psi=opts.truncation_psi,
                    output_path=opts.output_path,
                    save_interval=opts.save_interval)


# def generate_images(generator: SynthesisNetwork, n_images: int, save_interval: int)


def generate_images(generator_path: Path, n_images: int, truncation_psi: float, output_path: Path, save_interval: int):

    print('Loading generator from "%s"...' % generator_path)
    device = torch.device('cuda')
    G = SG3Generator(generator_path).decoder

    # with open(generator_path, "rb") as f:
    #     G = pickle.load(f)['G_ema'].cuda()
    #
    output_path.mkdir(exist_ok=True, parents=True)

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
        print(estimator(img).shape)
        print(logits)
        attr_preds = torch.nn.functional.softmax(logits).cpu().detach().numpy()[:, 1] # The last bit gives the percentage of it being positive according to the AnyCostGAN docs
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


def save_latents_and_scores(preds: List[np.ndarray], ws: List[np.ndarray],
                            output_path: Path):
    ws = np.vstack(ws)
    preds = np.array(preds)
    dir_path = output_path

    # First, save the latent vectors created
    np.save(dir_path / 'latents.npy', ws)

    dir_path = dir_path / 'logits'
    dir_path.mkdir(exist_ok=True, parents=True)
    
    # Then save each of the logits separately
    for attr_name in attr_list:
        sub_vector = np.expand_dims(preds[:, attr_list.index(attr_name)], -1)
        np.save(dir_path / f'logits_{attr_name}.npy', sub_vector)


if __name__ == '__main__':
    run()
