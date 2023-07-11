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

    # estimators for age and pose
    age_estimator = AgeEstimator()
    pose_estimator = PoseEstimator()

    # Create the Arcface network as well as a random unit vector for a new identity
    arcface_estimator = ArcFace().to(torch.device('cuda'))
    arcface_estimator.eval()  # Turn off batch norm training
    random_identity = torch.randn(1, 512)
    random_identity = random_identity / \
        torch.linalg.vector_norm(random_identity)

    face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

    preds, ages, poses, ws, ids = [], [], [], [], []
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
        attr_preds = torch.nn.functional.softmax(logits).cpu().detach().numpy()
        preds.append(attr_preds)

        # get predicted age
        age = age_estimator.extract_ages(img).cpu().detach().numpy()[0]
        ages.append(age)

        # get predicted pose
        pose = pose_estimator.extract_yaw(img).cpu().detach().numpy()[0]
        poses.append(pose)

        # Get predicted identity
        id = arcface_estimator(img).cpu().detach()

        id_pred = torch.dot(random_identity.squeeze(), id.squeeze())
        ids.append(id_pred)

        if seed_idx % save_interval == 0 and seed > 0:
            save_latents_and_scores(
                preds, ws, ages, poses, ids, saving_batch_id, output_path)
            saving_batch_id = saving_batch_id + 1
            preds, ages, poses, ws = [], [], [], []
            print(f'Generated {save_interval} images!')


def save_latents_and_scores(preds: List[np.ndarray], ws: List[np.ndarray], ages: List[float], poses: List[float], ids: List[float],
                            batch_id: int, output_path: Path):
    ws = np.vstack(ws)
    preds = np.array(preds)
    ages = np.vstack(ages)
    poses = np.vstack(poses)
    ids = np.vstack(ids)
    dir_path = output_path / f'id_{batch_id}'
    dir_path.mkdir(exist_ok=True, parents=True)
    np.save(dir_path / 'ws.npy', ws)
    np.save(dir_path / 'scores.npy', preds)
    np.save(dir_path / 'ages.npy', ages)
    np.save(dir_path / 'poses.npy', poses)
    np.save(dir_path / 'ids.npy', ids)


if __name__ == '__main__':
    run()
