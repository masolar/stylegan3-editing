"""
A new script to compute grad control on both the emotion and identity portions of the edit vector.

This should hopefully create a new identity with the same emotions.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pyrallis
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")

from inversion.options.train_options import TrainOptions
from inversion.video.generate_videos import generate_reconstruction_videos
from prepare_data.landmarks_handler import LandmarksHandler
from inversion.video.post_processing import postprocess_and_smooth_inversions
from editing.interfacegan.helpers import anycostgan
from inversion.video.video_config import VideoConfig
from inversion.video.video_editor import InterFaceGANVideoEditor, StyleCLIPVideoEditor
from inversion.video.video_handler import VideoHandler
from utils.common import tensor2im
from utils.inference_utils import get_average_image, run_on_batch, load_encoder, IMAGE_TRANSFORMS
from models.arcface import ArcFaceModel
from models.emonet import EmoNet
from PIL.Image import Image
from editing.interfacegan.helpers.anycostgan import attr_list
from typing import Optional, List
import torchvision

@pyrallis.wrap()
def run_inference_on_video(video_opts: VideoConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # prepare all the output paths
    video_opts.output_path.mkdir(exist_ok=True, parents=True)

    # parse video
    video_handler = VideoHandler(video_path=video_opts.video_path,
                                 output_path=video_opts.output_path,
                                 raw_frames_path=video_opts.raw_frames_path,
                                 aligned_frames_path=video_opts.aligned_frames_path,
                                 cropped_frames_path=video_opts.cropped_frames_path)
    video_handler.parse_video()

    aligned_paths, cropped_paths = video_handler.get_input_paths()
    
    # Here for testing with just a few vectors
    aligned_paths = aligned_paths[:10]
    cropped_paths = cropped_paths[:10]

    input_images = video_handler.load_images(aligned_paths)
    cropped_images = video_handler.load_images(cropped_paths)
    if video_opts.max_images is not None:
        aligned_paths = aligned_paths[:video_opts.max_images]
        input_images = input_images[:video_opts.max_images]
        cropped_images = cropped_images[:video_opts.max_images]

    # load pretrained encoder
    net, opts = load_encoder(video_opts.checkpoint_path,
                             test_opts=video_opts, generator_path=video_opts.generator_path)

    # loads/computes landmarks transforms for the video frames
    landmarks_handler = LandmarksHandler(output_path=video_opts.output_path,
                                         landmarks_transforms_path=video_opts.landmarks_transforms_path)
    video_opts.landmarks_transforms_path = landmarks_handler.landmarks_transforms_path
    landmarks_transforms = landmarks_handler.get_landmarks_transforms(input_paths=aligned_paths,
                                                                      cropped_frames_path=video_handler.cropped_frames_path,
                                                                      aligned_frames_path=video_handler.aligned_frames_path)
    # run inference
    results = run_inference(input_paths=aligned_paths,
                            input_images=input_images,
                            landmarks_transforms=landmarks_transforms,
                            net=net,
                            opts=opts)

    # save inverted latents (can be used for editing, pti, etc)
    results_latents_path = opts.output_path / "latents.npy"
    np.save(results_latents_path, np.array(results["result_latents"]))
    
    result_images = [np.array(tensor2im(im))
                     for im in results["result_images"]]

    result_latents = np.array(list(results["result_latents"].values()))

    landmarks_transforms = np.array(list(
        map(lambda x: x.cpu(), list(results["landmarks_transforms"]))))
    
    result_images_smoothed = postprocess_and_smooth_inversions(
        results, net, video_opts)

    # get video reconstruction
    generate_reconstruction_videos(input_images=cropped_images,
                                   result_images=result_images,
                                   result_images_smoothed=result_images_smoothed,
                                   video_handler=video_handler,
                                   opts=video_opts)
    
    # Load the arcface network
    arcface_net = ArcFaceModel('pretrained_models/arcface.pth', 'r50')
    arcface_net.eval().to(device)
    
    # Load EmoNet
    emo_net = EmoNet()
    emo_net.load_state_dict(torch.load('pretrained_models/emonet_8.pth'))
    emo_net.eval().to(device)

    # Load the anycost network
    estimator = anycostgan.get_pretrained('attribute-predictor')
    estimator.eval().to(device)
    
    cropped_images_tensor = torch.stack(list(map(torchvision.transforms.ToTensor(), cropped_images))).to(device)
    latents_tensor = torch.Tensor(result_latents).to(device)
    
    landmarks_transforms_tensor = torch.stack([lt for lt in landmarks_transforms]).to(device)
   
    edited_latents = edit(
        orig_images=cropped_images_tensor,
        latents=latents_tensor,
        id_net=arcface_net,
        emo_net=emo_net,
        anycost=estimator,
        anycost_features=['Male', 'Age'],
        generator=net.decoder,
        user_transforms=landmarks_transforms_tensor
    )

    if opts.interfacegan_directions is not None:
        editor = InterFaceGANVideoEditor(
            generator=net.decoder, opts=video_opts)
        for interfacegan_edit in video_opts.interfacegan_edits:
            edit_images_start, edit_images_end, edit_latents_start, edit_latents_end = editor.edit(
                edit_direction=interfacegan_edit.direction,
                start=interfacegan_edit.start,
                end=interfacegan_edit.end,
                result_latents=result_latents,
                landmarks_transforms=landmarks_transforms
            )
            edited_images_start_smoothed = editor.postprocess_and_smooth_edits(
                results, edit_latents_start, video_opts)
            edited_images_end_smoothed = editor.postprocess_and_smooth_edits(
                results, edit_latents_end, video_opts)
            editor.generate_edited_video(input_images=cropped_images,
                                         result_images_smoothed=result_images_smoothed,
                                         edited_images_smoothed=edited_images_start_smoothed,
                                         video_handler=video_handler,
                                         save_name=f"edited_video_{interfacegan_edit.direction}_start")
            editor.generate_edited_video(input_images=cropped_images,
                                         result_images_smoothed=result_images_smoothed,
                                         edited_images_smoothed=edited_images_end_smoothed,
                                         video_handler=video_handler,
                                         save_name=f"edited_video_{interfacegan_edit.direction}_end")

    if opts.styleclip_directions is not None:
        editor = StyleCLIPVideoEditor(generator=net.decoder, opts=video_opts)
        for styleclip_edit in video_opts.styleclip_edits:
            edited_images, edited_latents = editor.edit(edit_direction=styleclip_edit.target_text,
                                                        alpha=styleclip_edit.alpha,
                                                        beta=styleclip_edit.beta,
                                                        result_latents=result_latents,
                                                        landmarks_transforms=landmarks_transforms)
            edited_images_smoothed = editor.postprocess_and_smooth_edits(
                results, edited_latents, video_opts)
            editor.generate_edited_video(input_images=cropped_images,
                                         result_images_smoothed=result_images_smoothed,
                                         edited_images_smoothed=edited_images_smoothed,
                                         video_handler=video_handler,
                                         save_name=styleclip_edit.save_name)


def edit(orig_images: torch.Tensor,
         latents: torch.Tensor,
         id_net: nn.Module,
         emo_net: nn.Module,
         anycost: nn.Module,
         anycost_features: List[str],
         generator: nn.Module,
         num_exclude_dims = 4000,
         factor: int = 1,
         user_transforms: torch.Tensor = None) -> torch.Tensor:
    '''
    Computes edited latent vectors for the given arguments

    Arguments:
        orig_images: The original images that will be edited. This should be a tensor of shape
                     n x 3 x h x w, with values [0, 1]
        latents: The latent vectors for each image
        id_net: The network for computing the id vector
        emo_net: The network for computing the valence/arousal
        anycost: The anycost network for computing other features
        anycost_features: The features to disentangle
        generator: The generator to go from latent to image
        num_exclude_dims: The number of dimensions to mask for each vector
        factor: The step size when moving through the latent space
        user_transforms: The transforms to apply to the images being generated by the latent vectors
        apply_user_transformations: Whether or not to apply the given transformations

    Returns:
        The edited latent vectors for each image
    '''
    edited_latents = []
    id_net.requires_grad_(True)
    emo_net.requires_grad_(True)
    generator.requires_grad_(True)
    anycost.requires_grad_(True)

    emonet_transform = lambda image: torchvision.transforms.Resize((256, 256))(image)
   
    for image, latent, transform in zip(orig_images, latents, user_transforms):
        image = image.unsqueeze(0)
        latent = latent.unsqueeze(0)
       
        transform = transform.unsqueeze(0)

        # Compute the identity and emotion of the given image
        with torch.no_grad():
            id_vec = id_net((_latents_to_image(generator, latent, transform) - .5 ) / .5).squeeze()
            emo_out = emo_net(emonet_transform(image))
            actual_val = emo_out['valence']
            actual_arousal = emo_out['arousal']
        
        for _ in range(101):
            # Compute the identity and emotion for the reconstruction image
            latent.requires_grad_(True)
            recon_id_vec = _latents_to_image(generator, latent, transform)
            loss = recon_id_vec.sum()
            id_grad = torch.autograd.grad(outputs=loss,inputs=latent)
            print(id_grad)
            # recon_id_vec = id_net((_latents_to_image(generator, latent, transform) - .5) / .5).squeeze()
            #
            # loss = torch.dot(recon_id_vec, id_vec)
            #
            # id_grad = torch.autograd.grad(outputs=loss,
            #                               inputs=latent)[0]
            id_dims_to_exclude = id_grad.detach().squeeze().cpu().numpy()
            id_dims_to_exclude = np.argsort(id_dims_to_exclude)[-num_exclude_dims:]

            recon_emo_out = emo_net(emonet_transform(_latents_to_image(generator, latent, transform)))
            recon_val = recon_emo_out['valence']
            recon_arousal = recon_emo_out['arousal']

            loss = torch.sqrt((actual_val - recon_val) ** 2 + (actual_arousal - recon_arousal) ** 2)

            emo_grad = torch.autograd.grad(outputs=loss,
                                           inputs=latent)[0]
            emo_dims_to_exclude = emo_grad.detach().squeeze().cpu().numpy()
            emo_dims_to_exclude = np.argsort(emo_dims_to_exclude)[-num_exclude_dims:]

            # Next, compute the gradient for the random networks so we can mask things
            dims_to_exclude = []
            for feature in anycost_features:
                with torch.no_grad():
                    # Get the result of the classifier on the original image for comparison
                    img_vec = torch.nn.functional.softmax(anycost(image).view(-1, 40, 2)[0])[attr_list.index(feature)]

                recon_img_vec = torch.nn.functional.softmax(anycost(_latents_to_image(generator, latent, transform)).view(-1, 40, 2)[0])[attr_list.index(feature)]

                loss = torch.nn.MSELoss()(img_vec, recon_img_vec)

                dim_c = torch.autograd.grad(outputs=loss,
                                            inputs=latent)[0]

                dim_c = dim_c.detach().squeeze().cpu().numpy()
                excluded = np.argsort(dim_c)[-num_exclude_dims:]
                dims_to_exclude.append(excluded)
            
            if len(dims_to_exclude) > 0:
                dims_to_exclude = np.unique(
                    np.concatenate(dims_to_exclude))
                id_mask = torch.ones(1, 512 * 16, 1).cuda()
                emo_mask = torch.ones(1, 512 * 16, 1).cuda()
                id_mask[:, dims_to_exclude, :] = 0
                id_mask[:, emo_dims_to_exclude] = 0
                emo_mask[:, dims_to_exclude, :] = 0
                emo_mask[:, id_dims_to_exclude, :] = 0

                id_grad *= id_mask
                emo_grad *= emo_mask
            id_grad /= torch.norm(id_grad)
            emo_grad /= torch.norm(emo_grad)

            with torch.no_grad():
                latent = latent - (id_grad * factor) + (emo_grad * factor)
        edited_latents.append(latent)

    return torch.Tensor(edited_latents)
            
def run_inference(input_paths: List[Path], input_images: List, landmarks_transforms: Dict[str, Any], net,
                  opts: TrainOptions):
    results = {"source_images": [], "result_images": [],
               "result_latents": {}, "landmarks_transforms": []}
    with torch.no_grad():
        avg_image = get_average_image(net)
    # run inference one frame at a time (technically can be run in batches, but done for simplicity)
    for input_image, input_path in tqdm(zip(input_images, input_paths)):
        results["source_images"].append(input_image)
        image_name = input_path.name
        if landmarks_transforms is not None:
            if image_name not in landmarks_transforms:
                continue
            image_landmarks_transform = torch.from_numpy(
                landmarks_transforms[image_name][-1]).cuda()
        else:
            image_landmarks_transform = None
        with torch.no_grad():
            transformed_image = IMAGE_TRANSFORMS(input_image)
            result_batch, latents = run_on_batch(inputs=transformed_image.unsqueeze(0).cuda(),
                                                 net=net,
                                                 opts=opts,
                                                 avg_image=avg_image,
                                                 landmarks_transform=image_landmarks_transform)
            # we'll save the last inversion and latent code
            results["result_images"].append(result_batch[0][-1])
            results["result_latents"][image_name] = latents[0][-1]
            results["landmarks_transforms"].append(image_landmarks_transform)
    return results

def _latents_to_image(generator: nn.Module,
                      latent: torch.Tensor, 
                      user_transforms: torch.Tensor):
    """
    Converts a latent vector to actual images, using the transforms supplied to generate
    translated and rotated images
    """
    #with torch.no_grad():
    generator.synthesis.input.transform = user_transforms.float()
    # generate the images
    image = generator.synthesis(latent, noise_mode='const')
    # image = tensor2im(image.squeeze())
    return image

if __name__ == '__main__':
    run_inference_on_video()
