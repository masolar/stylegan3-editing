"""
This script attempts to recreate the GradCtrl paper in the stylegan3 format.

This means we move back to the binary classifiers in the latent space rather than the image space classifiers
that don't seem to work well.
"""


import sys

sys.path.append(".")
sys.path.append("..")
import torchvision
from typing import Optional, List, Iterable, Tuple
from editing.interfacegan.helpers.anycostgan import attr_list
from PIL.Image import Image
from models.emonet import EmoNet
from models.arcface import ArcFaceModel
from utils.alignment_utils import align_face_optional, get_alignment_positions
from utils.cache_utils import cache_images
from utils.inference_utils import get_average_image, run_on_batch, load_encoder, IMAGE_TRANSFORMS, latents_to_image
from utils.common import tensor2im
from inversion.video.video_handler import VideoHandler
from inversion.video.video_editor import InterFaceGANVideoEditor, StyleCLIPVideoEditor, VideoEditor
from inversion.video.video_config import VideoConfig
from editing.interfacegan.helpers import anycostgan
from inversion.video.post_processing import postprocess_and_smooth_inversions
from prepare_data.landmarks_handler import LandmarksHandler, compute_landmarks_transforms
from inversion.video.generate_videos import generate_reconstruction_videos
from inversion.options.train_options import TrainOptions
from pathlib import Path
from typing import Dict, Any, List, Iterable, Optional

import numpy as np
import pyrallis
import torch
import torch.nn as nn
from tqdm import tqdm


from utils.images import crop_transform
from utils.video import video_reader, frame_reader
from utils.alignment_utils import align_face, get_alignment_transformation, get_alignment_positions
import dlib

def filter_none(data: Iterable[Optional[Any]]) -> Iterable[Any]:
    return filter(lambda entry: entry is not None, data)

def create_filtered_frame_generator(subset_path: Path, superset_path: Path) -> List[Path]:
    """
    A helper function for a really specific task. Given a path that contains frames of a video, and a target
    path also containing frames of a video, returns a generator that produces file paths from the second folder
    that exist within the first folder. This allows for the zippiung of two generators such that matching images
    are produced.

    It is assumed that the frames are named numerically and should be sorted before returning
    """
    # First, generate the set of filenames in the first path
    first_folder_names = {int(file.stem) for file in subset_path.iterdir()}

    second_folder_names = filter(lambda file: int(file.stem) in first_folder_names, 
                                 sorted(superset_path.iterdir(), 
                                        key=lambda file: int(file.stem)))

    return list(second_folder_names)

def create_sorted_filename_list(folder_path: Path) -> List[Path]:
    return sorted(folder_path.iterdir(), key=lambda filename: int(filename.stem))

@pyrallis.wrap()
def run_inference_on_video(video_opts: VideoConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_fn = lambda tensor: tensor.to(device)

    # prepare all the output paths
    video_opts.output_path.mkdir(exist_ok=True, parents=True)
    
    # Create output directories if they are not found
    if not video_opts.raw_frames_path:
        video_opts.raw_frames_path = video_opts.output_path / 'raw_frames'
    
    if not video_opts.aligned_frames_path:
        video_opts.aligned_frames_path = video_opts.output_path / 'aligned_frames'

    if not video_opts.cropped_frames_path:
        video_opts.cropped_frames_path = video_opts.output_path / 'cropped_frames'
    
    if len(list(video_opts.raw_frames_path.glob('*'))) == 0:
        video_opts.raw_frames_path.mkdir(exist_ok=True, parents=True)

        raw_frame_generator = tqdm(video_reader(video_opts.video_path), desc='Caching raw frames')
        
        # Give each frame a matching file name
        raw_frame_generator = map(lambda frame_tup: (frame_tup[1], video_opts.raw_frames_path / f'{frame_tup[0]}.jpg'), 
                                  enumerate(raw_frame_generator, 1))

        raw_frame_generator = cache_images(raw_frame_generator)
        
        # Force the generator to run to cache images
        for _ in raw_frame_generator:
            pass

    # TODO: Don't hard code this part
    predictor = dlib.shape_predictor(str('pretrained_models/shape_predictor_68_face_landmarks.dat'))
    detector = dlib.get_frontal_face_detector()
    
    if len(list(video_opts.aligned_frames_path.glob('*'))) == 0:
        video_opts.aligned_frames_path.mkdir(exist_ok=True, parents=True)

        aligned_frame_generator = tqdm(frame_reader(video_opts.raw_frames_path), desc='Caching aligned frames')
        
        # Because of the original code, we have to give it filenames rather than images, making my life oh so much harder
        frame_name_generator = map(lambda input: video_opts.raw_frames_path / f'{input[0]}.jpg', enumerate(aligned_frame_generator, 1))
       
        # Calculate the aligned frames from the raw ones
        aligned_frame_generator = map(lambda filepath: align_face_optional(str(filepath), detector, predictor), frame_name_generator)
        
        # Give each frame a matching file name
        aligned_frame_generator = map(lambda frame_tup: (frame_tup[1], video_opts.aligned_frames_path / f'{frame_tup[0]}.jpg'), 
                                  enumerate(aligned_frame_generator, 1))

        # Filter out frames that weren't processed correctly
        aligned_frame_generator = filter(lambda input: input[0] is not None, aligned_frame_generator)
        
        aligned_frame_generator = cache_images(aligned_frame_generator)

        # Force the generator to run
        list(aligned_frame_generator)
    
    if len(list(video_opts.cropped_frames_path.glob('*'))) == 0:
        video_opts.cropped_frames_path.mkdir(exist_ok=True, parents=True)

        cropped_frame_generator = tqdm(frame_reader(video_opts.raw_frames_path), desc='Caching cropped frames')

        # Once again, we have to generate frame names the hard way
        frame_name_generator = map(lambda input: video_opts.raw_frames_path / f'{input[0]}.jpg', enumerate(cropped_frame_generator, 1))
        
        # We have to use the first frame to bootstrap the rest of the process. This is lazy, so shouldn't be too much of a burden
        # TODO: Create a generator function instead of making this twice
        temp_cropped_frame_generator = frame_reader(video_opts.raw_frames_path)
        temp_frame_name_generator = map(lambda input: video_opts.raw_frames_path / f'{input[0]}.jpg', enumerate(temp_cropped_frame_generator, 1))
        
        """
        The old code assumes that the first frame can be used, but if the first frame fails, the whole code fails. This should
        keep trying frames until one is found that works for alignment.

        For instance, in a particular baby video, the baby has its head turned, and a face can therefore not be identified. This
        prevents that frame from being used for alignment.
        """
        while True:
            try:
                c, x, y = get_alignment_positions(str(next(iter(temp_frame_name_generator))), detector, predictor)
                alignment_transform, _ = get_alignment_transformation(c, x, y)
                alignment_transform = (alignment_transform + 0.5).flatten()
                break
            except Exception:
                continue

        cropped_frame_generator = map(lambda crop_filepath: crop_transform(crop_filepath, alignment_transform), frame_name_generator)

        # Give each frame a matching file name
        cropped_frame_generator = map(lambda frame_tup: (frame_tup[1], video_opts.cropped_frames_path / f'{frame_tup[0]}.jpg'), 
                                  enumerate(cropped_frame_generator, 1))

        # Filter out images that weren't cropped
        cropped_frame_generator = filter(lambda input: input[0] is not None, cropped_frame_generator)

        cropped_frame_generator = cache_images(cropped_frame_generator)

        # Force the generator to run
        list(cropped_frame_generator)

    # parse video
    # video_handler = VideoHandler(video_path=video_opts.video_path,
    #                              output_path=video_opts.output_path,
    #                              raw_frames_path=video_opts.raw_frames_path,
    #                              aligned_frames_path=video_opts.aligned_frames_path,
    #                              cropped_frames_path=video_opts.cropped_frames_path)
    # video_handler.parse_video()

    # aligned_paths, cropped_paths = video_handler.get_input_paths()

    # Here for testing with just a few vectors
    # aligned_paths = aligned_paths[:10]
    # cropped_paths = cropped_paths[:10]

    # input_images = video_handler.load_images(aligned_paths)
    # cropped_images = video_handler.load_images(cropped_paths)
    # if video_opts.max_images is not None:
    #     aligned_paths = aligned_paths[:video_opts.max_images]
    #     input_images = input_images[:video_opts.max_images]
    #     cropped_images = cropped_images[:video_opts.max_images]

    # load pretrained encoder
    net, opts = load_encoder(video_opts.checkpoint_path,
                             test_opts=video_opts, generator_path=video_opts.generator_path)
    
    # We can compute and save the landmarks here
    landmark_transforms = {}
    if not video_opts.landmarks_transforms_path:
        video_opts.landmarks_transforms_path = video_opts.output_path / 'landmarks_transforms.pt'

    if not video_opts.landmarks_transforms_path.exists():
        # We need an iterable of filename pairs
        aligned_paths_lst = create_sorted_filename_list(video_opts.aligned_frames_path)

        cropped_paths_lst = create_filtered_frame_generator(video_opts.aligned_frames_path, video_opts.cropped_frames_path)

        filename_pairs = tqdm(zip(aligned_paths_lst, cropped_paths_lst), total=len(aligned_paths_lst))

        landmark_transform_entries = map(lambda paths: compute_landmarks_transforms(paths, detector, predictor), filename_pairs)

        # Filter out frames that didn't have transforms
        landmark_transform_entries = filter_none(landmark_transform_entries)
        
        for key, val in landmark_transform_entries:
            landmark_transforms[key] = val

        torch.save(landmark_transforms, video_opts.landmarks_transforms_path)
    
    landmark_transforms = torch.load(video_opts.landmarks_transforms_path)

    # loads/computes landmarks transforms for the video frames
    landmarks_handler = LandmarksHandler(output_path=video_opts.output_path,
                                         landmarks_transforms_path=video_opts.landmarks_transforms_path)
    video_opts.landmarks_transforms_path = landmarks_handler.landmarks_transforms_path
    # landmarks_transforms = landmarks_handler.get_landmarks_transforms(input_paths=aligned_paths,
    #                                                                   cropped_frames_path=video_handler.cropped_frames_path,
    #                                                                   aligned_frames_path=video_handler.aligned_frames_path)

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

    # landmarks_transforms = np.array(list(
    #     map(lambda x: x.cpu(), list(results["landmarks_transforms"]))))

    landmarks_transforms = torch.stack(results['landmarks_transforms'])
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

    cropped_images_tensor = torch.stack(
        list(map(torchvision.transforms.ToTensor(), cropped_images))).to(device)
    latents_tensor = torch.Tensor(result_latents).to(device)

    print(f'Orig: {latents_tensor}')
    edited_latents = edit(
        orig_images=cropped_images_tensor,
        latents=latents_tensor,
        id_net=arcface_net,
        emo_net=emo_net,
        anycost=estimator,
        anycost_features=['Male', 'Young'],
        generator=net.decoder,
        user_transforms=landmarks_transforms
    )
    print(f'Edited: {edited_latents}')
    print(f'Edited shape: {edited_latents.shape}')
    editor = InterFaceGANVideoEditor(
        generator=net.decoder, opts=video_opts)

    edited_images_smoothed = editor.postprocess_and_smooth_edits(
        results, edited_latents.detach().cpu().numpy(), video_opts)
    editor.generate_edited_video(input_images=cropped_images,
                                 result_images_smoothed=result_images_smoothed,
                                 edited_images_smoothed=edited_images_smoothed,
                                 video_handler=video_handler,
                                 save_name=f"edited_video_speedup")

    # if opts.interfacegan_directions is not None:
    #     editor = InterFaceGANVideoEditor(
    #         generator=net.decoder, opts=video_opts)
    #     for interfacegan_edit in video_opts.interfacegan_edits:
    #         edit_images_start, edit_images_end, edit_latents_start, edit_latents_end = editor.edit(
    #             edit_direction=interfacegan_edit.direction,
    #             start=interfacegan_edit.start,
    #             end=interfacegan_edit.end,
    #             result_latents=result_latents,
    #             landmarks_transforms=landmarks_transforms
    #         )
    #         edited_images_start_smoothed = editor.postprocess_and_smooth_edits(
    #             results, edit_latents_start, video_opts)
    #         edited_images_end_smoothed = editor.postprocess_and_smooth_edits(
    #             results, edit_latents_end, video_opts)
    #         editor.generate_edited_video(input_images=cropped_images,
    #                                      result_images_smoothed=result_images_smoothed,
    #                                      edited_images_smoothed=edited_images_start_smoothed,
    #                                      video_handler=video_handler,
    #                                      save_name=f"edited_video_{interfacegan_edit.direction}_start")
    #         editor.generate_edited_video(input_images=cropped_images,
    #                                      result_images_smoothed=result_images_smoothed,
    #                                      edited_images_smoothed=edited_images_end_smoothed,
    #                                      video_handler=video_handler,
    #                                      save_name=f"edited_video_{interfacegan_edit.direction}_end")
    #
    # if opts.styleclip_directions is not None:
    #     editor = StyleCLIPVideoEditor(generator=net.decoder, opts=video_opts)
    #     for styleclip_edit in video_opts.styleclip_edits:
    #         edited_images, edited_latents = editor.edit(edit_direction=styleclip_edit.target_text,
    #                                                     alpha=styleclip_edit.alpha,
    #                                                     beta=styleclip_edit.beta,
    #                                                     result_latents=result_latents,
    #                                                     landmarks_transforms=landmarks_transforms)
    #         edited_images_smoothed = editor.postprocess_and_smooth_edits(
    #             results, edited_latents, video_opts)
    #         editor.generate_edited_video(input_images=cropped_images,
    #                                      result_images_smoothed=result_images_smoothed,
    #                                      edited_images_smoothed=edited_images_smoothed,
    #                                      video_handler=video_handler,
    #                                      save_name=styleclip_edit.save_name)
    #


def edit(orig_images: torch.Tensor,
         latents: torch.Tensor,
         id_net: nn.Module,
         emo_net: nn.Module,
         anycost: nn.Module,
         anycost_features: List[str],
         generator: nn.Module,
         num_exclude_dims=4000,
         factor: float = 4/9,
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

    def arcface_transform(image): return (image - .5) / .5

    def emonet_transform(image): return torchvision.transforms.Resize(
        (256, 256))(image)

    batch_size = 3

    # Create a batch data loader
    loader = BatchLoader((orig_images, latents, user_transforms), batch_size)

    # for image, latent, transform in tqdm(zip(orig_images, latents, user_transforms), total=len(orig_images)):
    for (image, latent, transform) in tqdm(loader):
        batch_size = latent.shape[0]
        # image = image.unsqueeze(0)
        latent = latent.flatten().unsqueeze(0)
        # transform = transform.unsqueeze(0)

        # Compute the identity and emotion of the given image
        with torch.no_grad():
            # Get the ground truth values from the original images
            id_vec = id_net(arcface_transform(image)).squeeze()
            # emo_out = emo_net(emonet_transform(image))
            # actual_val = emo_out['valence']
            # actual_arousal = emo_out['arousal']

        for i in range(101):
            # Compute the identity and emotion for the reconstruction image
            latent.requires_grad_(True)

            # Generate an image from the latent vector given
            gen_image = _latents_to_image(generator, latent, transform)
            recon_id_vec = id_net(arcface_transform(gen_image)).squeeze()

            loss = torch.sum(-torch.diag(recon_id_vec @ id_vec.T) + 1)

            id_grad = torch.autograd.grad(outputs=loss,
                                          inputs=latent)[0]
            id_dims_to_exclude = id_grad.detach().squeeze()
            id_dims_to_exclude = torch.abs(id_dims_to_exclude)
            id_dims_to_exclude = torch.argsort(
                id_dims_to_exclude)[-num_exclude_dims:]

            gen_image = _latents_to_image(generator, latent, transform)
            recon_emo_out = emo_net(emonet_transform(gen_image))
            recon_val = recon_emo_out['valence']
            recon_arousal = recon_emo_out['arousal']

            # loss = torch.sqrt((actual_val - recon_val) ** 2 + (actual_arousal - recon_arousal) ** 2)

            # emo_grad = torch.autograd.grad(outputs=loss,
            # inputs=latent,
            # retain_graph=True)[0]
            # emo_dims_to_exclude = emo_grad.detach().squeeze()
            # emo_dims_to_exclude = torch.abs(emo_dims_to_exclude)
            # emo_dims_to_exclude = torch.argsort(emo_dims_to_exclude)[-num_exclude_dims:]

            # Next, compute the gradient for the random networks so we can mask things
            dims_to_exclude = []
            with torch.no_grad():
                img_vec = torch.nn.functional.softmax(anycost(image).view(-1, 40, 2)[0], dim=0)[
                    [attr_list.index(feature) for feature in anycost_features]]

            gen_image = latents_to_image(generator, latent, transform)
            recon_img_vec = torch.nn.functional.softmax(anycost(
                gen_image).view(-1, 40, 2)[0], dim=0)[[attr_list.index(feature) for feature in anycost_features]]

            loss = torch.nn.MSELoss()(img_vec, recon_img_vec)

            dim_c = torch.autograd.grad(outputs=loss,
                                        inputs=latent)[0]

            dim_c = dim_c.detach().squeeze()
            dim_c = torch.abs(dim_c)
            excluded = torch.argsort(dim_c)[-num_exclude_dims:]
            dims_to_exclude = excluded.flatten().unique()

            # for feature in anycost_features:
            #     with torch.no_grad():
            #         # Get the result of the classifier on the original image for comparison
            #         img_vec = torch.nn.functional.softmax(anycost(image).view(-1, 40, 2)[0])[attr_list.index(feature)]
            #
            #     recon_img_vec = torch.nn.functional.softmax(anycost(gen_image).view(-1, 40, 2)[0])[attr_list.index(feature)]
            #
            #     loss = torch.nn.MSELoss()(img_vec, recon_img_vec)
            #
            #     dim_c = torch.autograd.grad(outputs=loss,
            #                                 inputs=latent,
            #                                 retain_graph=True)[0]
            #
            #     dim_c = dim_c.detach().squeeze()
            #     dim_c = torch.abs(dim_c)
            #     excluded = torch.argsort(dim_c)[-num_exclude_dims:]
            #     dims_to_exclude.append(excluded)

            if len(dims_to_exclude) > 0:
                # dims_to_exclude = torch.unique(
                #    torch.cat(dims_to_exclude))
                id_mask = torch.ones(1, batch_size * 512 * 16).cuda()
                # emo_mask = torch.ones(1, 512 * 16).cuda()
                id_mask[:, dims_to_exclude] = 0
                # id_mask[:, emo_dims_to_exclude] = 0
                # emo_mask[:, dims_to_exclude] = 0
                # emo_mask[:, id_dims_to_exclude] = 0

                id_grad *= id_mask
                # emo_grad *= emo_mask

            id_grad /= torch.norm(id_grad)
            # emo_grad /= torch.norm(emo_grad)

            with torch.no_grad():
                latent = latent - (id_grad * factor)  # - (emo_grad * factor)

        edited_latents.append(latent.detach().reshape(-1, 16, 512))

    return torch.cat(edited_latents)


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



if __name__ == '__main__':
    run_inference_on_video()
