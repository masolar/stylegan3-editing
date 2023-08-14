"""
A script to preprocess a video and save the results as frames under a folder.

This removes the need to have caching code in the experiments
"""
from typing import Tuple, Iterable
import pyrallis
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
from utils.video import video_reader
from utils.cache_utils import cache_images
import torch
import dlib
from utils.alignment_utils import align_face_optional, get_alignment_positions_img, get_alignment_transformation
from utils.images import crop_transform
from torchvision.io import read_image
from prepare_data.landmarks_handler import get_inverse_transform
from itertools import takewhile


@dataclass
class PreprocessOpts:
    # The path for outputs to be placed in
    output_path: Path
    # The path to the video to preprocess
    video_path: Path


def get_video_frame_generator(video_path: Path) -> Iterable[Tuple[torch.Tensor, Path]]:
    """
    Creates a generator that can read in frames of a video and give each frame a filename

    Arguments:
        video_path: The path to the video to read

    Returns:
        An iterable containing pairs of images along with a filename
    """
    raw_frame_generator = video_reader(video_path)

    # Give each frame a matching file name
    raw_frame_generator = map(lambda frame_tup: (frame_tup[1], Path(f'{frame_tup[0]}.jpg')),
                              enumerate(raw_frame_generator, 1))

    return raw_frame_generator


@pyrallis.wrap()
def main(opts: PreprocessOpts):
    # prepare all the output paths
    opts.output_path.mkdir(exist_ok=True, parents=True)

    # Load the facial detectors in for the alignment and cropping portions
    # TODO: Don't hard code this part
    predictor = dlib.shape_predictor(
        str('pretrained_models/shape_predictor_68_face_landmarks.dat'))
    detector = dlib.get_frontal_face_detector()

    # Create output directories if they are not found
    raw_frames_path = opts.output_path / 'raw_frames'
    aligned_frames_path = opts.output_path / 'aligned_frames'
    cropped_frames_path = opts.output_path / 'cropped_frames'

    raw_frames_path.mkdir(exist_ok=True, parents=True)
    aligned_frames_path.mkdir(exist_ok=True, parents=True)
    cropped_frames_path.mkdir(exist_ok=True, parents=True)

    raw_frame_generator = get_video_frame_generator(opts.video_path)

    # Make sure each image has a full path instead of just a frame name
    # Then cache the images that are read in here
    raw_frame_generator = map(lambda frame_tup: (
        frame_tup[0], raw_frames_path / frame_tup[1]), raw_frame_generator)

    raw_frame_generator = cache_images(raw_frame_generator)

    # Next, we want to create a generator that produces aligned images
    aligned_frame_generator = tqdm(map(lambda entry: (align_face_optional(
        entry[0], detector, predictor), aligned_frames_path / entry[1].name), raw_frame_generator), desc='Caching aligned images')

    # Remove any frames that weren't correctly aligned
    aligned_frame_generator = filter(
        lambda entry: entry[0] is not None, aligned_frame_generator)

    # Make sure we cache aligned frames
    aligned_frame_generator = cache_images(
        aligned_frame_generator)

    # list(aligned_frame_generator)

    # Now we generate cropped frames
    cropped_frame_generator = get_video_frame_generator(opts.video_path)

    """
    The old code assumes that the first frame can be used, but if the first frame fails, the whole code fails. This should
    keep trying frames until one is found that works for alignment.

    For instance, in a particular baby video, the baby has its head turned, and a face can therefore not be identified. This
    prevents that frame from being used for alignment.
    """
    temp_frame_generator = get_video_frame_generator(opts.video_path)
    print('Finding a frame that can be used for cropping')
    for image, _ in tqdm(temp_frame_generator):
        try:
            # Convert image to numpy
            image = (image * 255).to(torch.uint8).permute(1, 2, 0).numpy()
            c, x, y = get_alignment_positions_img(
                image, detector, predictor)
            alignment_transform, _ = get_alignment_transformation(c, x, y)
            alignment_transform = (alignment_transform + 0.5).flatten()
            break
        except Exception as e:
            print(e)
            continue

    cropped_frame_generator = tqdm(map(lambda entry: (crop_transform(
        entry[0], alignment_transform), cropped_frames_path / entry[1].name), cropped_frame_generator))

    # Filter out images that weren't cropped
    cropped_frame_generator = filter(
        lambda input: input[0] is not None, cropped_frame_generator)

    cropped_frame_generator = cache_images(cropped_frame_generator)

    # Force the generators to run
    # list(cropped_frame_generator)

    # We should also calculate the inverse transforms between images while we're here
    # First, create a set of cropped image paths and aligned image paths
    aligned_image_paths = set(
        map(lambda path: path.name, aligned_frames_path.iterdir()))
    cropped_image_paths = set(
        map(lambda path: path.name, cropped_frames_path.iterdir()))

    # Create a generator the gives matching images between the two
    image_path_pairs = ((aligned_frames_path / image_name, cropped_frames_path / image_name)
                        if image_name in aligned_image_paths else None for image_name in cropped_image_paths)
    image_path_pairs = filter(
        lambda entry: entry is not None, image_path_pairs)

    # Find inverse transforms between the pairs of images
    image_generator = map(lambda entry: (entry[0], read_image(
        str(entry[0])) / 255.0, read_image(str(entry[1])) / 255.0), image_path_pairs)

    # This generator should pair up a filename with an inverse transform
    inverse_generator = map(lambda entry: (entry[0].name, get_inverse_transform(
        entry[1], entry[2], detector, predictor)), image_generator)

    # Remove frames that didn't have transforms found
    inverse_generator = filter(
        lambda entry: entry[1] is not None, tqdm(inverse_generator))

    # inverse_generator = takewhile(
    #    lambda entry: entry[0] < 10, enumerate(inverse_generator))

    # The dictionary should hold a filename along with the inverse transform
    inverse_dict = {filename: transform for filename,
                    transform in inverse_generator}

    # Then we can save the transforms for later
    torch.save(inverse_dict, opts.output_path / 'inverse_transforms.pth')


if __name__ == '__main__':
    main()
