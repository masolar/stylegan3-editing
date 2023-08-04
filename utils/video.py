from os import PathLike
import torch
from pathlib import Path
from torchvision.io import read_image, read_video, write_video
from torchvision.transforms import ToTensor
from typing import Tuple, Union, Iterator, Iterable
from torch.utils.data import Dataset
import cv2


def generate_mp4(filepath: Union[str, PathLike, bytes], frames: torch.Tensor, fps: float = 24):
    """
    Generates an mp4 from a tensor.

    Arguments:
        filepath: The filepath to save the video into
        frames: The tensor representing the frames of the video. Should be T x C x H x W with range [0, 1]
        fps: The fps of the resulting video
    """
    # The torch api takes in a tensor of shape T x H x W x C as a uint8, so we'll convert it
    frames = frames.permute(0, 2, 3, 1)

    frames *= 255
    frames = frames.to(torch.uint8)

    write_video(str(filepath), frames, fps=fps)


class video_reader(Iterator[torch.Tensor]):
    '''
    Creates an iterable over a video file. Video frames are in the range [0, 1] with shape N x C x H x W
    '''

    def __init__(self, filepath: Union[str, PathLike, bytes]):
        self.filepath = filepath
        self.reader = cv2.VideoCapture(str(filepath))

        # Get the length of the video in case it's needed
        self.length = int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT))

        # self.frames, _, _ = read_video(
        #     str(filepath), output_format='TCHW', pts_unit='sec')
        #
        # self.frames = self.frames / 255.0

    def __len__(self):
        return self.length

    def __iter__(self):
        if not self.reader.isOpened():
            self.reader = cv2.VideoCapture(str(self.filepath))

        return self

    def __next__(self):
        while self.reader.isOpened():
            ret, frame = self.reader.read()
            if ret:
                # Convert to a tensor with range [0, 1]
                return ToTensor()(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            else:
                self.reader.release()
                raise StopIteration


class frame_reader(Iterable[torch.Tensor]):
    '''
    Creates an iterable over frames from a folder. Frames are assumed to be named using numerical identifiers, such as 1.jpg, 123.jpg, etc.
    Frames are returned as Tensors in the range [0, 1] with dimensions C x H x W
    '''

    def __init__(self, folder_name: Union[str, PathLike, bytes]):
        # Load and sort the filenames from this folder
        self.folder_name = Path(str(folder_name))

        # We use a map to save only the filename. This saves space rather than having to save the full path for every file
        self.file_names = list(map(lambda filename: filename.name, sorted(
            Path(str(folder_name)).glob('*'), key=lambda file_name: int(file_name.stem))))

    def __iter__(self):
        return map(lambda image_file: read_image(str(image_file)) / 255.0,
                   map(lambda filename: self.folder_name / filename,
                       self.file_names))

    def __len__(self):
        return len(self.file_names)


class frame_path_reader(Iterable[torch.Tensor]):
    """
    A class representing an iterable of frames from an iterable of filenames. Useful when you have a list of
    filenames representing frames and you want to generate the corresponding images.
    """

    def __init__(self, file_paths: Iterable[Union[str, PathLike, bytes]]):
        self.file_paths = file_paths

    def __iter__(self):
        return map(lambda file_path: read_image(str(file_path)) / 255.0, self.file_paths)


def read_mp4(filepath: Union[str, PathLike, bytes]) -> Tuple[torch.Tensor, float, int]:
    """
    Reads in a video as a tensor.

    Arguments:
        filepath: The path to the video

    Returns:
        A tensor of shape T x C x H x W with range [0, 1] representing the frames of the video,
        the fps of the video, and the number of video frames which were read
    """
    frames, _, metadata = read_video(str(filepath), output_format='TCHW')

    video_fps = metadata['video_fps']

    # Scale frames to floats, since many applications assume that
    frames = frames / 255.0

    return frames, video_fps, frames.shape[0]


def read_video_frames(folder_path: Path) -> Tuple[Iterable[torch.Tensor], int]:
    """
    Reads in video frames given at a particular folder path. Assumes that the frames have names
    in the pattern name_x, where x is the frame number.

    Arguments:
        folder_path: The folder path where the frames are located

    Returns:
        A generator of video frames. Each frame is a Tensor of shape C x H x W with range [0, 1].
    """
    frame_names = sorted(folder_path.glob(
        '*'), key=lambda filepath: int(filepath.stem))

    frame_generator = map(lambda frame_name: read_image(
        str(frame_name)) / 255.0, frame_names)

    return frame_generator, len(frame_names)


def get_video_frame_names(folder_path: Path) -> Tuple[Iterable[str], int]:
    """
    Reads the filenames that represent the video frames.

    Arguments:
        folder_path: The location of the video frames

    Returns:
        A generator that produces the filenames for each frame
    """
    frame_names = sorted(
        folder_path.glob('*'), key=lambda filepath: int(filepath.stem))

    name_generator = map(lambda filepath: filepath.name, frame_names)

    return name_generator, len(frame_names)


class VideoDataset(Dataset):
    def __init__(self, video_path: Path):
        '''
        Creates a dataset from a single video file. Loads in the whole video into memory
        '''
        self.frames, _, self.metadata = read_video(
            str(video_path), output_format='TCHW')

    def __len__(self):
        return self.frames.shape[0]

    def __getitem__(self, idx):
        return self.frames[idx]


class VideoFramesDataset(Dataset):
    def __init__(self, folder_path: Path):
        '''
        Loads a dataset for frames under a single folder.
        Assumes frames are named with a number, such as 1.jpg, 123.jpg, etc
        '''
        self.file_names = sorted(folder_path.glob(
            '*'), key=lambda filename: int(filename.stem))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        '''
        Returns images as tensors of shape 3 x H x W with range [0, 1]
        '''
        return read_image(str(self.file_names[idx])) / 255.0


class VideoFilenamesDataset(Dataset):
    def __init__(self, folder_path: Path):
        '''
        Creates a simple `dataset` that holds the original filenames of the frames
        '''
        self.file_names = sorted(folder_path.glob(
            '*'), key=lambda filename: int(filename.stem))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        return self.file_names[idx]


# Simple test code
if __name__ == '__main__':
    from pathlib import Path

    test_frames = torch.randn((10, 3, 256, 256))

    generate_mp4(Path('test.mp4'), test_frames)

    new_frames, fps, vid_len = read_mp4('test.mp4')
