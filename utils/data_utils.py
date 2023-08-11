"""
Code adopted from pix2pixHD (https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py)
"""
from pathlib import Path
from typing import Iterator, Iterable
import torch
import math

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename: Path):
    return any(str(filename).endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir: Path):
    images = []
    assert dir.is_dir(), '%s is not a valid directory' % dir
    for fname in dir.glob("*"):
        if is_image_file(fname):
            # path = dir / fname
            path = fname  # This is duplicating the folder name for some reason
            images.append(path)
    return images


class tensor_batcher(Iterator):
    '''
    An iterable that batches tensors together
    '''

    def __init__(self, tensor_gen: Iterable[torch.Tensor], batch_size: int):
        self.generator = tensor_gen
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.generator) / self.batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        input_lst = []
        for input in self.generator:
            input_lst.append(input)

            if len(input_lst) == self.batch_size:
                yield torch.stack(input_lst)

                input_lst = []
        if len(input_lst) > 0:
            yield input_lst
