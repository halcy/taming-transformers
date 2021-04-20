import os
import numpy as np
import math
import cv2
import albumentations
from PIL import Image
from torch.utils.data import Dataset
from glob import glob
import random

# With semantic map and scene label
class PouetBase(Dataset):
    def __init__(self, config=None, size=None, interpolation="bicubic", coord=False, val_size=1000):
        self.split = self.get_split()
        self.coord = coord
        self.rnd = random.Random()
        self.rnd.seed(69)
        self.data_root = "data/pouet"

        self.all_files = sorted(glob(self.data_root + "/*.png"))
        self.rnd.shuffle(self.all_files)

        if self.split == "train":
            self.image_paths = self.all_files[val_size:]
        else:
            self.image_paths = self.all_files[:val_size]

        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.abspath(l) for l in self.image_paths],
        }

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)

        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = np.pad(
          image,
          ((0, math.ceil(image.shape[0]/16) * 16 - image.shape[0]),
          (0, math.ceil(image.shape[1]/16) * 16 - image.shape[1]),
          (0, 0))
        )

        processed = {"image": image}
        example["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32)

        if self.coord == True:
            h,w,_ = example["image"].shape
            example["coord"] = np.arange(h*w).reshape(h,w,1)/(h*w)

        return example


class PouetTrain(PouetBase):
    def get_split(self):
        return "train"


class PouetValidation(PouetBase):
    def get_split(self):
        return "validation"


if __name__ == "__main__":
    dset = PouetValidation()
    print(dset[0]["image"].shape)
    # TODO debug code would go here
