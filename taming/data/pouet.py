import os
import numpy as np
import cv2
import albumentations
from PIL import Image
from torch.utils.data import Dataset
from glob import glob
import random

# With semantic map and scene label
class PouetBase(Dataset):
    def __init__(self, config=None, size=None, random_crop=False, interpolation="bicubic", crop_size=None, coord=False, val_size=1000):
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

        size = None if size is not None and size<=0 else size
        self.size = size
        if crop_size is None:
            self.crop_size = size if size is not None else None
        else:
            self.crop_size = crop_size
        if self.size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
            self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                 interpolation=self.interpolation)
            self.segmentation_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                        interpolation=cv2.INTER_NEAREST)

        if crop_size is not None:
            self.center_crop = not random_crop
            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=self.crop_size, width=self.crop_size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.crop_size, width=self.crop_size)
            self.preprocessor = self.cropper

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)

        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        if self.size is not None:
            image = self.image_rescaler(image=image)["image"]
      
        if self.size is not None:
            processed = self.preprocessor(image=image)
        else:
            processed = {"image": image}
        example["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32)

        if self.coord == True:
            h,w,_ = example["image"].shape
            example["coord"] = np.arange(h*w).reshape(h,w,1)/(h*w)

        return example


class PouetTrain(PouetBase):
    # default to random_crop=True
    def __init__(self, config=None, size=None, random_crop=True, interpolation="bicubic", crop_size=None, coord=False):
        super().__init__(config=config, size=size, random_crop=random_crop,
                          interpolation=interpolation, crop_size=crop_size, coord=coord)

    def get_split(self):
        return "train"


class PouetValidation(PouetBase):
    def get_split(self):
        return "validation"


if __name__ == "__main__":
    dset = PouetValidation()
    print(dset[0])
    # TODO debug code would go here
