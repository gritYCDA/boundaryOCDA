from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms
import cv2

class BaseDataset(data.Dataset):
    def __init__(self, root, list_path, set_,
                 max_iters, image_size, labels_size, mean):
        self.root = Path(root)
        self.set = set_
        self.list_path = list_path.format(self.set)
        self.image_size = image_size
        if labels_size is None:
            self.labels_size = self.image_size
        else:
            self.labels_size = labels_size
        self.mean = mean
        with open(self.list_path) as f:
            self.img_ids = [i_id.strip() for i_id in f]
            # self.img_ids_rev = self.img_ids[::-1]

        # if max_iters is not None:
            # self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for name, name_rev in zip(self.img_ids, self.img_ids_rev):
        #     img_file, img_file_rev, label_file = self.get_metadata(name, name_rev)
        #     self.files.append((img_file, img_file_rev, label_file, name))

        for idx, name in enumerate(self.img_ids):
            try:
                name_next = self.img_ids[idx+1]
            except:
                name_next = self.img_ids[idx]

            img_file, img_file_next, label_file = self.get_metadata(name, name_next)
            self.files.append((img_file, img_file_next, label_file, name, name_next))

        self.data_aug = self.data_transform()

    def data_transform(self):
        color_jitter = transforms.ColorJitter(0, 1, 0, 0)

        return color_jitter

    def get_metadata(self, name, name_next):
        raise NotImplementedError

    def __len__(self):
        return len(self.files)

    def preprocess(self, image):
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        return image.transpose((2, 0, 1))

    def get_image(self, file):
        return _load_img(file, self.image_size, Image.BICUBIC, rgb=True)

    def get_labels(self, file):
        return _load_img(file, self.labels_size, Image.NEAREST, rgb=False)


def _load_img(file, size, interpolation, rgb):
    # color_jitter = transforms.ColorJitter(0, 0, 0, 0.5)
    gray_scale = transforms.Grayscale(num_output_channels=3)
    # gaus_blur = transforms.GaussianBlur(3)
    img = Image.open(file)
    if rgb:
        img = img.convert('RGB')
    img = img.resize(size, interpolation)        
    if rgb:
        img_aug = gray_scale(img)
    if not rgb:
        return np.asarray(img, np.float32)
    else:
        return np.asarray(img, np.float32), np.asarray(img_aug, np.float32)
