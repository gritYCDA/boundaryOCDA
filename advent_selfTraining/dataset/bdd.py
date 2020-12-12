import numpy as np

from advent.utils import project_root
from advent.utils.serialization import json_load
from advent.dataset.base_dataset import BaseDataset
import cv2

DEFAULT_INFO_PATH = project_root / 'advent/dataset/compound_list/info.json'


class BDDataSet(BaseDataset):
    def __init__(self, root, list_path, set='val',
                 max_iters=None,
                 crop_size=(321, 321), mean=(128, 128, 128),
                 load_labels=True,
                 info_path=DEFAULT_INFO_PATH, labels_size=None):
        super().__init__(root, list_path, set, max_iters, crop_size, labels_size, mean)
        DEFAULT_INFO_PATH = project_root / 'advent/dataset/compound_list/info.json'
        self.load_labels = load_labels
        self.info = json_load(info_path)
        self.class_names = np.array(self.info['label'], dtype=np.str)
        self.mapping = np.array(self.info['label2train'], dtype=np.int)

        self.map_vector = np.zeros((self.mapping.shape[0],), dtype=np.int64)
        for source_label, target_label in self.mapping:
            self.map_vector[source_label] = target_label

    def get_metadata(self, name, name_next):
        if self.set  == 'train':
            name = name.split('\t')[0]
            name_next = name_next.split('\t')[0]
            img_file = self.root / name
            img_file_rev = self.root / name_next
            label_file = None

        else:
            img_file = self.root / name
            img_file_rev = img_file
            label_file = name.replace("images", "labels").replace(".jpg","_train_id.png")
            label_file = self.root / label_file
        return img_file, img_file_rev, label_file

    def map_labels(self, input_):
        return self.map_vector[input_.astype(np.int64, copy=False)]

    def __getitem__(self, index):
        img_file, img_file_rev, label_file, name, name_next = self.files[index]
        if not label_file == None:
            label = self.get_labels(label_file)

        image, image_aug = self.get_image(img_file)
        img_file_rev, _ = self.get_image(img_file_rev)
        image = self.preprocess(image)
        # edge_image = cv2.Canny(image, 50,200)
        image_aug = self.preprocess(image_aug)
        img_file_rev = self.preprocess(img_file_rev)
        if label_file == None:
            label = image.copy()
        return image.copy(), img_file_rev.copy(), label, np.array(image.shape), name, name_next
