import numpy as np

from advent.utils import project_root
from advent.utils.serialization import json_load
from advent.dataset.base_dataset import BaseDataset

DEFAULT_INFO_PATH = project_root / 'advent/dataset/compound_list/info.json'


# class BDDdataset(BaseDataset):
#     def __init__(self, root, list_path, set='val',
#                  max_iters=None,
#                  crop_size=(321, 321), mean=(128, 128, 128),
#                  load_labels=True,
#                  info_path=DEFAULT_INFO_PATH, labels_size=None):
#         super().__init__(root, list_path, set, max_iters, crop_size, labels_size, mean)
#         DEFAULT_INFO_PATH = project_root / 'advent/dataset/compound_list/info.json'
#         self.load_labels = load_labels
#         self.info = json_load(info_path)
#         self.class_names = np.array(self.info['label'], dtype=np.str)
#         self.mapping = np.array(self.info['label2train'], dtype=np.int)
#         self.map_vector = np.zeros((self.mapping.shape[0],), dtype=np.int64)
#         for source_label, target_label in self.mapping:
#             self.map_vector[source_label] = target_label

#         self.id_to_trainid = {3:255, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10, 12:255, 13:11, 14:12, 15:13,
#                               16:255, 17:14, 18:255}

#     def get_metadata(self, name):
#         if self.set  == 'train':
#             img_file = self.root / name
#             label_file = None
#             # import pdb
#             # pdb.set_trace()

#         else:
#             img_file = self.root / name
#             label_file = name.replace("images", "labels").replace(".jpg","_train_id.png")
#             label_file = self.root / label_file
#         return img_file, label_file

#     def map_labels(self, input_):
#         return self.map_vector[input_.astype(np.int64, copy=False)]

#     def __getitem__(self, index):
#         img_file, label_file, name = self.files[index]
#         if not label_file == None:
#             label = self.get_labels(label_file)
#             # label = self.map_labels(label).copy()
#             label_copy = 255 * np.ones(label.shape, dtype=np.float32)
#             for k, v in self.id_to_trainid.items():
#                 label_copy[label == k] = v

#         image = self.get_image(img_file)
#         image = self.preprocess(image)
#         if label_file == None:
#             label_copy = image.copy()
#         return image.copy(), label_copy, np.array(image.shape), name

class BDDdataset(BaseDataset):
    def __init__(self, root, list_path, set='val',
                 max_iters=None,
                 crop_size=(321, 321), mean=(128, 128, 128),
                 load_labels=True,
                 info_path=DEFAULT_INFO_PATH, labels_size=None):
        super().__init__(root, list_path, set, max_iters, crop_size, labels_size, mean)
        self.load_labels = load_labels
        self.info = {}
        self.info = json_load(info_path)
        self.class_names = np.array(self.info['label'], dtype=np.str)
        self.mapping = np.array(self.info['label2train'], dtype=np.int)

        self.map_vector = np.zeros((self.mapping.shape[0],), dtype=np.int64)
        for source_label, target_label in self.mapping:
            self.map_vector[source_label] = target_label

    def get_metadata(self, name):
        if self.set  == 'train':
            img_file = self.root / name
            label_file = None

        else:
            img_file = self.root / name
            label_file = name.replace("images", "labels").replace(".jpg","_train_id.png")
            label_file = self.root / label_file
        return img_file, label_file

    def map_labels(self, input_):
        return self.map_vector[input_.astype(np.int64, copy=False)]

    def __getitem__(self, index):
        img_file, label_file, name = self.files[index]
        if not label_file == None:
            label = self.get_labels(label_file)
            # label = self.map_labels(label).copy()
            # label_copy = 255 * np.ones(label.shape, dtype=np.float32)
            # for k, v in self.id_to_trainid.items():
            #     label_copy[label == k] = v

        image = self.get_image(img_file)
        image = self.preprocess(image)
        if label_file == None:
            label = image.copy()
        return image.copy(), label, np.array(image.shape), name
