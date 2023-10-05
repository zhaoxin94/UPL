import os.path as osp
import os

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, write_json

from .datasetbase import UPLDatasetBase


@DATASET_REGISTRY.register()
class SSUCM_OVDA(UPLDatasetBase):
    """UCM dataset: Open Vocabulary Domain Adaptation
    """
    dataset_dir = "UCM"

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, 'images')
        self.prompt_path = "./prompts/full_prompts/ucm_prompts_full.json"

        # self.train_ratio = cfg.DATASET.TRAIN_RATIO
        # assert self.train_ratio in [
        #     50, 80, 100
        # ], "not belong to the pre-defined train ratio"

        train = self._read_data(split='train')
        test = self._read_data(split='test')
        sstrain = self.read_sstrain_data()
        super().__init__(train_x=train,
                         train_u=train,
                         test=test,
                         sstrain=train)

    def _read_data(self, split='train'):
        items = []
        # if split == 'train':
        #     txt_name = "train_" + str(
        #         self.train_ratio) if self.train_ratio != 100 else "train"
        # elif split == 'test':
        #     txt_name = "test_" + str(
        #         100 - self.train_ratio) if self.train_ratio != 100 else "train"
        # else:
        #     raise NotImplementedError
        txt_name = 'train'
        txt_file = osp.join(self.dataset_dir, 'image_list', txt_name + ".txt")
        print(f"Reading {split} data in the {txt_file}")

        with open(txt_file, "r") as f:
            for line in f.readlines():
                path, label = line.split()
                label = int(label)
                class_name = path.split('/')[-2]
                item = Datum(impath=path,
                             label=label,
                             classname=class_name.lower())
                items.append(item)

        return items

    def read_sstrain_data(self, predict_label_dict=None):
        def _convert(items):
            out = []
            for item in items:
                # impath = os.path.join(path_prefix, impath)
                sub_impath = './data/' + item.impath.split('/data/')[1]
                if sub_impath in predict_label_dict:
                    item = Datum(impath=item.impath,
                                 label=predict_label_dict[sub_impath],
                                 classname=self._lab2cname[
                                     predict_label_dict[sub_impath]])
                    out.append(item)
            return out

        def _convert_no_label(items):
            out = []
            for item in items:
                new_item = Datum(impath=item.impath, label=-1, classname=None)
                out.append(new_item)
            return out

        # print(f"Reading split from {filepath}")
        # split = read_json(filepath)
        train = self._read_data(split='train')
        if predict_label_dict is not None:
            train = _convert(train)
        else:
            train = _convert_no_label(train)
        return train
    
    def add_label(self, predict_label_dict, dataset_name):
        """add label when training for self-supervised learning

        Args:
            predict_label_dict ([dict]): [a dict {'imagepath': 'label'}]
        """
        # print(predict_label_dict, 'predict_label_dict')
        print(dataset_name)
        if dataset_name == 'SSFGVCAircraft':
            sstrain = self.read_data_without_label(self.cname2lab,
                                                   "images_variant_train.txt",
                                                   predict_label_dict)
        elif dataset_name == 'SSImageNet':
            sstrain = self.read_sstrain_data(predict_label_dict)
        else:
            sstrain = self.read_sstrain_data(predict_label_dict)

        self.sstrain = sstrain
        return sstrain
