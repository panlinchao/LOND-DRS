import numpy as np
from numpy.testing import assert_array_almost_equal
import torchvision
from PIL import Image
from copy import deepcopy
import os.path as osp
import random
import pickle as pkl


from data.noisy_cifar import NoisyCIFAR100


class AddOpenNoisyCIFAR100(NoisyCIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False,
                 noise_type='clean', closeset_ratio=0.0, openset_ratio=0.2, random_state=0, verbose=True,
                 ood_noise_name="place365",
                 ood_noise_root_dir="",
                 ood_noise_num=20000,
                 train_used_idx=None):
        super(AddOpenNoisyCIFAR100, self).__init__(root, train, transform, target_transform, download,
                                                noise_type, closeset_ratio, 0, random_state, False)
        self.ood_noise_name = ood_noise_name
        assert self.ood_noise_name in ("place365", "tiny_imagenet", "cifar100", "cifar20")
        if self.ood_noise_name == "cifar100":
            raise Exception(f"Original input and ood noise input are same")
        self.ood_noise_root_dir = ood_noise_root_dir
        self.ood_noise_num = ood_noise_num
        self.train_used_idx = train_used_idx or []

        self.openset_select_ids = []
        self.openset_images = []
        self.openset_noise_labels = []
        self.openset_clean_labels = []

        self._load_openset_noise(self.train_used_idx)

    def _load_openset_noise(self, train_used_noise: list):
        if self.ood_noise_name == "place365":
            openset_images = np.load(osp.join(self.ood_noise_root_dir, "places365_test.npy")).transpose((0, 2, 3, 1))
        elif self.ood_noise_name == "tiny_imagenet":
            openset_images = np.load(osp.join(self.ood_noise_root_dir, "TIN_train.npy")).transpose((0, 2, 3, 1))
        elif self.ood_noise_name == "cifar20":
            load_key = 'images_train' if self.train else 'images_test'
            openset_images = np.load(osp.join(self.ood_noise_root_dir, "cifar-20.npz"))[load_key].transpose((0, 2, 3, 1))
        else:
            # cifar100
            with open(osp.join(self.ood_noise_root_dir, 'train'), "rb") as f:
                openset_images = pkl.load(f, encoding='latin1')['data'].reshape((50000, 3, 32, 32)).transpose(
                    (0, 2, 3, 1))

        idx = list(range(openset_images.shape[0]))
        if not self.train:
            idx = list(set(idx) - set(train_used_noise))
        random.shuffle(idx)
        self.ood_noise_name = min(self.ood_noise_num, len(idx))
        idx = idx[0: self.ood_noise_num]
        self.openset_select_ids = idx

        self.openset_noise_labels = np.random.choice(list(range(self.nb_classes)),
                                                     size=self.ood_noise_num,
                                                     replace=True).tolist()
        self.openset_clean_labels = [-1] * self.ood_noise_num
        self.openset_images = openset_images[idx]

        self.data = np.concatenate([self.data, self.openset_images])
        if self.train:
            self.noisy_labels = self.noisy_labels + self.openset_noise_labels
            self.is_clean = np.concatenate([self.is_clean, np.array([False] * self.ood_noise_num, dtype=bool)])
        # self.noisy_labels = self.noisy_labels + self.openset_noise_labels
        self.targets = self.targets + self.openset_clean_labels
        self.open_cls.append(-1)