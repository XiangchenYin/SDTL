import os
import torch
import torch.utils.data
import PIL
from PIL import Image
import re
from datasets.data_augment import *



class LSRW:
    def __init__(self, config):
        self.config = config
    def get_loaders(self):

        train_dataset = AllWeatherDataset(self.config, os.path.join(self.config.data.data_dir, 'Train'),
                                          patch_size=self.config.data.patch_size)
        val_dataset = AllWeatherDataset(self.config, os.path.join(self.config.data.data_dir, 'Eval'),
                                        patch_size=self.config.data.patch_size, train=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                                 num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class AllWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, config, dir, patch_size, train=True):
        super().__init__()

        self.dir = dir
        self.train = train
        self.config = config

        input_subpath = 'low'
        gt_subpath = 'high'
        imgs = os.listdir(os.path.join(self.dir, input_subpath))

        input_names = [os.path.join(input_subpath, i.strip()) for i in imgs]
        gt_names = [i.strip().replace(input_subpath, gt_subpath) for i in input_names]

        self.input_names = input_names
        self.gt_names = gt_names
        self.patch_size = patch_size
        if self.train:
            self.transforms = PairCompose([
                PairRandomCrop(self.patch_size),
                # transforms.RandomChoice([
                #     # PairRandomRotation(degrees=30, resample=False, expand=False, center=None),
                #     # PairColorJitter(brightness=0, contrast=0.5, saturation=0.5, hue=0.5),
                #     PairRandomHorizontalFilp(),
                #     PairRandomGrayscale(num_output_channels=3),
                #     # PairRandomVerticalFlip(),
                #     # PairRandomScale([0.75, 1, 1.25]),
                # ]),
                PairToTensor()
            ])
        else:
            self.transforms = PairCompose([
                PairToTensor()
            ])

    def get_images(self, index):
        input_name = self.input_names[index].replace('\n', '')
        gt_name = self.gt_names[index].replace('\n', '')
        # img_id = re.split('/', input_name)[-1][:-4]
        img_id = re.split('/', input_name)[-1]

        input_img = Image.open(os.path.join(self.dir, input_name)) if self.dir else PIL.Image.open(input_name)
        gt_img = Image.open(os.path.join(self.dir, gt_name)) if self.dir else PIL.Image.open(gt_name)

        input_img, gt_img = self.transforms(input_img, gt_img)

        return torch.cat([input_img, gt_img], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
