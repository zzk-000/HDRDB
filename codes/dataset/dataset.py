#!/usr/bin/env python
# -*- coding:utf-8 -*-
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch
import numpy as np
import os
import cv2


class PNG_dataset(data.Dataset):
    def __init__(self, base_dir, gt_dir, num, is_random=False):
        super(PNG_dataset, self).__init__()
        self.base_list = []
        self.gt_list = []
        name_list = os.listdir(base_dir)
        for name in name_list:
            self.base_list.append(base_dir + '/' + name)
            self.gt_list.append(gt_dir + '/' + name)
        if is_random:
            rnd_index = np.arange(len(self.base_list))
            np.random.shuffle(rnd_index)
            self.base_list = np.array(self.base_list)[rnd_index]
            self.gt_list = np.array(self.gt_list)[rnd_index]
        if num != 0:
            self.base_list = self.base_list[:num]
            self.gt_list = self.gt_list[:num]

    def __getitem__(self, index):
        base = np.array(cv2.imread(self.base_list[index], flags=-1)[:, :, ::-1], np.float32)/65535
        gt = np.array(cv2.imread(self.gt_list[index], flags=-1)[:, :, ::-1], np.float32)/65535

        base = torch.from_numpy(base).float().permute(2, 0, 1)
        gt = torch.from_numpy(gt).float().permute(2, 0, 1)
        return {'base': base, 'gt': gt}

    def __len__(self):
        return len(self.base_list)


def create_dataset(opt):
    train_set = PNG_dataset(opt.train_base, opt.train_gt, opt.num)
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True, drop_last=False)
    print('--PNG数据加载完成')
    return train_loader
