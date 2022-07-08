import os
from wave import _wave_params
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import pickle

import cv2
from PIL import Image
from torchvision import transforms

from config import cfg
import pdb



def load_image_in_PIL_to_Tensor(path, mode='RGB', transform=None):
    img_PIL = Image.open(path).convert(mode)
    if transform:
        img_tensor = transform(img_PIL)
        return img_tensor
    return img_PIL


def load_audio_lm(audio_lm_path):
    with open(audio_lm_path, 'rb') as fr:
        audio_log_mel = pickle.load(fr)
    audio_log_mel = audio_log_mel.detach() # [5, 1, 96, 64]
    return audio_log_mel


class S4Dataset(Dataset):
    """Dataset for single sound source segmentation"""
    def __init__(self, split='train'):
        super(S4Dataset, self).__init__()
        self.split = split
        self.mask_num = 1 if self.split == 'train' else 5
        df_all = pd.read_csv(cfg.DATA.ANNO_CSV, sep=',')
        self.df_split = df_all[df_all['split'] == split]
        print("{}/{} videos are used for {}".format(len(self.df_split), len(df_all), self.split))
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])


    def __getitem__(self, index):
        df_one_video = self.df_split.iloc[index]
        video_name, category = df_one_video[0], df_one_video[2]
        img_base_path =  os.path.join(cfg.DATA.DIR_IMG, self.split, category, video_name)
        audio_lm_path = os.path.join(cfg.DATA.DIR_AUDIO_LOG_MEL, self.split, category, video_name + '.pkl')
        mask_base_path = os.path.join(cfg.DATA.DIR_MASK, self.split, category, video_name)
        audio_log_mel = load_audio_lm(audio_lm_path)
        # audio_lm_tensor = torch.from_numpy(audio_log_mel)
        imgs, masks = [], []
        for img_id in range(1, 6):
            img = load_image_in_PIL_to_Tensor(os.path.join(img_base_path, "%s_%d.png"%(video_name, img_id)), transform=self.img_transform)
            imgs.append(img)
        for mask_id in range(1, self.mask_num + 1):
            mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png"%(video_name, mask_id)), transform=self.mask_transform, mode='1')
            masks.append(mask)
        imgs_tensor = torch.stack(imgs, dim=0)
        masks_tensor = torch.stack(masks, dim=0)

        if self.split == 'train':
            return imgs_tensor, audio_log_mel, masks_tensor
        else:
            return imgs_tensor, audio_log_mel, masks_tensor, category, video_name


    def __len__(self):
        return len(self.df_split)




if __name__ == "__main__":
    train_dataset = S4Dataset('train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                     batch_size=2,
                                                     shuffle=False,
                                                     num_workers=8,
                                                     pin_memory=True)

    for n_iter, batch_data in enumerate(train_dataloader):
        imgs, audio, mask = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
        # imgs, audio, mask, category, video_name = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
        pdb.set_trace()
    print('n_iter', n_iter)
    pdb.set_trace()
