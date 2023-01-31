import torch
from torch.nn import functional as F

import os
import shutil
import logging
import cv2
import numpy as np
from PIL import Image

import sys
import time
import pandas as pd
import pdb
from torchvision import transforms


def save_color_mask(pred_masks, save_base_path, video_name_list, v_pallete, resize, resized_mask_size, T=10):
    # pred_mask: [bs*5, N_CLASSES, 224, 224]
    # print(f"=> {len(video_name_list)} videos in this batch")

    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path, exist_ok=True)

    BT, N_CLASSES, H, W = pred_masks.shape
    bs = BT // T
    
    pred_masks = torch.softmax(pred_masks, dim=1)
    pred_masks = torch.argmax(pred_masks, dim=1) # [BT, 224, 224]
    pred_masks = pred_masks.cpu().numpy()
    
    pred_rgb_masks = np.zeros((pred_masks.shape + (3,)), np.uint8) # [BT, H, W, 3]
    for cls_idx in range(N_CLASSES):
        rgb = v_pallete[cls_idx]
        pred_rgb_masks[pred_masks == cls_idx] = rgb
    pred_rgb_masks = pred_rgb_masks.reshape(bs, T, H, W, 3)

    for idx in range(bs):
        video_name = video_name_list[idx]
        mask_save_path = os.path.join(save_base_path, video_name)
        if not os.path.exists(mask_save_path):
            os.makedirs(mask_save_path, exist_ok=True)
        one_video_masks = pred_rgb_masks[idx] # [5, 224, 224, 3]
        for video_id in range(len(one_video_masks)):
            one_mask = one_video_masks[video_id]
            output_name = "%s_%d.png"%(video_name, video_id)
            im = Image.fromarray(one_mask)#.convert('RGB')
            if resize:
                im = im.resize(resized_mask_size)
            im.save(os.path.join(mask_save_path, output_name), format='PNG')



if __name__ == "__main__":
    one_mask = torch.randn(224, 224)
    one_mask = (torch.sigmoid(one_mask) > 0.5).numpy().astype(np.uint8)
    one_real_mask = one_mask * 255

    pdb.set_trace()


