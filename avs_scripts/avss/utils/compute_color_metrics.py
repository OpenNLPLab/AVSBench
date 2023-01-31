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
from torchvision import transforms

import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

import pdb


def _batch_miou_fscore(output, target, nclass, T, beta2=0.3):
    """batch mIoU and Fscore"""
    # output: [BF, C, H, W],
    # target: [BF, H, W]
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1
    target = target.float() + 1
    # pdb.set_trace()
    predict = predict.float() * (target > 0).float() # [BF, H, W]
    intersection = predict * (predict == target).float() # [BF, H, W]
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    batch_size = target.shape[0] // T
    cls_count = torch.zeros(nclass).float()
    ious = torch.zeros(nclass).float()
    fscores = torch.zeros(nclass).float()

    # vid_miou_list = torch.zeros(target.shape[0]).float()
    vid_miou_list = []
    for i in range(target.shape[0]):
        area_inter = torch.histc(intersection[i].cpu(), bins=nbins, min=mini, max=maxi) # TP
        area_pred = torch.histc(predict[i].cpu(), bins=nbins, min=mini, max=maxi) # TP + FP
        area_lab = torch.histc(target[i].cpu(), bins=nbins, min=mini, max=maxi) # TP + FN
        area_union = area_pred + area_lab - area_inter
        assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
        iou = 1.0 * area_inter.float() / (2.220446049250313e-16 + area_union.float())
        # iou[torch.isnan(iou)] = 1.
        ious += iou
        cls_count[torch.nonzero(area_union).squeeze(-1)] += 1

        precision = area_inter / area_pred
        recall = area_inter / area_lab
        fscore = (1 + beta2) * precision * recall / (beta2 * precision + recall)
        fscore[torch.isnan(fscore)] = 0.
        fscores += fscore

        vid_miou_list.append(torch.sum(iou) / (torch.sum( iou != 0 ).float()))

    return ious, fscores, cls_count, vid_miou_list


def calc_color_miou_fscore(pred, target, T=10):
    r"""
    J measure
        param: 
            pred: size [BF x C x H x W], C is category number including background
            target: size [BF x H x W]
    """  
    nclass = pred.shape[1]
    pred = torch.softmax(pred, dim=1) # [BF, C, H, W]
    # miou, fscore, cls_count = _batch_miou_fscore(pred, target, nclass, T) 
    miou, fscore, cls_count, vid_miou_list = _batch_miou_fscore(pred, target, nclass, T) 
    return miou, fscore, cls_count, vid_miou_list


def _batch_intersection_union(output, target, nclass, T):
    """mIoU"""
    # output: [BF, C, H, W],
    # target: [BF, H, W]
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1
    target = target.float() + 1

    # pdb.set_trace()

    predict = predict.float() * (target > 0).float() # [BF, H, W]
    intersection = predict * (predict == target).float() # [BF, H, W]
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    batch_size = target.shape[0] // T
    cls_count = torch.zeros(nclass).float()
    ious = torch.zeros(nclass).float()
    for i in range(target.shape[0]):
        area_inter = torch.histc(intersection[i].cpu(), bins=nbins, min=mini, max=maxi)
        area_pred = torch.histc(predict[i].cpu(), bins=nbins, min=mini, max=maxi)
        area_lab = torch.histc(target[i].cpu(), bins=nbins, min=mini, max=maxi)
        area_union = area_pred + area_lab - area_inter
        assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
        iou = 1.0 * area_inter.float() / (2.220446049250313e-16 + area_union.float())
        ious += iou
        cls_count[torch.nonzero(area_union).squeeze(-1)] += 1
        # pdb.set_trace()
    # ious = ious / cls_count
    # ious[torch.isnan(ious)] = 0
    # pdb.set_trace()
    # return area_inter.float(), area_union.float()
    # return ious
    return ious, cls_count


def calc_color_miou(pred, target, T=10):
    r"""
    J measure
        param: 
            pred: size [BF x C x H x W], C is category number including background
            target: size [BF x H x W]
    """  
    nclass = pred.shape[1]
    pred = torch.softmax(pred, dim=1) # [BF, C, H, W]
    # correct, labeled = _batch_pix_accuracy(pred, target)
    # inter, union = _batch_intersection_union(pred, target, nclass, T)
    ious, cls_count = _batch_intersection_union(pred, target, nclass, T)

    # pixAcc = 1.0 * correct / (2.220446049250313e-16 + labeled)
    # IoU = 1.0 * inter / (2.220446049250313e-16 + union)
    # mIoU = IoU.mean().item()
    # pdb.set_trace()
    # return mIoU
    return ious, cls_count


def calc_binary_miou(pred, target, eps=1e-7, size_average=True):
    r"""
        param: 
            pred: size [N x C x H x W]
            target: size [N x H x W]
        output:
            iou: size [1] (size_average=True) or [N] (size_average=False)
    """
    # assert len(pred.shape) == 3 and pred.shape == target.shape
    nclass = pred.shape[1]
    pred = torch.softmax(pred, dim=1) # [BF, C, H, W]
    pred = torch.argmax(pred, dim=1) # [BF, H, W]
    binary_pred = (pred != (nclass - 1)).float() # [BF, H, W]
    # pdb.set_trace()
    pred = binary_pred
    target = (target != (nclass - 1)).float()

    N = pred.size(0)
    num_pixels = pred.size(-1) * pred.size(-2)
    no_obj_flag = (target.sum(2).sum(1) == 0)

    temp_pred = torch.sigmoid(pred)
    pred = (temp_pred > 0.5).int()
    inter = (pred * target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)

    inter_no_obj = ((1 - target) * (1 - pred)).sum(2).sum(1)
    inter[no_obj_flag] = inter_no_obj[no_obj_flag]
    union[no_obj_flag] = num_pixels

    iou = torch.sum(inter / (union+eps)) / N
    # pdb.set_trace()
    return iou



if __name__ == "__main__":
    print("done")
    pred = torch.ones(5, 10, 10)
    pred[:, :5, :5] = 0
    pred[:, :]
    label = torch.ones(5, 10, 10)

