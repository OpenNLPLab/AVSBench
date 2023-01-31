import os
import time
import random
import shutil
import torch
import numpy as np
import argparse
import logging

from config import cfg
from color_dataloader import V2Dataset
from torchvggish import vggish

from utils import pyutils
from utils.utility import logger 
from utils.system import setup_logging

from utils.vis_mask import save_color_mask
from utils.compute_color_metrics import calc_color_miou_fscore

import pdb

import sys
sys.path.append('../')
from color_dataloader import get_v2_pallete


class audio_extractor(torch.nn.Module):
    def __init__(self, cfg, device):
        super(audio_extractor, self).__init__()
        self.audio_backbone = vggish.VGGish(cfg, device)

    def forward(self, audio):
        audio_fea = self.audio_backbone(audio)
        return audio_fea


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_name", default="AVSS", type=str, help="the AVSS setting")
    parser.add_argument("--visual_backbone", default="resnet", type=str, help="use resnet50 or pvt-v2 as the visual backbone")

    parser.add_argument("--test_batch_size", default=4, type=int)
    parser.add_argument("--max_epoches", default=15, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)

    parser.add_argument("--tpavi_stages", default=[], nargs='+', type=int, help='add non-local block in which stages: [0, 1, 2, 3')
    parser.add_argument("--tpavi_vv_flag", action='store_true', default=False, help='visual-visual self-attention')
    parser.add_argument("--tpavi_va_flag", action='store_true', default=False, help='visual-audio cross-attention')

    parser.add_argument("--weights",type=str)
    parser.add_argument("--save_pred_mask", action='store_true', default=False, help="save predited masks or not")
    parser.add_argument('--log_dir', default='./test_logs', type=str)

    args = parser.parse_args()

    if (args.visual_backbone).lower() == "resnet":
        from model import ResNet_AVSModel as AVSModel
        print('==> Use ResNet50 as the visual backbone...')
    elif (args.visual_backbone).lower() == "pvt":
        from model import PVT_AVSModel as AVSModel
        print('==> Use pvt-v2 as the visual backbone...')
    else:
        raise NotImplementedError("only support the resnet50 and pvt-v2")

    # Log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    # Logs
    prefix = args.session_name
    log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S')))
    if os.path.exists(log_dir):
        log_dir = os.path.join(args.log_dir, '{}_{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S_'), np.random.randint(1, 10)))
    args.log_dir = log_dir

    # Save scripts
    script_path = os.path.join(log_dir, 'scripts')
    if not os.path.exists(script_path):
        os.makedirs(script_path, exist_ok=True)
    
    scripts_to_save = ['train.sh', 'train.py', 'test.sh', 'test.py', 'config.py', 'color_dataloader.py', './model/ResNet_AVSModel.py', './model/PVT_AVSModel.py', 'loss.py']
    for script in scripts_to_save:
        dst_path = os.path.join(script_path, script)
        try:
            shutil.copy(script, dst_path)
        except IOError:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(script, dst_path)

    # Set logger
    log_path = os.path.join(log_dir, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    setup_logging(filename=os.path.join(log_path, 'log.txt'))
    logger = logging.getLogger(__name__)
    logger.info('==> Config: {}'.format(cfg))
    logger.info('==> Arguments: {}'.format(args))
    logger.info('==> Experiment: {}'.format(args.session_name))

    # Model
    model = AVSModel.Pred_endecoder(channel=256, \
                                        config=cfg, \
                                        tpavi_stages=args.tpavi_stages, \
                                        tpavi_vv_flag=args.tpavi_vv_flag, \
                                        tpavi_va_flag=args.tpavi_va_flag)
    model.load_state_dict(torch.load(args.weights))
    model = torch.nn.DataParallel(model).cuda()
    logger.info('Load trained model %s'%args.weights)

    # audio backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_backbone = audio_extractor(cfg, device)
    audio_backbone.cuda()
    audio_backbone.eval()

    # Test data
    split = 'test'
    test_dataset = V2Dataset(split)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=args.test_batch_size,
                                                        shuffle=False,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)

    N_CLASSES = test_dataset.num_classes


    # Test
    model.eval()

    # for save predicted rgb masks
    v2_pallete = get_v2_pallete(cfg.DATA.LABEL_IDX_PATH)
    resize_pred_mask = cfg.DATA.RESIZE_PRED_MASK
    if resize_pred_mask:
        pred_mask_img_size = cfg.DATA.SAVE_PRED_MASK_IMG_SIZE
    else:
        pred_mask_img_size = cfg.DATA.IMG_SIZE

    # metrics
    miou_pc = torch.zeros((N_CLASSES)) # miou value per class (total sum)
    Fs_pc = torch.zeros((N_CLASSES)) # f-score per class (total sum)
    cls_pc = torch.zeros((N_CLASSES)) # count per class
    with torch.no_grad():
        for n_iter, batch_data in enumerate(test_dataloader):
            imgs, audio, mask, vid_temporal_mask_flag, gt_temporal_mask_flag, video_name_list = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]

            #! notice:
            vid_temporal_mask_flag = vid_temporal_mask_flag.cuda()
            gt_temporal_mask_flag  = gt_temporal_mask_flag.cuda()

            imgs = imgs.cuda()
            # audio = audio.cuda()
            mask = mask.cuda()
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B*frame, C, H, W)
            mask = mask.view(B*frame, H, W)
    
            #! notice
            vid_temporal_mask_flag = vid_temporal_mask_flag.view(B*frame) # [B*T]
            gt_temporal_mask_flag  = gt_temporal_mask_flag.view(B*frame)  # [B*T]

            # audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])
            with torch.no_grad():
                audio_feature = audio_backbone(audio)
                #! notice:
                audio_feature = audio_feature * vid_temporal_mask_flag.unsqueeze(-1)


            output, _, _,= model(imgs, audio_feature, vid_temporal_mask_flag) # [5, 24, 224, 224] = [bs=1 * T=5, 24, 224, 224]
            if args.save_pred_mask:
                mask_save_path = os.path.join(log_dir, 'pred_masks')
                save_color_mask(output, mask_save_path, video_name_list, v2_pallete, resize_pred_mask, pred_mask_img_size, T=10)

            _miou_pc, _fscore_pc, _cls_pc = calc_color_miou_fscore(output, mask,)
            # compute miou, J-measure
            miou_pc += _miou_pc
            cls_pc += _cls_pc
            # compute f-score, F-measure
            Fs_pc += _fscore_pc

            batch_iou = miou_pc / cls_pc
            batch_iou[torch.isnan(batch_iou)] = 0
            batch_iou = torch.sum(batch_iou) / torch.sum(cls_pc != 0)
            batch_fscore = Fs_pc / cls_pc
            batch_fscore[torch.isnan(batch_fscore)] = 0
            batch_fscore = torch.sum(batch_fscore) / torch.sum(cls_pc != 0)
            print('n_iter: {}, iou: {}, F_score: {}, cls_num: {}'.format(n_iter, batch_iou, batch_fscore, torch.sum(cls_pc != 0).item()))

        miou_pc = miou_pc / cls_pc
        print(f"[test miou] {torch.sum(torch.isnan(miou_pc)).item()} classes are not predicted in this batch")
        miou_pc[torch.isnan(miou_pc)] = 0
        miou = torch.mean(miou_pc).item()
        miou_noBg = torch.mean(miou_pc[:-1]).item()
        f_score_pc = Fs_pc / cls_pc
        print(f"[test fscore] {torch.sum(torch.isnan(f_score_pc)).item()} classes are not predicted in this batch")
        f_score_pc[torch.isnan(f_score_pc)] = 0
        f_score = torch.mean(f_score_pc).item()
        f_score_noBg = torch.mean(f_score_pc[:-1]).item()

        logger.info('test | cls {}, miou: {:.4f}, miou_noBg: {:.4f}, F_score: {:.4f}, F_score_noBg: {:.4f}'.format(\
            torch.sum(cls_pc !=0).item(), miou, miou_noBg,  f_score, f_score_noBg))
        # pdb.set_trace()






