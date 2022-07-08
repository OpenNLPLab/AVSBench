import os, glob, sys
import pandas as pd
import numpy as np
import random, pickle
from tqdm import tqdm
import imageio
from moviepy.editor import VideoFileClip
import cv2
import shutil
import torch
sys.path.append('../avs_scripts/avs_s4')
from torchvggish import vggish_input
import json

import pdb

AVSBenchDataset = ['baby_laughter', 'female_singing', 'male_speech',
'dog_barking', 'cat_meowing', 'lions_roaring', 'horse_clip-clop', 'coyote_howling', 'mynah_bird_singing',
'playing_acoustic_guitar', 'playing_tabla', 'playing_violin', 'playing_piano', 'playing_ukulele', 'playing_glockenspiel',
'helicopter', 'race_car', 'driving_buses', 'ambulance_siren',
'typing_on_computer_keyboard', 'chainsawing_trees', 'cap_gun_shooting', 'lawn_mowing'] # 23 categories


def video_frame_sample(frame_interval, video_length, sample_num):
    num = []
    for l in range(video_length):
        for i in range(sample_num):
            num.append(int(l * frame_interval + (i * 1.0 / sample_num) * frame_interval))
    return num


def extract_frame_for_each_video(root_path, trimed_video_base_path, df_one_video):
    video_name, start_time, category, split = df_one_video[0], df_one_video[1], df_one_video[2], df_one_video[3]
    trimed_video_path = os.path.join(trimed_video_base_path, split, category, video_name + ".mp4")
    extract_frames_base_path = os.path.join(root_path, "visual_frames")

    t = 5 if (start_time <= 5) else (10 - start_time) # length of video
    sample_num = 16 # frame number for each second

    vid = imageio.get_reader(trimed_video_path, 'ffmpeg')
    frame_interval = int(round(vid.get_meta_data()['fps']))

    frame_num = video_frame_sample(frame_interval, t, sample_num)
    imgs = []
    for i, im in enumerate(vid):
        x_im = cv2.resize(im, (224, 224))
        imgs.append(x_im)
    vid.close()

    frame_save_path = os.path.join(extract_frames_base_path, split, category, video_name)
    if not os.path.exists(frame_save_path):
        os.makedirs(frame_save_path)

    extract_frame = []
    for n in frame_num:
        if n >= len(imgs):
            n = len(imgs) - 1
        extract_frame.append(imgs[n])

    count = 0
    for k, _ in enumerate(extract_frame):
        if k % sample_num == 15:
            count += 1
            cv2.imwrite(os.path.join(frame_save_path, video_name + '_' + str(count) + '.png'), cv2.cvtColor(extract_frame[k], cv2.COLOR_RGB2BGR))


def extract_frames(root_path="../avsbench_data/Single-source/s4_data",\
                   csv_path="../avsbench_data/Single-source/s4_meta_data.csv"):
    """sampling video frames """
    trimed_video_path = os.path.join(root_path, "raw_videos")
    anno_path = csv_path
    df_anno = pd.read_csv(anno_path, sep=',')
    wrong_video_list = []
    count = 0
    for item in AVSBenchDataset:
        print(f"processing for [{item}]...")
        df_one_class = df_anno[df_anno['category'] == item]
        for i in tqdm(range(len(df_one_class))):
            try:
                extract_frame_for_each_video(root_path, trimed_video_path, df_one_class.iloc[i])
                count += 1
            except Exception as e:
                print(f"Error {e}")
                print(df_one_class[i])
                wrong_video_list.append(df_one_class[i])


def split_audio(trimed_video_base_path, wav_save_base_path, df_one_video):
    """extract the .wav file from one video"""
    video_name, start_time, category, split = df_one_video[0], df_one_video[1], df_one_video[2], df_one_video[3]
    trimed_video_path = os.path.join(trimed_video_base_path, split, category, video_name + ".mp4")
    wav_save_path = os.path.join(wav_save_base_path, split, category, video_name + ".wav")
    if not os.path.exists(os.path.join(wav_save_base_path, split, category)):
        os.makedirs(os.path.join(wav_save_base_path, split, category))
    video = VideoFileClip(trimed_video_path)
    audio = video.audio
    audio.write_audiofile(wav_save_path, fps=16000)

def extract_audio_wav(root_path="../avsbench_data/Single-source/s4_data",\
                   csv_path="../avsbench_data/Single-source/s4_meta_data.csv"):
    """extract the .wav files for videos"""
    wav_save_base_path = os.path.join(root_path, "audio_wav")
    trimed_video_path = os.path.join(root_path, "raw_videos")
    anno_path = csv_path
    df_anno = pd.read_csv(anno_path, sep=',')
    wrong_video_list = []
    count = 0
    for item in AVSBenchDataset:
        print(f"processing for [{item}]...")
        df_one_class = df_anno[df_anno['category'] == item]
        for i in tqdm(range(len(df_one_class))):
            try:
                split_audio(trimed_video_path, wav_save_base_path, df_one_class.iloc[i])
                count += 1
            except Exception as e:
                print(f"Error {e}")
                print(df_one_class[i])
                wrong_video_list.append(df_one_class[i])
    print('wrong_list: ', wrong_video_list)
    print('#wrong_list: ', len(wrong_video_list))


def extract_one_log_mel(wav_base_path, lm_save_base_path, df_one_video):
    """extract the .wav file from one video and save to .pkl"""
    video_name, start_time, category, split = df_one_video[0], df_one_video[1], df_one_video[2], df_one_video[3]
    wav_path = os.path.join(wav_base_path, split, category, video_name + ".wav")
    lm_save_path = os.path.join(lm_save_base_path, split, category, video_name + ".pkl")
    if not os.path.exists(os.path.join(lm_save_base_path, split, category)):
        os.makedirs(os.path.join(lm_save_base_path, split, category))

    log_mel_tensor = vggish_input.wavfile_to_examples(wav_path)
    wrong_list = []
    if log_mel_tensor.shape[0] != 5:
        wrong_list.append(df_one_video)
        print('start time: ', start_time)
        print('video_name: ', video_name)
        print('lm.shape: ', log_mel_tensor.shape)
        N_SECONDS, CHANNEL, N_BINS, N_BANDS = log_mel_tensor.shape
        new_lm_tensor = torch.zeros(5, CHANNEL, N_BINS, N_BANDS)
        new_lm_tensor[:N_SECONDS] = log_mel_tensor
        new_lm_tensor[N_SECONDS:] = log_mel_tensor[-1].repeat(5-N_SECONDS, 1, 1, 1)
        log_mel_tensor = new_lm_tensor

    with open(lm_save_path, "wb") as fw:
        pickle.dump(log_mel_tensor, fw)
    return wrong_list

def extract_audio_log_mel(root_path="../avsbench_data/Single-source/s4_data",\
                   csv_path="../avsbench_data/Single-source/s4_meta_data.csv"):
    """extract and save the log_mel map for each .wav"""
    wav_path = os.path.join(root_path, "audio_wav")
    anno_path = csv_path
    lm_save_base_path= os.path.join(root_path, "audio_log_mel")
    df_anno = pd.read_csv(anno_path, sep=',')
    wrong_video_list = []
    video_no5s_list = []
    count = 0
    for item in AVSBenchDataset:
        print(f"processing for [{item}]...")
        df_one_class = df_anno[df_anno['category'] == item]
        for i in tqdm(range(len(df_one_class))):
            try:
                wrong_item = extract_one_log_mel(wav_path, lm_save_base_path, df_one_class.iloc[i])
                video_no5s_list.extend(wrong_item)
                count += 1
            except Exception as e:
                print(f"Error {e}")
                print(df_one_class[i])
                wrong_video_list.append(df_one_class[i])
    print('wrong_list: ', wrong_video_list)
    print('#wrong_list: ', len(wrong_video_list))
    print('video_no5s_list: ', video_no5s_list)
    print('#video_no5s_list: ', len(video_no5s_list))





if __name__ == "__main__":
    extract_frames()
    extract_audio_wav()
    extract_audio_log_mel()


