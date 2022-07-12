## Audio-Visual Segmentation
[[Project Page]]()  [[Arxiv]](https://arxiv.org/abs/2207.05042)

This repository provides the PyTorch implementation for the ECCV2022 paper "Audio-Visual Segmentation".
This paper proposes the audio-visual segmentation problem and the AVSBench dataset accordingly.


### News
- We are preparing the AVSBench-v2 which is 10 times larger than AVSBench.

---

### Data preparation
- AVSBench dataset

The csv file that contains the video ids for downloading the raw YouTube videos and the annoated ground truth segmentation maps can be downloaded from [here](https://drive.google.com/drive/folders/1wKFKymVYn6rNkNE_7xV6Bm-9PfCAIKdT?usp=sharing). 


We also provide some scripts to process the raw video data and extract the frames/mel-spectrogram features.
```
cd preprocess_scripts
python preprocess_s4.py # for Single-source set
python preprocess_ms3.py # for Multi-sources set
```
The data should be placed to the directory `avsbench_data`.

Feel free to contact us if you have any questions when downloading and processing the video data, you could send an email to zhoujxhfut@gmail.com.


- pretrained backbone

The pretrained ResNet50/PVT-v2-b5 (vision) and VGGish (audio) backbones can be downloaded from [here](https://drive.google.com/drive/folders/1386rcFHJ1QEQQMF6bV1rXJTzy8v26RTV?usp=sharing) and placed to the directory `pretrained_backbones`.

**Notice:** please update the path of data and pretrained backbone in avs_s4/config.py and avs_ms3/config.py accordingly.

---

### S4 setting
- Train AVS Model
```
cd avs_scripts/avs_s4
bash train.sh
```

- Test AVS Model
```
cd avs_scripts/avs_s4
bash test.sh
```
---
### MS3 setting
- Train AVS Model
```
cd avs_scripts/avs_ms3
bash train.sh
```

- Test AVS Model
```
cd avs_scripts/avs_ms3
bash test.sh
```

### Citation
If you use this dataset or code, please consider cite:
```
@inproceedings{zhou2022avs,
  title     = {Audio-Visual Segmentation},
  author    = {Zhou, Jinxing and Wang, Jianyuan and Zhang, Jiayi and Sun, Weixuan and Zhang, Jing and Birchfield, Stan and Guo, Dan and Kong, Lingpeng and Wang, Meng and Zhong, Yiran},
  booktitle = {European Conference on Computer Vision},
  year      = {2022}
}
```


### License
This project is released under the Apache 2.0 license as found in the [LICENSE](./LICENSE) file.
