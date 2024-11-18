## Audio-Visual Segmentation


This repository provides the PyTorch implementation for the **ECCV2022** paper "Audio-Visual Segmentation". This paper proposes the audio-visual segmentation (AVS) problem and the AVSBench dataset accordingly.  [[Project Page]](https://opennlplab.github.io/AVSBench/)  [[arXiv]](https://arxiv.org/abs/2207.05042) 

**Audio-Visual Semantic Segmentation** Recently, we expanded the AVS task to include one more challenging setting, i.e., the fully-supervised audio-visual semantic segmentation (AVSS) that requires generating semantic masks of the sounding objects. Accordingly, we collected a new AVSBench-semantic dataset. Please refer to our [arXiv paper](https://arxiv.org/abs/2301.13190) ''Audio-Visual Segmentation with Semantics'' for more details. [[Online Benchmark]](http://www.avlbench.opennlplab.cn/dataset/avsbench)

---


### Updates
- (2024.10.15) Our extension work on Audio-Visual Semantic Segmentation has been accepted by **IJCV-2024**. The online version is accessible via this [link](https://link.springer.com/article/10.1007/s11263-024-02261-x). We look forward to witnessing further outstanding contributions in this practical and challenging field!
- (2023.1.31) The AVSBench-semantic dataset has been released, you can download it from our official [benchmark website](http://www.avlbench.opennlplab.cn/download). Please refer to our [arXiv paper](https://arxiv.org/abs/2301.13190) for more details of this dataset. 
- (2022.10.18) We have completed the collection and annotation of AVSBench-semantic. Compared to the original AVSBench dataset, it contains ~7k more multi-source videos covering 70 categories, and the ground truths are provided in the form of multi-label semantic maps (labels of original AVSBench dataset are also updated). We will release it as soon as possible.
- (2022.7.13) We are preparing the AVSBench-semantic dataset that will pay more attention to multi-source situations and provide semantic annotations.

---

### Data preparation
#### 1. AVSBench dataset

The AVSBench dataset is first proposed in our [ECCV paper](https://arxiv.org/abs/2207.05042). It contains a Single-source and a Multi-sources subset. Ground truths of these two subsets are binary segmentation maps indicating pixels of the sounding objects. Recently, we collected a new Semantic-labels subset that provides semantic segmentation maps as labels. We add it to the original AVSBench dataset as the third subset. For convenience, we denote the original AVSBench dataset as **AVSBench-object**, and the newly added Semantic-labels subset as **AVSBench-semantic**.

AVSBench-object is used for the Single Sound Source Segmentation (S4) and Multiple Sound Source Segmentation (MS3),  while AVSBench-semantic is used for the Audio-Visual Semantic Segmentation (AVSS).

**The updated AVSBench dataset is available at [http://www.avlbench.opennlplab.cn/download](http://www.avlbench.opennlplab.cn/download).** You may request the dataset by mail at [opennlplab@gmail.com.](mailto:opennlplab@gmail.com). We will reply as soon as we receive the application.

These downloaded data should be placed in the directory `avsbench_data`.

#### 2. pretrained backbones

The pretrained ResNet50/PVT-v2-b5 (vision) and VGGish (audio) backbones can be downloaded from [here](https://drive.google.com/drive/folders/1386rcFHJ1QEQQMF6bV1rXJTzy8v26RTV?usp=sharing) and placed to the directory `pretrained_backbones`.

**Notice:** Please update the path of data and pretrained backbone in `avs_s4/config.py`, `avs_ms3/config.py`, and `avss/config.py` accordingly.

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
---
### AVSS setting
- Train AVS model
```
cd avs_scripts/avss
bash train.sh
```

- Test AVS model
```
cd avs_scripts/avss
bash test.sh
```

Notably, the AVSS setting can be viewed as an independent task by the research community, i.e., the audio-visual semantic segmentation task. The pretrained AVSS models are available at [here](https://drive.google.com/drive/folders/1faSG_Hs2D2PkYeXLyRqlo-cjZ-DmggKB?usp=sharing).

---

### Citation

If you use this dataset or code, please consider citing the following papers:
```
@inproceedings{zhou2022avs,
  title     = {Audio-Visual Segmentation},
  author    = {Zhou, Jinxing and Wang, Jianyuan and Zhang, Jiayi and Sun, Weixuan and Zhang, Jing and Birchfield, Stan and Guo, Dan and Kong, Lingpeng and Wang, Meng and Zhong, Yiran},
  booktitle = {European Conference on Computer Vision},
  year      = {2022}
}

@article{zhou2024avss,
  title={Audio-visual segmentation with semantics},
  author={Zhou, Jinxing and Shen, Xuyang and Wang, Jianyuan and Zhang, Jiayi and Sun, Weixuan and Zhang, Jing and Birchfield, Stan and Guo, Dan and Kong, Lingpeng and Wang, Meng and Zhong, Yiran},
  journal={International Journal of Computer Vision},
  pages={1--21},
  year={2024}
}

```


### License
This project is released under the Apache 2.0 license as found in the [LICENSE](./LICENSE) file.
