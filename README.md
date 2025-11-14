# Spatio-Temporal and Retrieval-Augmented Modeling for Chest X-Ray Report Generation

## 📋 Abstract
Chest X-ray report generation has attracted increasing research attention. However, most existing methods neglect the temporal information and typically generate reports conditioned on a fixed number of images. In this paper, we propose STREAM: Spatio-Temporal and REtrieval-Augmented Modelling for automatic chest X-ray report generation. It mimics clinical diagnosis by integrating current and historical studies to interpret the present condition (temporal), with each study containing images from multi-views (spatial). Concretely, our STREAM is built upon an encoder-decoder architecture, utilizing a large language model (LLM) as the decoder. Overall, spatio-temporal visual dynamics are packed as visual prompts and regional semantic entities are retrieved as textual prompts. First, a token packer is proposed to capture condensed spatio-temporal visual dynamics, enabling the flexible fusion of images from current and historical studies. Second, to augment the generation with existing knowledge and regional details, a progressive semantic retriever is proposed to retrieve semantic entities from a preconstructed knowledge bank as heuristic text prompts. The knowledge bank is constructed to encapsulate anatomical chest X-ray knowledge into structured entities, each linked to a specific chest region. Extensive experiments on public datasets have shown the state-of-the-art performance of our method. 

![Overall framework of our STREAM](https://github.com/yangyan22/STREAM/blob/main/models/STREAM.png)


## 📊 Datasets
Please download the MIMIC-CXR dataset from https://physionet.org/content/mimic-cxr-jpg/2.0.0/ with your account being "credentialed".
Please download the IU X-Ray dataset from https://openi.nlm.nih.gov/

## 🔧 Environment and Installation
Python = 3.10.13 and torch = 2.1.1+cu118

1. Install packages
   
pip install -r requirements.txt

2. Download pretrained models

download pretrained swin-base-patch4-window7-224, TinyLlama-1.1B-Chat-v1.0, BiomedVLP-CXR-BERT-specialized from hugging face.

3. Put the [evalcap](https://github.com/wang-zhanyu/R2GenGPT) to the main directory

it is for evaluation using machine translation metrics.

## 🎯 Knowledge banks of CXR anotomical regions
 
Link: https://drive.google.com/drive/folders/174I2qsoRvb_yF3xVXWLn55BRj4h-poyn?usp=drive_link

regions = [
    "right lung"               # 1
    "right hilar structures"  # 5
    "right apical zone"       # 6
    "left lung"               # 9
    "left hilar structures"    # 13
    "left apical zone"         # 14
    "trachea"                  # 17
    "spine"                 # 18
    "mediastinum"          # 22
    "cardiac silhouette"       # 25
    "abdomen"                   # 29
]

We construct the Faiss files with the visual encoder of [MedCLIP](https://github.com/RyanWangZf/MedCLIP).

The knowledge bank is constructed of a region label, an abnormality attribute, a description, a temporal connection, and an embedding (The embeddings are included in the Faiss files).

## Training the CXR region detector

Train the CXR region detector to detect anotomical regions with the [Chest ImaGenome dataset](https://physionet.org/content/chest-imagenome/1.0.0/). Related codes can be found at [RGRG](https://github.com/ttanida/rgrg).
The trained detector checkpoint is available at (https://drive.google.com/file/d/1FnunugJvAVmqWgV_pm4HF_ggz8m04hrm/view?usp=drive_link). 

## Usage
1. Construct the JSON file incorporating temporal and multi-view information. The temporal and multi-view information is generated from the "mimic-cxr-2.0.0-metadata.csv" file included in the official dataset.
2. Construct the JSON file with retrieved entity descriptions using the Retriever.
3. Run train.py with the constructed JSON file.

## Acknowledgement
Our codes are partialy based on the codes from [R2GenGPT](https://github.com/wang-zhanyu/R2GenGPT).

 
## ✅ Citation

If you find our method useful in your research, please cite our paper:

```
@ARTICLE{10938723,
  author={Yang, Yan and You, Xiaoxing and Zhang, Ke and Fu, Zhenqi and Wang, Xianyun and Ding, Jiajun and Sun, Jiamei and Yu, Zhou and Huang, Qingming and Han, Weidong and Yu, Jun},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Spatio-Temporal and Retrieval-Augmented Modeling for Chest X-Ray Report Generation}, 
  year={2025},
  volume={44},
  number={7},
  pages={2892-2905}}
```
