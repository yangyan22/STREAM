# STREAM
Official implementation of STREAM: Spatio-Temporal and Retrieval-Augmented Modelling for Chest X-Ray Report Generation

The paper has been submitted to IEEE Transactions on Medical Imaging. 

We will open the related codes and the constructed knowledge bank! 

# Abstract
Chest X-ray report generation has attracted increasing research attention. However, most existing methods neglect the temporal information and typically generate reports conditioned on a fixed number of images. In this paper, we propose STREAM: Spatio-Temporal and REtrieval-Augmented Modelling for automatic chest X-ray report generation. It mimics clinical diagnosis by integrating current and historical studies to interpret the present condition (temporal), with each study containing images from multi-views (spatial). Concretely, our STREAM is built upon an encoder-decoder architecture, utilizing a large language model (LLM) as the decoder. Overall, spatio-temporal visual dynamics are packed as visual prompts and regional semantic entities are retrieved as textual prompts. First, a token packer is proposed to capture condensed spatio-temporal visual dynamics, enabling the flexible fusion of images from current and historical studies. Second, to augment the generation with existing knowledge and regional details, a progressive semantic retriever is proposed to retrieve semantic entities from a preconstructed knowledge bank as heuristic text prompts. The knowledge bank is constructed to encapsulate anatomical chest X-ray knowledge into structured entities, each linked to a specific chest region. Extensive experiments on public datasets have shown the state-of-the-art performance of our method. Related codes and the knowledge bank will be released.

# Dataset and Weight
you can download the MIMIC-CXR images from https://physionet.org/content/mimic-cxr-jpg/2.0.0/

you can download the IU X-Ray images from https://openi.nlm.nih.gov/

# Environment and Install
Python = 3.10.13 and torch = 2.1.1+cu118

1. install packages
   
pip install -r requirements.txt

2. download pretrained models

download pretrained swin-base-patch4-window7-224, TinyLlama-1.1B-Chat-v1.0, BiomedVLP-CXR-BERT-specialized from hugging face.

3. Coming soon


