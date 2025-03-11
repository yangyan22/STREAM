import os
import json
import re
import numpy as np
from PIL import Image
import torch.utils.data as data
from transformers import BertTokenizer, AutoImageProcessor
import torch

class FieldParser:
    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(args.vision_model)


    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values[0] 

    # from https://github.com/cuhksz-nlp/R2Gen/blob/main/modules/tokenizers.py
    def clean_report(self, report):
        # clean Iu-xray reports
        if self.dataset == "iu_xray":
            report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                            replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        # clean MIMIC-CXR reports
        else:
            report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
                .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
                .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
                .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
                .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
                .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                                .replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .' 
        
        return report

    def parse(self, features):
      
        to_return = {'id': features['id']}
        report = features.get("token", "")
        report = self.clean_report(report)
        to_return['input_text'] = report
        p_report = features.get("p_report", "")
        to_return["p_report"] = p_report
      
        images = []
         # current study reading
        for image_path in features['image_paths'][:3]:
            with Image.open(os.path.join(self.args.base_dir, image_path)).convert("RGB") as pil:
                image = self._parse_image(pil)
                images.append(image)
        
        if len(images) < 3:
            while len(images) < 3:
                images.append(images[0])  # padding
        
        # temporal study reading
        temp_images1 = []
        temp_images2 = []
        temporal = features["temp"]
        for i in range(min(2, len(temporal))):
            temp_images = []
            current_temporal = temporal[i]
            if current_temporal != 'no_view_found': 
                for image_path in current_temporal[:3]:
                    with Image.open(os.path.join(self.args.base_dir, image_path)).convert("RGB") as pil:
                        image = self._parse_image(pil)
                    temp_images.append(image)
                
                if len(temp_images) < 3:
                    while len(temp_images) < 3:
                        temp_images.append(temp_images[0])    # padding

            if i == 0:
                temp_images1 = temp_images
            else:
                temp_images2 = temp_images

        # padding
        if not temp_images1:   
            temp_images1 = images 
        if not temp_images2:
            temp_images2 = images 

        to_return["images"] =   torch.stack(images)    
        to_return["temp_images1"] =  torch.stack( temp_images1)  
        to_return["temp_images2"] = torch.stack( temp_images2)  
         
        text = ""
        flag = {}
        if  features["retriver"] != []:
            for i, item in enumerate(features["retriver"][0]):
                if item != "" and item not in flag:
                    flag[item] = 1
                    text += regions[i] + ": " + item+ " "
        else:
            text = "no findings"
        to_return["retriver_text"] = text
      
        return to_return

    def transform_with_parse(self, inputs):
        return self.parse(inputs)


class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.meta = json.load(open(args.annotation, 'r'))
        self.meta = self.meta[split]
        self.parser = FieldParser(args)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index])


def create_datasets(args):
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'val')
    test_dataset = ParseDataset(args, 'test')
    return train_dataset, dev_dataset, test_dataset


regions = [
    "right lung",               # 0
    "right upper lung zone",    # 1
    "right mid lung zone",      # 2
    "right lower lung zone",    # 3
    "right hilar structures",   # 4
    "right apical zone",        # 5
    "right costophrenic angle", # 6
    "right hemidiaphragm",      # 7
    "left lung",                # 8
    "left upper lung zone",     # 9
    "left mid lung zone",       # 10
    "left lower lung zone",     # 11
    "left hilar structures",    # 12
    "left apical zone",         # 13
    "left costophrenic angle",  # 14
    "left hemidiaphragm",       # 15
    "trachea",                  # 16
    "spine",                    # 17
    "right clavicle",           # 18
    "left clavicle",            # 19
    "aortic arch",              # 20
    "mediastinum",              # 21
    "upper mediastinum",        # 22
    "svc",                      # 23
    "cardiac silhouette",       # 24
    "cavoatrial junction",      # 25
    "right atrium",             # 26
    "carina",                   # 27
    "abdomen"                   # 28
]
 