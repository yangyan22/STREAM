import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"  
from medclip import MedCLIPModel, MedCLIPVisionModelViT
from transformers import AutoProcessor
from PIL import Image, ImageFile
import pandas as pd
import faiss
from PIL import Image
import csv
import numpy as np
import ast
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained('microsoft/swin-tiny-patch4-window7-224')
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
state_dict = torch.load('/data/new/pytorch_model.bin')  
if 'text_model.model.embeddings.position_ids' in state_dict:
    print('Removing unexpected key: text_model.model.embeddings.position_ids')
    del state_dict['text_model.model.embeddings.position_ids']

model.load_state_dict(state_dict, strict=False)
model.to(device)
ImageFile.LOAD_TRUNCATED_IMAGES = True
for iddddd in [1, 5 , 6, 9, 13, 14, 17, 18, 22, 25, 29]:
    with open(f'/data/new/knowledge/bank_{iddddd}.csv', 'r') as file:  
        df  = pd.read_csv(file)
        
        leng = df.shape[0]
        image_embeds = []
        additional_info = []
        for i in range(leng):
            bbox = df.iloc[i]["bbox"]
            bbox_label = df.iloc[i]["bbox_label"] 
            image_path = df.iloc[i]["path"]
            image_path = "/data2/yinanwangbei/med-cxr/MIMIC_original_images/files/" + str(image_path)
            
            image = Image.open(image_path).convert("RGB")
            bbox = ast.literal_eval(bbox)
            report = df.iloc[i]["report"]
            # print(bbox[0])
            cropped_img = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            inputs = processor(images=cropped_img, return_tensors="pt")
            outputs = model(**inputs)
            a = outputs["img_embeds"].cpu().detach().numpy()
            image_embeds.append(a)
            additional_info.append({
                "path": image_path,
                "report": report,
                "bbox": bbox,
                "bbox_label": bbox_label,
            })
            print(image_path)
                

    with open(f"additional_info_{iddddd}.csv", "w", newline='') as csvfile:   
            fieldnames = ["path", "report", "bbox", "bbox_label"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for info in additional_info: 
                writer.writerow(info)

    dim = 512  
    image_embeds = np.squeeze(image_embeds, axis=1)
    faiss.normalize_L2(image_embeds)

    index = faiss.IndexFlatIP(dim)
    index.add(image_embeds)
    faiss.write_index(index, f"faiss_index_{iddddd}.index") 

    index = faiss.read_index(f"faiss_index_{iddddd}.index") 
    additional_info = []
    with open(f"additional_info_{iddddd}.csv", "r") as csvfile: 
        reader = csv.DictReader(csvfile)
        for row in reader:
            additional_info.append(row)

    # Query Examples
    query_embeds = image_embeds[0:1]  
    faiss.normalize_L2(query_embeds)
    D, I = index.search(query_embeds, k=5) 

    print("Index：", I)
    print("Distance：", D)
    for idx in I[0]:
        print("additional info：", additional_info[idx])

