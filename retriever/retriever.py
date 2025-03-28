import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from medclip import MedCLIPModel, MedCLIPVisionModelViT
from transformers import AutoProcessor
import faiss
import csv
import json
import torch
from object_detector.object_detector import ObjectDetector
from PIL import Image, ImageFile
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import numpy as np
import ast

ImageFile.LOAD_TRUNCATED_IMAGES = True
faiss.omp_set_num_threads(4)
use_history = True
a = 0
b = -1
# [1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 22, 24, 25, 27, 29] - 1
# [1: 1.569, 9:2.134, 29:0.366]
num=28
base_url = "/data2/yinanwangbei/med-cxr/MIMIC_original_images/files/"

regions = [
    "right lung", "right upper lung zone", "right mid lung zone", "right lower lung zone",
    "right hilar structures", "right apical zone", "right costophrenic angle", "right hemidiaphragm",
    "left lung", "left upper lung zone", "left mid lung zone", "left lower lung zone",
    "left hilar structures", "left apical zone", "left costophrenic angle", "left hemidiaphragm",
    "trachea", "spine", "right clavicle", "left clavicle", "aortic arch", "mediastinum",
    "upper mediastinum", "svc", "cardiac silhouette", "cavoatrial junction", "right atrium",
    "carina", "abdomen"
]

def extract_id(filepath):
    # 获取文件名(包含扩展名)
    filename = os.path.basename(filepath)
    # 获取文件名(不含扩展名)
    name_without_ext = os.path.splitext(filename)[0]
    return name_without_ext

def extract_vectors_from_index(index):
    vectors = []
    for i in range(index.ntotal):
        vector = index.reconstruct(i)
        vectors.append(vector)
    return np.array(vectors).astype('float32')

index_list = []
original_vector = []
for i in range(0, num):
    # print(i)
    index_list.append([])
    original_vector.append([])
index = faiss.read_index(f"./filter_faiss/faiss_index_{num+1}.index")
print("读取索引结束")
res = faiss.StandardGpuResources()
print("资源获取结束")

data = extract_vectors_from_index(index)

# n_clusters = 1200

# kmeans = faiss.Kmeans(d=data.shape[1], k=n_clusters, niter=50, verbose=True, gpu=True)
# kmeans.train(data)
original_vector.append(data)

# quantizer = faiss.IndexFlatIP(data.shape[1])
# ivf_index = faiss.IndexIVFFlat(quantizer, data.shape[1], n_clusters, faiss.METRIC_INNER_PRODUCT)
# ivf_index.train(data)
# ivf_index.add(data)
print("索引训练和添加数据结束")

gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
print("索引构建结束")
index_list.append(gpu_index)

additional_info_list = []
for i in range(0, num):
    additional_info_list.append([])
# print(i)
additional_info = []
with open(f"./filter_faiss/{num+1}.csv", "r", encoding='utf-8-sig') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        additional_info.append(row)
additional_info_list.append(additional_info)

# 加载medclip
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained('microsoft/swin-tiny-patch4-window7-224')
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
state_dict = torch.load('./pytorch_model.bin')

if 'text_model.model.embeddings.position_ids' in state_dict:
    print('Removing unexpected key: text_model.model.embeddings.position_ids')
    del state_dict['text_model.model.embeddings.position_ids']

model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model = torch.nn.DataParallel(model)

with open("/data/SJM/mimic_views_ranked_pr_temp.json", "r") as f:
    data = json.load(f)

train_data = data["train"]
test_data = data["test"]
val_data = data["val"]

object_detector = ObjectDetector(return_feature_vectors=False)
checkpoint = "/data/yinanwangbei/medcxr/retriever/object_detector/run_14/weights/val_loss_12.548_epoch_18.pth"
state_dict = torch.load(checkpoint, map_location="cpu")
object_detector.load_state_dict(state_dict)
object_detector.to(device)
object_detector = torch.nn.DataParallel(object_detector)

IMAGE_INPUT_SIZE = 512
mean = 0.471
std = 0.302

val_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
        A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

apset = {}

for item in train_data:
    apset[item["image_path"][0]] = item["view"]
for item in test_data:
    apset[item["image_path"][0]] = item["view"]
for item in val_data:
    apset[item["image_path"][0]] = item["view"]

with open("id_view.json", "r") as f:
    images_view_dict = json.load(f)

# cv2读取图片
def read_image_cv2(image_path):
    image = cv2.imread(base_url + image_path, cv2.IMREAD_UNCHANGED)
    original_height, original_width = image.shape[:2]
    return image, original_height, original_width

# Image读取图片
def read_image_IMAGE(image_path):
    new_image = Image.open(base_url + image_path).convert("RGB")
    return new_image

def reverse_transform_bbox(bbox, original_height, original_width):
    pad_height = max(0, IMAGE_INPUT_SIZE - original_height)
    pad_width = max(0, IMAGE_INPUT_SIZE - original_width)

    bbox = bbox.copy()

    if pad_height > 0:
        bbox[1] -= pad_height // 2
        bbox[3] -= pad_height // 2

    if pad_width > 0:
        bbox[0] -= pad_width // 2
        bbox[2] -= pad_width // 2

    scale = min(IMAGE_INPUT_SIZE / original_width, IMAGE_INPUT_SIZE / original_height)
    bbox /= scale

    return bbox

def get_images(item):
    image_path = item["image_path"][0]
    now_image, original_height, original_width = read_image_cv2(image_path)
    history_image1, history_image_height1, history_image_width1 = None, None, None
    if len(item["temp"]) > 0:
        temp1 = item["temp"][0]
        for history_image_path in temp1:
            if images_view_dict[extract_id(history_image_path)] in ["AP", "PA"]:
                history_image1, history_image_height1, history_image_width1 = read_image_cv2(history_image_path)
                break
    history_image2, history_image_height2, history_image_width2 = None, None, None
    if len(item["temp"]) > 1 and history_image1 is not None:
        temp2 = item["temp"][1]
        for history_image_path in temp2:
            if images_view_dict[extract_id(history_image_path)] in ["AP", "PA"]:
                history_image2, history_image_height2, history_image_width2 = read_image_cv2(history_image_path)
                break

    return [
        now_image, original_height, original_width,
        history_image1, history_image_height1, history_image_width1,
        history_image2, history_image_height2, history_image_width2
    ]

# 
def encode_image(data):
    now_image, original_height, original_width, history_image1, history_image_height1, history_image_width1, history_image2, history_image_height2, history_image_width2 = data

    image_list = [val_transforms(image=now_image)["image"].to(device)]
    if history_image1 is not None:
        image_list.append(val_transforms(image=history_image1)["image"].to(device))
    if history_image2 is not None:
        image_list.append(val_transforms(image=history_image2)["image"].to(device))

    loss_dict, detections, class_detected = object_detector(torch.stack(image_list), None)
    detections = detections["top_region_boxes"].cpu().numpy()
    original_bboxes = []
    original_bboxes.append([reverse_transform_bbox(bbox, original_height, original_width) for bbox in detections[0]])
    if history_image_height1 is not None:
        original_bboxes.append([reverse_transform_bbox(bbox, history_image_height1, history_image_width1) for bbox in detections[1]])

    if history_image_height2 is not None:
        original_bboxes.append([reverse_transform_bbox(bbox, history_image_height2, history_image_width2) for bbox in detections[2]])

    return original_bboxes, class_detected

def medclip_encode(bbox, image_path):
    image_path = base_url + image_path
    new_image = Image.open(image_path).convert("RGB")
    x_min, y_min, x_max, y_max = bbox
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    cropped_img = new_image.crop((x_min, y_min, x_max, y_max))
    inputs = processor(images=cropped_img, return_tensors="pt")
    inputs = inputs.to(device)
    outputs = model(**inputs)["img_embeds"]
    return outputs

def get_initial_embedding(bbox, image_path):
    outputs = medclip_encode(bbox, image_path)
    embedding = outputs.cpu().detach().numpy().reshape(1, -1)
    faiss.normalize_L2(embedding)
    return embedding

def search_index(index, embedding, top_k):
    index.nprobe = 2
    D, I = index.search(embedding, top_k)
    return I[0], D[0]

def build_new_index(original_indices, loc):
    new_embeddings = []
    original_to_new_map = {}
    for idx in original_indices:
        idx = int(idx)
        history = ast.literal_eval(additional_info_list[loc][idx]["history"])
        if history:
            new_embeddings.append(original_vector[loc][history[0]].reshape(1, -1))
            original_to_new_map[len(new_embeddings) - 1] = idx
    if not new_embeddings:
        return None, None
    new_index = faiss.IndexFlatIP(512)
    new_index.add(np.concatenate(new_embeddings, axis=0))
    return new_index, original_to_new_map

def get_embedding_for_bbox_step(item, original_bboxes, class_detected, step, loc):
    if len(original_bboxes) <= step or not class_detected[step][loc]:
        return None
    bbox = original_bboxes[step][loc]
    for history_image_path in item["temp"][step-1]:
        print(history_image_path)
        if images_view_dict[extract_id(history_image_path)] in ["AP", "PA"]:
            return get_initial_embedding(bbox, history_image_path)
    return None

def filter_with_bbox(item, original_bboxes, class_detected, loc, new_index, original_to_new_map, step):
    embedding = get_embedding_for_bbox_step(item, original_bboxes, class_detected, step, loc)
    if embedding is None:
        return None, None, None
    I, D = search_index(new_index, embedding, 10 if step == 1 else 1)

    if max(D) < 0.992:
        return None, None, None

    filtered_info = []
    new_embeddings = []
    original_to_new_map2 = {}
    for idx in I:
        if int(idx) == -1:
            continue
        original_idx = original_to_new_map[int(idx)]
        history = ast.literal_eval(additional_info_list[loc][original_idx]["history"])
        if step == 1 and len(history) >= 2:
            new_embeddings.append(original_vector[loc][history[1]].reshape(1, -1))
            original_to_new_map2[len(new_embeddings) - 1] = original_idx
            filtered_info.append(additional_info_list[loc][original_idx])
        elif step == 2:
            filtered_info.append(additional_info_list[loc][original_idx])
    if step == 1 and len(new_embeddings) > 0:
        new_index = faiss.IndexFlatIP(512)
        new_index.add(np.concatenate(new_embeddings, axis=0))
    return new_index, filtered_info, original_to_new_map2

# def check_abnormality(indices, loc, path=None):
#     abnormal = 0
#     flag = False
#     report = ""
#     for idx in indices:
#         info = additional_info_list[loc][idx]
#         if info["path"] == path:
#             continue
#         if info["Abnormal"] == "True":
#             abnormal += 1
#             if not flag:
#                 abnormal_report = info["report"]
#                 flag = True
        
#     if abnormal > (len(indices) - abnormal):
#         return report
#     else:
#         return ""
def check_abnormality(indices, loc, path=None):
    abnormal_count = 0
    normal_count = 0
    first_abnormal_report = ""
    first_normal_report = ""
    
    for idx in indices:
        info = additional_info_list[loc][idx]
        # print(info)
        if info["path"] == path:
            continue
        
        if info["Abnormal"] == "True":
            abnormal_count += 1
            if not first_abnormal_report:
                first_abnormal_report = info["report"]
        else:
            normal_count += 1
            if not first_normal_report:
                first_normal_report = info["report"]
    
    if normal_count > abnormal_count:
        return first_normal_report
    else:
        return first_abnormal_report

def get_text(item, original_bboxes, class_detected, loc):
    if not class_detected[0][loc]:
        return ""

    embedding = get_initial_embedding(original_bboxes[0][loc], item["image_path"][0])
    initial_indices, _ = search_index(index_list[loc], embedding, 20)
     
    # return  check_abnormality(initial_indices, loc, item["image_path"][0])
    ##############################################################
    new_index, original_to_new_map = build_new_index(initial_indices, loc)
    if not new_index or len(class_detected) == 1:
        return check_abnormality(initial_indices, loc, item["image_path"][0])
    # print("haha")
    new_index, filtered_info, original_to_new_map2 = filter_with_bbox(item, original_bboxes, class_detected, loc, new_index, original_to_new_map, 1)
    # return check_abnormality(original_to_new_map.values(), loc, item["image_path"][0])

    if not new_index or not filtered_info or len(class_detected) == 2:
        return check_abnormality(original_to_new_map.values(), loc, item["image_path"][0])

    new_index, filtered_info, _ = filter_with_bbox(item, original_bboxes, class_detected, loc, new_index, original_to_new_map2, 2)
    if not filtered_info or filtered_info[0]["Abnormal"] == "False":
        return ""
    else:
        return filtered_info[0]["report"]

def operate(dataset):
    with torch.no_grad():
        object_detector.eval()
        for item in tqdm(dataset, desc="Processing dataset"):
            if not use_history:
                item["temp"] = []
            else:
                # print(item)
                for image_list in item["temp"]:
                    if image_list == "no_view_found":
                        item["temp"] = []
            item["retriver"] = []
            if item["view"] not in ["AP", "PA"]:
                if "retriver" not in item:
                    # for i in range(29):
                    item["retriver"].append("")
                continue
            data = get_images(item)
            original_bboxes, class_detected = encode_image(data)
            # for i in range(len(original_bboxes[0])):
            text = get_text(item, original_bboxes, class_detected, num)
            # print(text)
            item["retriver"].append(text)
    return dataset

train_data = operate(train_data[a:])
with open(f"./history_retrieve/train_multiview_{a}_{b}_{num}.json", "w") as f:
    json.dump(train_data, f)