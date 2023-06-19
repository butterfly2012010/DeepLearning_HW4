"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""

import config
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)

ImageFile.LOAD_TRUNCATED_IMAGES = True    
    
class ADE20KDataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        # anchors,
        image_size=416,
        # S=[13, 26, 52],
        # C=20,
        transform=None,
    ):
        self.annotations = pd.read_csv(csv_file, sep=' ', header=None, dtype=str)
        self.img_dir = img_dir
        self.mask_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        # self.S = S
        # self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        # self.num_anchors = self.anchors.shape[0]
        # self.num_anchors_per_scale = self.num_anchors // 3
        # self.C = C
        # self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, 'ADE_val_'+self.annotations.iloc[idx, 0]+'.jpg')
        mask_path = os.path.join(self.mask_dir, ('ADE_val_'+self.annotations.iloc[idx, 0]+'_seg.jpg').replace('.jpg', '.png'))

        # image = np.array(Image.open(image_path).convert('RGB').resize(self.image_size, self.image_size))  # (width, height)
        image = np.array(Image.open(image_path).resize((self.image_size, self.image_size)))  # (width, height)
        mask = np.array(Image.open(mask_path).resize((self.image_size, self.image_size)))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # # Normalize image
        # image = image / 255.0

        return image, mask

# 設定資料集路徑和轉換
# root_dir = '/path/to/ADE20K_dataset'
# transform = transforms.Compose([
#     transforms.Resize((416, 416)),  # 調整圖像大小
#     transforms.ToTensor()  # 轉換為Tensor格式
# ])

# # 創建訓練資料集的dataloader
# train_dataset = ADE20KDataset(root_dir, split='training', transform=transform)
# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# # 創建驗證資料集的dataloader
# val_dataset = ADE20KDataset(root_dir, split='validation', transform=transform)
# val_dataloader = DataLoader(val_dataset, batch_size=32)

# # 使用dataloader進行訓練和驗證迴圈
# for images, masks in train_dataloader:
#     # 在這裡執行模型訓練

# for images, masks in val_dataloader:
#     # 在這裡執行模型驗證

    

class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size=416,
        S=[13, 26, 52],
        C=20,
        transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return image, tuple(targets)


def test():
    anchors = config.ANCHORS

    transform = config.test_transforms

    dataset = YOLODataset(
        "PASCAL_VOC/train.csv",
        "PASCAL_VOC/images/",
        "PASCAL_VOC/labels/",
        S=[13, 26, 52],
        anchors=anchors,
        transform=transform,
    )
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for x, y in loader:
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            print(anchor.shape)
            print(y[i].shape)
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        print(boxes)
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)


if __name__ == "__main__":
#     ### transform PASCAL_VOC2007 to YOLO txt file
#     import os
#     import xml.etree.ElementTree as ET
#     from PIL import Image

#     # 資料集路徑
#     dataset_path = "./'VOC 2007'/"
    
#     # 分別處理訓練集、驗證集和測試集
#     sets = ["train", "val", "test"]
    
#     # 清單檔案路徑
#     list_files = ["train.txt", "val.txt", "test.txt"]

#     # YOLO格式的類別索引
#     classes = config.PASCAL_CLASSES  # 替換成你的類別列表

#     # YOLO格式的標籤檔案儲存目錄
#     output_dir = "./'VOC 2007'/labels"

#     # 處理每個標註檔案的函式
#     def process_annotation(annotation_path, image_width, image_height):
#         # 讀取XML標註檔案
#         tree = ET.parse(annotation_path)
#         root = tree.getroot()

#         # 解析標註資訊
#         objects = root.findall("object")
#         for obj in objects:
#             # 取得物件類別
#             class_name = obj.find("name").text

#             # 取得物件邊界框
#             bbox = obj.find("bndbox")
#             xmin = float(bbox.find("xmin").text)
#             ymin = float(bbox.find("ymin").text)
#             xmax = float(bbox.find("xmax").text)
#             ymax = float(bbox.find("ymax").text)

#             # 計算物件中心座標及相對寬高
#             x_center = (xmin + xmax) / (2 * image_width)
#             y_center = (ymin + ymax) / (2 * image_height)
#             width = (xmax - xmin) / image_width
#             height = (ymax - ymin) / image_height

#             # 取得類別索引
#             class_index = classes.index(class_name)

#             # 儲存標籤至YOLO格式的檔案
#             image_filename = os.path.splitext(os.path.basename(annotation_path))[0] + ".txt"
#             output_path = os.path.join(output_dir, image_filename)

#             with open(output_path, "a") as file:
#                 file.write(f"{class_index} {x_center} {y_center} {width} {height}\n")


#     # 處理每個子集
#     for subset, list_file in zip(sets, list_files):
#         image_dir = os.path.join(dataset_path, "VOCdevkit/VOC2007/JPEGImages")
#         annotation_dir = os.path.join(dataset_path, "VOCdevkit/VOC2007/Annotations")
#         output_dir = os.path.join("./'VOC 2007'/labels", subset)

#         # 建立標籤儲存目錄
#         os.makedirs(output_dir, exist_ok=True)

#         # 讀取清單檔案
#         with open(os.path.join(dataset_path, list_file), "r") as f:
#             image_files = f.read().splitlines()

#         for image_file in image_files:
#             # 讀取圖像檔案
#             image_path = os.path.join(image_dir, image_file + ".jpg")
#             image = Image.open(image_path)
#             image_width, image_height = image.size

#             # 取得對應的標註檔案
#             annotation_file = image_file + ".xml"
#             annotation_path = os.path.join(annotation_dir, annotation_file)

#             # 處理標註檔案
#             process_annotation(annotation_path, image_width, image_height)
#     # 讀取資料集目錄下的圖像檔案和標註檔案
#     image_dir = os.path.join(dataset_path, "JPEGImages")
#     annotation_dir = os.path.join(dataset_path, "Annotations")
#     image_files = os.listdir(image_dir)

#     for image_file in image_files:
#         # 讀取圖像檔案
#         image_path = os.path.join(image_dir, image_file)
#         image = Image.open(image_path)
#         image_width, image_height = image.size

#         # 取得對應的標註檔案
#         annotation_file = os.path.splitext(image_file)[0] + ".xml"
#         annotation_path = os.path.join(annotation_dir, annotation_file)

#         # 處理標註檔案
#         process_annotation(annotation_path, image_width, image_height)
    
    test()
