import os
import numpy as np
import torch
import pickle

import pandas as pd
import cv2
from tqdm import tqdm
import sys

from mega_core.utils.comm import is_main_process
from mega_core.data.datasets.vid import VIDDataset

class VisDroneDataset(VIDDataset):

    # classes = ['__background__',  # always index 0
    #             'airplane', 'antelope', 'bear', 'bicycle',
    #             'bird', 'bus', 'car', 'cattle',
    #             'dog', 'domestic_cat', 'elephant', 'fox',
    #             'giant_panda', 'hamster', 'horse', 'lion',
    #             'lizard', 'monkey', 'motorcycle', 'rabbit',
    #             'red_panda', 'sheep', 'snake', 'squirrel',
    #             'tiger', 'train', 'turtle', 'watercraft',
    #             'whale', 'zebra']

    classes = [
        {'id': 0, 'name': 'ignored-regions'},
        {'id': 1, 'name': 'pedestrian'},
        {'id': 2, 'name': 'people'},
        {'id': 3, 'name': 'bicycle'},
        {'id': 4, 'name': 'car'},
        {'id': 5, 'name': 'van'},
        {'id': 6, 'name': 'truck'},
        {'id': 7, 'name': 'tricycle'},
        {'id': 8, 'name': 'awning-tricycle'},
        {'id': 9, 'name': 'bus'},
        {'id': 10, 'name': 'motor'},
        {'id': 11, 'name': 'others'},
    ]

    def __init__(self, image_set, data_dir, img_dir, anno_path, img_index, transforms, is_train=True):
        self.det_vid = image_set.split("_")[0]
        self.image_set = image_set
        self.transforms = transforms

        self.data_dir = data_dir
        self.img_dir = img_dir
        self.anno_path = anno_path
        self.img_index = img_index

        self.is_train = is_train

        self._img_dir = os.path.join(self.img_dir, "%s.JPEG")
        self._anno_path = os.path.join(self.anno_path, "%s.xml")

        with open(self.img_index) as f:
            lines = [x.strip().split(" ") for x in f.readlines()]
        if len(lines[0]) == 2:
            self.image_set_index = [x[0] for x in lines]
            self.frame_id = [int(x[1]) for x in lines]
        else:
            self.image_set_index = ["%s/%06d" % (x[0], int(x[2])) for x in lines]
            self.pattern = [x[0] + "/%06d" for x in lines]
            self.frame_id = [int(x[1]) for x in lines]
            self.frame_seg_id = [int(x[2]) for x in lines]
            self.frame_seg_len = [int(x[3]) for x in lines]

        if self.is_train:
            keep = self.filter_annotation()

            if len(lines[0]) == 2:
                self.image_set_index = [self.image_set_index[idx] for idx in range(len(keep)) if keep[idx]]
                self.frame_id = [self.frame_id[idx] for idx in range(len(keep)) if keep[idx]]
            else:
                self.image_set_index = [self.image_set_index[idx] for idx in range(len(keep)) if keep[idx]]
                self.pattern = [self.pattern[idx] for idx in range(len(keep)) if keep[idx]]
                self.frame_id = [self.frame_id[idx] for idx in range(len(keep)) if keep[idx]]
                self.frame_seg_id = [self.frame_seg_id[idx] for idx in range(len(keep)) if keep[idx]]
                self.frame_seg_len = [self.frame_seg_len[idx] for idx in range(len(keep)) if keep[idx]]

        self.classes_to_ind = dict(zip(self.classes_map, range(len(self.classes_map))))
        self.categories = dict(zip(range(len(self.classes)), self.classes))

        self.classes = [
            {'id': 0, 'name': 'ignored-regions'},
            {'id': 1, 'name': 'pedestrian'},
            {'id': 2, 'name': 'people'},
            {'id': 3, 'name': 'bicycle'},
            {'id': 4, 'name': 'car'},
            {'id': 5, 'name': 'van'},
            {'id': 6, 'name': 'truck'},
            {'id': 7, 'name': 'tricycle'},
            {'id': 8, 'name': 'awning-tricycle'},
            {'id': 9, 'name': 'bus'},
            {'id': 10, 'name': 'motor'},
            {'id': 11, 'name': 'others'},
        ]

        self.annos = self.load_annos(os.path.join(self.cache_dir, self.image_set + "_anno.pkl"))


    def _preprocess_annotation(self, frame_anno , video_size): #frame_anno is a file in annotations (linked to a video)
        boxes = []  # a list of lists
        gt_classes = []  # list

        im_info = video_size



        for target in frame_anno.iterrows():
            target_id = int(target[1][1])
            bbox_left = int(target[1][2])
            bbox_top = int(target[1][3])
            bbox_right = bbox_left+int(target[1][4])
            bbox_bottom = bbox_top+int(target[1][4])
            object_class_num = int(target[1][7])

            box = [
                bbox_left,
                bbox_top,
                bbox_right,
                bbox_bottom
            ]
            boxes.append(box)

            gt_classes.append(self.classes[object_class_num]['name'])

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(gt_classes),
            "im_info": im_info,
        }
        return res


    def load_annos(self, cache_file):

        # video_name = 'uav0000013_00000_v'
        ann_dir = f'C:/Library/School/Project/visDrone/VisDrone2019-VID-train/annotations'
        # ann_file = f'C:/Library/School/Project/visDrone/VisDrone2019-VID-train/annotations/{video_name}.txt'

        annos = []

        for ann_file in os.listdir(ann_dir):
            video_name = ann_file.rstrip(".txt")
            image_dir = ann_dir.rstrip("annotations")+f'sequences/{video_name}'

            # image_dir = f'C:/Library/School/Project/visDrone/VisDrone2019-VID-train/sequences/{video_name}'

            image_name = '0000001.jpg'
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path)
            im_size = tuple(map(int, (image.shape[0].text, image.shape[1].text)))

            ann_df = pd.read_csv(ann_file)
            ann_df.columns = ['frame_index', 'target_id', 'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'score', 'object_category', 'truncation', 'occlusion']
            frame_id_list = sorted(ann_df.loc[:, 'frame_index'].unique().tolist())


            for frame_id in tqdm(frame_id_list):
                mask = ann_df.loc[:, 'frame_index'] == frame_id
                df = ann_df.loc[mask, :]
                anno = self._preprocess_annotation(df, im_size)  # each anno belongs to a single frame
                annos.append(anno)

        return annos



    """
    def _preprocess_annotation(self, video_target , video_size): #video_target is a file in annotations (linked to a video)
        boxes = []  # a list of lists
        gt_classes = []  # list



        # size = target.find("size")
        # im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        im_info = video_size

        ann_df = pd.read_csv(video_target)
        ann_df.columns = ['frame_index', 'target_id', 'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'score', 'object_category', 'truncation', 'occlusion']
        frame_id_list = sorted(ann_df.loc[:, 'frame_index'].unique().tolist())

        for frame_id in tqdm(frame_id_list):
            mask = ann_df.loc[:, 'frame_index'] == frame_id
            df = ann_df.loc[mask, :]

            for target in df.iterrows():
                target_id = int(target[1][1])
                bbox_left = int(target[1][2])
                bbox_top = int(target[1][3])
                bbox_right = bbox_left+int(target[1][4])
                bbox_bottom = bbox_top+int(target[1][4])
                object_class_num = int(target[1][7])

                box = [
                    bbox_left,
                    bbox_top,
                    bbox_right,
                    bbox_bottom
                ]
                boxes.append(box)

                gt_classes.append(self.classes[object_class_num]['name'])
                # gt_classes.append(self.classes_to_ind[obj.find("name").text.lower().strip()])

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(gt_classes),
            "im_info": im_info,
        }
        return res

    """