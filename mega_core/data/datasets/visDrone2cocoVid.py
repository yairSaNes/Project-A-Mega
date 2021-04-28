import os
import numpy as np
import pandas as pd
import json
import cv2
from tqdm import tqdm


"""
Convert VisDrone VID dataset to CocoVid format.
CocoVid is and extension of COCO format to vidoes, made by mmtracking. see:
https://github.com/open-mmlab/mmtracking/blob/master/docs/tutorials/customize_dataset.md
annotation file format:
videos: contains a list of videos. Each video is a dictionary with keys name, id. Optional keys include fps, width, and height.
images: contains a list of images. Each image is a dictionary with keys file_name, height, width, id, frame_id, and video_id. Note that the frame_id is 0-index based.
annotations: contains a list of instance annotations. Each annotation is a dictionary with keys bbox, area, id, category_id, instance_id, image_id. The instance_id is only required for tracking.
categories: contains a list of categories. Each category is a dictionary with keys id and name.
example here:
https://github.com/open-mmlab/mmtracking/blob/master/tests/assets/demo_cocovid_data/ann.json
---------------------------------
VisDron VID annotation format
https://github.com/VisDrone/VisDrone2018-VID-toolkit
 <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
        Name	                                                      Description
 ----------------------------------------------------------------------------------------------------------------------------------
    <frame_index>     The frame index of the video frame
     <target_id>      In the DETECTION result file, the identity of the target should be set to the constant -1. 
                      In the GROUNDTRUTH file, the identity of the target is used to provide the temporal corresponding 
	              relation of the bounding boxes in different frames.
     <bbox_left>      The x coordinate of the top-left corner of the predicted bounding box
     <bbox_top>	      The y coordinate of the top-left corner of the predicted object bounding box
    <bbox_width>      The width in pixels of the predicted object bounding box
    <bbox_height>     The height in pixels of the predicted object bounding box
      <score>	      The score in the DETECTION file indicates the confidence of the predicted bounding box enclosing 
                      an object instance.
                      The score in GROUNDTRUTH file is set to 1 or 0. 1 indicates the bounding box is considered in 
                      evaluation, while 0 indicates the bounding box will be ignored.
  <object_category>   The object category indicates the type of annotated object: 
                            ignored regions (0), pedestrian (1), people (2), bicycle (3), car (4), van (5), truck (6), 
                            tricycle (7), awning-tricycle (8), bus (9), motor (10), others (11))
   <truncation>       The score in the DETECTION file should be set to the constant -1.
                      The score in the GROUNDTRUTH file indicates the degree of object parts appears outside a frame 
                      (i.e., no truncation = 0 (truncation ratio 0%), and partial truncation = 1 (truncation ratio 1% °´ 50%)).
    <occlusion>	      The score in the DETECTION file should be set to the constant -1.
                      The score in the GROUNDTRUTH file indicates the fraction of objects being occluded 
                      (i.e., no occlusion = 0 (occlusion ratio 0%), partial occlusion = 1 (occlusion ratio 1% °´ 50%),
                       and heavy occlusion = 2 (occlusion ratio 50% ~ 100%)).
"""


def change_bounding_box_type(bbox_in, type_in, type_out):
    """
    Change bounding box type.
    Parameters
    ----------
    bbox_in : array-like
        Array defining a bounding box, according to type_in.
    type_in, type_out : str
        Types of input and output bounding boxes, one of {ltrb, ltwh, cvat_polygon}:
        ltrb:
            array of 4 numbers defining the top left and bottom right corners coordinates:
                [x_left, y_top, x_right, y_bottom].
        ltwh:
            array of 4 numbers defining the top left corner and bounding box size:
                [x_left, y_top, width, height].
        polygon:
            array of shape (5, 2) of the following form:
                [[x_left, y_top],
                 [x_right, y_top],
                 [x_right, y_bottom],
                 [x_left, y_bottom],
                 [x_left, y_top]].
        corners:
            array of shape (4, 2) of the following form:
                [[x_left, y_top],
                 [x_right, y_top],
                 [x_right, y_bottom],
                 [x_left, y_bottom]].
    Returns
    -------
    bbox_out : ndarray
        Array defining a bounding box, according to type_out.
    """

    # get input bounding box parameters
    if type_in == 'ltrb':
        left = bbox_in[0]
        top = bbox_in[1]
        right = bbox_in[2]
        bottom = bbox_in[3]
    elif type_in == 'ltwh':
        left = bbox_in[0]
        top = bbox_in[1]
        right = left + bbox_in[2]
        bottom = top + bbox_in[3]
    else:
        left = bbox_in[:, 0].min()
        top = bbox_in[:, 1].min()
        right = bbox_in[:, 0].max()
        bottom = bbox_in[:, 1].max()

    # calculate output bounding box
    if type_out == 'ltrb':
        bbox_out = np.array([left, top, right, bottom])
    elif type_out == 'ltwh':
        bbox_out = np.array([left, top, right-left, bottom-top])
    elif type_out == 'polygon':
        bbox_out = np.array([[left, top],
                             [right, top],
                             [right, bottom],
                             [left, bottom],
                             [left, top]])
    elif type_out == 'corners':
        bbox_out = np.array([[left, top],
                             [right, top],
                             [right, bottom],
                             [left, bottom]])

    # verify dtype float32
    bbox_out = bbox_out.astype(np.float32)

    return bbox_out


if __name__ == '__main__':


    # video_name = ' '
    video_name = 'uav0000072_06432_v'

    # ann_file = '/media/moshes2/93d125f5-062c-4c1e-be31-72b3aa5280e51/datasets/VisDrone2019/VisDrone2019-VID-train/annotations/uav0000013_00000_v.txt'
    # image_dir = '/media/moshes2/93d125f5-062c-4c1e-be31-72b3aa5280e51/datasets/VisDrone2019/VisDrone2019-VID-train/sequences/uav0000013_00000_v'
    # ann_file = '/media/moshes2/93d125f5-062c-4c1e-be31-72b3aa5280e51/datasets/VisDrone2019/VisDrone2019-VID-train/annotations/uav0000072_06432_v.txt'
    # image_dir = '/media/moshes2/93d125f5-062c-4c1e-be31-72b3aa5280e51/datasets/VisDrone2019/VisDrone2019-VID-train/sequences/uav0000072_06432_v'
    ann_file = f'/media/moshes2/93d125f5-062c-4c1e-be31-72b3aa5280e51/datasets/VisDrone2019/VisDrone2019-VID-train/annotations/{video_name}.txt'
    image_dir = f'/media/moshes2/93d125f5-062c-4c1e-be31-72b3aa5280e51/datasets/VisDrone2019/VisDrone2019-VID-train/sequences/{video_name}'

    output_root = '/home/moshes2/Projects/fiftyone/dataset/'
    output_dir = os.path.join(output_root, video_name)
    os.makedirs(output_dir, exist_ok=True)

    # video_id = 0
    video_id = 1

    # set video and categories dictionaries
    videos = {
        'id': video_id,
        'name': video_name,
    }

    categories = [
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

    # read annotations
    ann_df = pd.read_csv(ann_file, dtype=float)

    # set column names according visdrone VID annotation format
    ann_df.columns = ['frame_index','target_id','bbox_left','bbox_top','bbox_width','bbox_height','score','object_category','truncation','occlusion']

    # cast type to native python types (since json cannot handle ndarrays)
    columns_int = ['frame_index','target_id', 'object_category','truncation','occlusion']
    # columns_float = ['bbox_left','bbox_top','bbox_width','bbox_height','score']
    ann_df[columns_int] = ann_df[columns_int].astype('int')
    # ann_df[columns_float] = ann_df[columns_float].astype('float')

    # get unique frame indices
    frame_id_list = sorted(ann_df.loc[:, 'frame_index'].unique().tolist())

    # iterate over frames and fill dictionaries
    ann_id = 0
    # video_id = 0

    image_name_template = '{}.jpg'
    image_name_length = 11  # controls number of leading zeros
    images = []
    annotations = []
    for frame_id in tqdm(frame_id_list):

        image_name = image_name_template.format(frame_id).zfill(image_name_length)
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)

        mask = ann_df.loc[:, 'frame_index'] == frame_id
        df = ann_df.loc[mask, :]

        image_dict = {
            'file_name': image_name,
            'height': image.shape[0],
            'width': image.shape[1],
            'id': frame_id,  # ?
            'video_id': video_id,
            'frame_id': frame_id,  # ?
        }

        images.append(image_dict)

        for ind, row in df.iterrows():

            bbox = [row['bbox_left'], row['bbox_top'], row['bbox_width'], row['bbox_height']]
            polygon = change_bounding_box_type(bbox, type_in='ltwh', type_out='polygon').astype(float).flatten().tolist()

            ann_dict = {
                'id': ann_id,
                'image_id': frame_id,
                'video_id': video_id,
                'category_id': int(row['object_category']),
                'instance_id': int(row['target_id']),
                'bbox': bbox,
                'segmentation': polygon,
                'area': row['bbox_width'] * row['bbox_height'],
                'occluded': int(row['occlusion']),
                'truncated': int(row['truncation']),
                'iscrowd': 0,
                'ignore': 1 if (row['score'] == 1) else 0,  # maybe row['object_category'] == 0,  ?
                'is_vid_train_frame': True,  # maybe row['object_category'] == 0,  ?
            }

            annotations.append(ann_dict)

            # advance ids
            ann_id += 1
    # video_id += 1

    data_dict = {
        'categories': categories,
        'videos': videos,
        'images': images,
        'annotations': annotations,
    }

    labels_file_name = 'labels.json'
    labels_file_path = os.path.join(output_dir, labels_file_name)
    with open(labels_file_path, 'w') as fp:
        json.dump(data_dict, fp, indent=4)

    print('Done!')