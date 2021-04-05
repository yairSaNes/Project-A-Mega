import glob
import os
import argparse

from mega_core.config import cfg
from predictor import VIDDemo


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Visualization")
    parser.add_argument(
        "method",
        choices=["base", "dff", "fgfa", "rdn", "mega"],
        default="base",
        type=str,
        help="which method to use",
    )
    parser.add_argument(
        "config",
        default="configs/vid_R_101_C4_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "checkpoint",
        default="R_101.pth",
        help="The path to the checkpoint for test.",
    )
    parser.add_argument(
        "--visualize-path",
        default="datasets/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00003001",
        # default="datasets/ILSVRC2015/Data/VID/snippets/val/ILSVRC2015_val_00003001.mp4",
        help="the folder or a video to visualize.",
    )
    parser.add_argument(
        "--suffix",
        default=".JPEG",
        help="the suffix of the images in the image folder.",
    )
    parser.add_argument(
        "--output-folder",
        default="demo/visualization/base",
        help="where to store the visulization result.",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="if True, input a video for visualization.",
    )
    parser.add_argument(
        "--output-video",
        action="store_true",
        help="if True, output a video.",
    )
    for arg in sys.argv[1:]:
        print(arg)

    args = parser.parse_args()
    print(args)
    cfg.merge_from_file("/home/adiyair/basic_code/mega.pytorch-master/configs/BASE_RCNN_1gpu.yaml")
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(["MODEL.WEIGHT", args.checkpoint])

    vid_demo = VIDDemo(
        cfg,
        method=args.method,
        confidence_threshold=0.7,
        output_folder=args.output_folder
    )

    if not args.video:
        visualization_results = vid_demo.run_on_image_folder(args.visualize_path, suffix=args.suffix)
    else:
        visualization_results = vid_demo.run_on_video(args.visualize_path)

    if not args.output_video:
        vid_demo.generate_images(visualization_results)
    else:
        vid_demo.generate_video(visualization_results)

if __name__ == '__main__':
    import sys

    method = 'fgfa'
    config_file = '/home/adiyair/basic_code/mega.pytorch-master/configs/FGFA/vid_R_50_C4_FGFA_1x.yaml'
    checkpoint_file = '/home/adiyair/basic_code/mega.pytorch-master/models/FGFA_R_50.pth'
    video_name = '/home/adiyair/dataset/ILSVRC2015/Data/VID/snippets/train/ILSVRC2015_VID_train_0001/ILSVRC2015_train_00033007.mp4'
    ouput_folder = '/home/adiyair/out'

    sys.argv.append(method)
    sys.argv.append(config_file)
    sys.argv.append(checkpoint_file)
    sys.argv.append('--video')
    sys.argv.append('--visualize-path')
    sys.argv.append(video_name)
    sys.argv.append('--output-folder')
    sys.argv.append(ouput_folder)
    sys.argv.append('--output-video')

    main()

    print('done!')

