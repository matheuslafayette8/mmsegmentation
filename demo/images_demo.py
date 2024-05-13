# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmengine.model import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model, show_result_pyplot

import os


def main():
    parser = ArgumentParser()
    parser.add_argument('src_folder', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('dst_folder', help='Destination folder for saving results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--with-labels',
        action='store_true',
        default=False,
        help='Whether to display the class labels.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    # test
    os.makedirs(args.dst_folder, exist_ok=True)
    img_files = os.listdir(args.src_folder)
    
    for img_file in img_files:
        # construct the full path to the input image
        img_path = os.path.join(args.src_folder, img_file)

        # test a single image
        result = inference_model(model, img_path)

        # construct the full path to the output result image
        result_img_path = os.path.join(args.dst_folder, f'result_{img_file}')
    
        # show the results
        show_result_pyplot(
            model,
            img_path,
            result,
            title=args.title,
            opacity=args.opacity,
            with_labels=args.with_labels,
            draw_gt=False,
            show=False,
            out_file=result_img_path)


if __name__ == '__main__':
    main()
