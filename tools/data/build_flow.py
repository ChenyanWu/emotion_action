# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import os.path as osp
import sys
import warnings
from multiprocessing import Lock, Pool
import csv


def extract_frame(vid_item):
    """Generate optical flow using dense flow.

    Args:
        vid_item (list): Video item containing video full path,
            video (short) path, video id.

    Returns:
        bool: Whether generate optical flow successfully.
    """
    full_path, vid_path, vid_id, method, task, report_file = vid_item
    if '/' in vid_path:
        act_name = osp.basename(osp.dirname(vid_path))
        out_full_path = osp.join(args.out_dir, act_name)
    else:
        out_full_path = args.out_dir

    run_success = -1

    if task == 'rgb':
       raise
    elif task == 'flow':
        # cmd = osp.join(
        #     f"denseflow '{full_path}' -a={method} -b=20 -s=1 -o='{out_full_path}'"  # noqa: E501
        #     f'-v')
        cmd = f"denseflow '{full_path}' -a={method} -b=20 -s=1 -o='{out_full_path}' -v"
        run_success = os.system(cmd)
    else:
        raise

    if run_success == 0:
        print(f'{task} {vid_id} {vid_path} {method} done')
        sys.stdout.flush()

        lock.acquire()
        with open(report_file, 'a') as f:
            line = full_path + '\n'
            f.write(line)
        lock.release()
    else:
        print(f'{task} {vid_id} {vid_path} {method} got something wrong')
        sys.stdout.flush()

    return True


def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('src_dir', type=str, help='source video directory')
    parser.add_argument('out_dir', type=str, help='output rawframe directory')
    parser.add_argument(
        '--task',
        type=str,
        default='flow',
        choices=['rgb', 'flow', 'both'],
        help='which type of frames to be extracted')
    parser.add_argument(
        '--level',
        type=int,
        choices=[1, 2],
        default=2,
        help='directory level of data')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=8,
        help='number of workers to build rawframes')
    parser.add_argument(
        '--flow-type',
        type=str,
        default=None,
        choices=[None, 'tvl1', 'warp_tvl1', 'farn', 'brox'],
        help='flow type to be generated')
    parser.add_argument(
        '--out-format',
        type=str,
        default='jpg',
        choices=['jpg', 'h5', 'png'],
        help='output format')
    parser.add_argument(
        '--ext',
        type=str,
        default='avi',
        choices=['avi', 'mp4', 'webm'],
        help='video file extensions')
    parser.add_argument(
        '--mixed-ext',
        action='store_true',
        help='process video files with mixed extensions')
    parser.add_argument(
        '--new-width', type=int, default=0, help='resize image width')
    parser.add_argument(
        '--new-height', type=int, default=0, help='resize image height')
    parser.add_argument(
        '--new-short',
        type=int,
        default=0,
        help='resize image short side length keeping ratio')
    parser.add_argument('--num-gpu', type=int, default=8, help='number of GPU')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume optical flow extraction instead of overwriting')
    parser.add_argument(
        '--use-opencv',
        action='store_true',
        help='Whether to use opencv to extract rgb frames')
    parser.add_argument(
        '--input-frames',
        action='store_true',
        help='Whether to extract flow frames based on rgb frames')
    parser.add_argument(
        '--report-file',
        type=str,
        default='build_report.txt',
        help='report to record files which have been successfully processed')
    args = parser.parse_args()

    return args


def init(lock_):
    global lock
    lock = lock_


if __name__ == '__main__':
    args = parse_args()

    if not osp.isdir(args.out_dir):
        print(f'Creating folder: {args.out_dir}')
        os.makedirs(args.out_dir)

    if args.level == 2:
        classes = os.listdir(args.src_dir)
        for classname in classes:
            new_dir = osp.join(args.out_dir, classname)
            if not osp.isdir(new_dir):
                print(f'Creating folder: {new_dir}')
                os.makedirs(new_dir)

    video_dir = '/data/bold/videos'
    csv_path_1 = '/data/bold/annotations/bold_test_ijcv.csv'
    csv_path_2 = '/data/bold/annotations/train.csv'
    csv_path_3 = '/data/bold/annotations/val.csv'

    # get video name list
    video_name_list = []
    with open(csv_path_1, 'r') as csv_file:
        csvreader = csv.reader(csv_file)
        for row in csvreader:
            video_name_list.append(row[0])
    with open(csv_path_2, 'r') as csv_file:
        csvreader = csv.reader(csv_file)
        for row in csvreader:
            video_name_list.append(row[0])
    with open(csv_path_3, 'r') as csv_file:
        csvreader = csv.reader(csv_file)
        for row in csvreader:
            video_name_list.append(row[0])
    
    video_name_list = list(set(video_name_list))
    
    fullpath_list = [os.path.join(video_dir, video_name) for video_name in video_name_list]
    print('Total number of videos found: ', len(fullpath_list))

    if args.level == 2:
        vid_list = list(
            map(
                lambda p: osp.join(
                    osp.basename(osp.dirname(p)), osp.basename(p)),
                fullpath_list))
    elif args.level == 1:
        vid_list = list(map(osp.basename, fullpath_list))

    lock = Lock()
    pool = Pool(args.num_worker, initializer=init, initargs=(lock, ))
    pool.map(
        extract_frame,
        zip(fullpath_list, vid_list, range(len(vid_list)),
            len(vid_list) * [args.flow_type],
            len(vid_list) * [args.task],
            len(vid_list) * [args.report_file]))
    pool.close()
    pool.join()
