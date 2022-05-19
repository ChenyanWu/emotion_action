
# %%
import numpy as np
import os
dir_path = '/ocean/projects/iri180005p/chenyan/dataset/emotion_bold/BOLD_public/joints'
video_name = '003/x-6CtPWVi6E.mp4/0208.npy'
pose_npy = np.load(os.path.join(dir_path, video_name))
# %%
import csv
BOLD_train_path = '/ocean/projects/iri180005p/chenyan/dataset/emotion_bold/BOLD_public/annotations/train.csv'
BOLD_val_path = '/ocean/projects/iri180005p/chenyan/dataset/emotion_bold/BOLD_public/annotations/val.csv'

BOLD_train_list = []
with open(BOLD_train_path, 'r') as csv_file:
    csvreader = csv.reader(csv_file)
    for row in csvreader:
        BOLD_train_list.append(row)
print(BOLD_train_list[0])
#%% An example to extract video
exp = BOLD_train_list[0]
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
video_path = os.path.join('/ocean/projects/iri180005p/chenyan/coding/emotion_body/mmaction2/data/BOLD_public/videos', exp[0], '0114_0124_0005.mp4')
# video_path = os.path.join('/ocean/projects/iri180005p/chenyan/coding/emotion_body/mmaction2/data/BOLD_public/videos', exp[0])
start_time, end_time = int(exp[2]) / 25, int(exp[3]) / 25
# start_time, end_time = 0, 5
# ffmpeg_extract_subclip(video_path, start_time, end_time-1, targetname="test.mp4")
ffmpeg_extract_subclip('raw_process.mp4', start_time, end_time-1, targetname="test.mp4")
# print(video_path)
# os.system('cp {} {}'.format(video_path, 'raw.mp4'))
# %%
# from moviepy.editor import *

# loading video gfg
# clip = VideoFileClip(video_path)
# print(clip.fps)
# n_frames = clip.reader.nframes
# print(n_frames)
# getting only first 5 seconds
# clip = clip.subclip(0, 5)
# clip.write_videofile("clip.mp4")
# %%
# import cv2
# from tqdm import tqdm
# fps_file = open('fps.txt', 'a+')

# for _exp in tqdm(BOLD_train_list):
#     video_path = os.path.join('/ocean/projects/iri180005p/chenyan/coding/emotion_body/mmaction2/data/BOLD_public/videos', _exp[0])
#     cap=cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     line = _exp[0] + ' {}\n'.format(fps)
#     fps_file.write(line)
    # if fps != 25.0:
    #     print(_exp[0])
# %%
import cv2
from tqdm import tqdm
fps_file = open('gen_fps.txt', 'a+')
for _exp in tqdm(BOLD_train_list):
    video_dir = os.path.join('/ocean/projects/iri180005p/chenyan/coding/emotion_body/mmaction2/data/BOLD_public/upload_cma', _exp[0])
    genvideo_name = os.listdir(video_dir)[0]
    part_video_path = os.path.join(_exp[0], genvideo_name)
    video_path = os.path.join('/ocean/projects/iri180005p/chenyan/coding/emotion_body/mmaction2/data/BOLD_public/upload_cma', part_video_path)
    # print(part_video_path)
    # if not os.path.exists(video_path):
    #     print('no')
    cap=cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    line = part_video_path + ' {}\n'.format(fps)
    fps_file.write(line)
# %%
