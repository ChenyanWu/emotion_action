# %%
import csv
import numpy as np
import os
LMA_anno_path = '/ocean/projects/iri180005p/bvw546/data/BOLD_public/annotations/LMA_coding_cleaned.csv'
LMA_ann_list = []
with open(LMA_anno_path, 'r') as LMA_ann_file:
    LMA_anns = csv.reader(LMA_ann_file)
    for row_id,row in enumerate(LMA_anns):
        if row_id == 0:
            print(row)
        else:
            LMA_ann_list.append(row)
#%%
LMA_video_list = []
LMA_person_list = []
for ann in LMA_ann_list:
    LMA_video_list.append(ann[0])
    LMA_person_list.append(int(ann[2]))
# show the len of the video number and the person instance number
print('video number', len(LMA_video_list))
print('instance number', len(set(LMA_video_list)))
# %%
import csv
BOLD_train_path = '/ocean/projects/iri180005p/chenyan/dataset/emotion_bold/BOLD_public/annotations/train.csv'
BOLD_val_path = '/ocean/projects/iri180005p/chenyan/dataset/emotion_bold/BOLD_public/annotations/val.csv'
BOLD_train_video_list = []
BOLD_train_person_list = []
BOLD_val_video_list = []
BOLD_val_person_list = []

with open(BOLD_train_path, 'r') as csv_file:
    csvreader = csv.reader(csv_file)
    for row in csvreader:
        BOLD_train_video_list.append(row[0])
        BOLD_train_person_list.append(int(row[1]))
print('video number', len(BOLD_train_video_list))
print('instance number', len(set(BOLD_train_video_list)))
with open(BOLD_val_path, 'r') as csv_file:
    csvreader = csv.reader(csv_file)
    for row in csvreader:
        BOLD_val_video_list.append(row[0])
        BOLD_val_person_list.append(int(row[1]))
print('video number', len(BOLD_val_video_list))
print('instance number', len(set(BOLD_val_video_list)))
#%%
overlap_train_video_list = []
num=0
for vid,video in enumerate(BOLD_train_video_list):
    if video in LMA_video_list:
        num += 1
        if BOLD_train_person_list[vid] == LMA_person_list[LMA_video_list.index(video)]:
            overlap_train_video_list.append((video, BOLD_train_person_list[vid]))

# %%
overlap_val_video_list = []
num=0
for vid,video in enumerate(BOLD_val_video_list):
    # if os.path.exists(os.path.join('/ocean/projects/iri180005p/chenyan/coding/emotion_body/mmaction2/data/BOLD_public/videos', video)):
        # num += 1
    if video in LMA_video_list:
        if BOLD_val_person_list[vid] == LMA_person_list[LMA_video_list.index(video)]:
            overlap_val_video_list.append((video, BOLD_val_person_list[vid]))
# %%
