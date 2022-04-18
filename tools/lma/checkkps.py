# %%
import numpy as np
import os
dir_path = '/ocean/projects/iri180005p/chenyan/dataset/emotion_bold/BOLD_public/joints'
video_name = '003/x-6CtPWVi6E.mp4/0208.npy'
pose_npy = np.load(os.path.join(dir_path, video_name))
# %%
import csv
LMA_anno_path = '/ocean/projects/iri180005p/bvw546/data/BOLD_public/annotations/LMA_coding_cleaned.csv'
LMA_ann_list = []
with open(LMA_anno_path, 'r') as LMA_ann_file:
    LMA_anns = csv.reader(LMA_ann_file)
    for row_id,row in enumerate(LMA_anns):
        if row_id == 0:
            print(row)
        else:
            LMA_ann_list.append(row)
# %%
