# %%
import numpy as np
import os
dir_path = '/ocean/projects/iri180005p/chenyan/dataset/emotion_bold/BOLD_public/joints'
video_name = '003/x-6CtPWVi6E.mp4/0208.npy'
pose_npy = np.load(os.path.join(dir_path, video_name))
# %%
