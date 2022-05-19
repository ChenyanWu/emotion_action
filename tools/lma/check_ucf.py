#%%
import pickle
import os
import numpy as np
file_path = '/ocean/projects/iri180005p/chenyan/dataset/UCF101/ucf101.pkl'
with open(file_path, 'rb') as f:
    annot = pickle.load(f)
# %%

