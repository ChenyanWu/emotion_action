
#%%
import pickle
import os
import numpy as np
import imagesize
import csv
bold_dir = '/ocean/projects/iri180005p/chenyan/dataset/emotion_bold/BOLD_public'
#%%
train_csv_path = os.path.join(bold_dir, 'annotations', 'train.csv')
val_csv_path = os.path.join(bold_dir, 'annotations', 'val.csv')
train_list = []
val_list = []
with open(train_csv_path, 'r') as csv_file:
    csvreader = csv.reader(csv_file)
    for row in csvreader:
        train_list.append(row)
with open(val_csv_path, 'r') as csv_file:
    csvreader = csv.reader(csv_file)
    for row in csvreader:
        val_list.append(row)
#%% construct the dict to store annotation
output = {'split':{'train':[], 'val':[]}, 'annotations':[]}
for val_id, val_annot in enumerate(val_list):
    person_id = int(val_annot[1])
    name_w_personid = val_annot[0] + '_personid{}'.format(person_id)
    output['split']['val'].append(name_w_personid)
    joint_annot_path = os.path.join(bold_dir, 'joints', val_annot[0][:-4]+'.npy')
    joint_annot = np.load(joint_annot_path)
    start_frame = int(joint_annot[:,0].min())
    total_frame = int(joint_annot[:,0].max() - start_frame + 1)
    processed_annot = {}
    kps_np = np.zeros([1,total_frame, 18, 2])
    kps_score = np.zeros([1, total_frame, 18])
    for joint_annot_sample in joint_annot:
        if int(joint_annot_sample[1]) == person_id:
            frame_id = int(joint_annot_sample[0] - start_frame)
            kps_np[0, frame_id] = joint_annot_sample[2:].reshape([18,3])[:,:2]
            kps_score[0, frame_id] = joint_annot_sample[2:].reshape([18,3])[:,2]
        else:
            pass
    processed_annot['keypoint'] = kps_np
    processed_annot['keypoint_score'] = kps_score
    processed_annot['frame_dir'] = name_w_personid
    processed_annot['total_frames'] = total_frame
    img_path = os.path.join(bold_dir, 'mmextract', val_annot[0], 'img_00001.jpg')
    width, height = imagesize.get(img_path)
    processed_annot['original_shape'] = (height, width)
    processed_annot['img_shape'] = (height, width)
    # get the label 
    emotion_cls = np.zeros(26)
    for idx in range(4, 4+26):
        emotion_cls[idx-4] = float(val_annot[idx])
    multi_label = np.argwhere(emotion_cls > 0.5)
    label = []
    for _label in multi_label:
        label.append(_label[0])
    processed_annot['label'] = label
    output['annotations'].append(processed_annot)
#%% load the annotation for train.csv
for val_id, val_annot in enumerate(train_list):
    person_id = int(val_annot[1])
    name_w_personid = val_annot[0] + '_personid{}'.format(person_id)
    output['split']['train'].append(name_w_personid)
    joint_annot_path = os.path.join(bold_dir, 'joints', val_annot[0][:-4]+'.npy')
    joint_annot = np.load(joint_annot_path)
    start_frame = int(joint_annot[:,0].min())
    total_frame = int(joint_annot[:,0].max() - start_frame + 1)
    processed_annot = {}
    kps_np = np.zeros([1,total_frame, 18, 2])
    kps_score = np.zeros([1, total_frame, 18])
    for joint_annot_sample in joint_annot:
        if int(joint_annot_sample[1]) == person_id:
            frame_id = int(joint_annot_sample[0] - start_frame)
            kps_np[0, frame_id] = joint_annot_sample[2:].reshape([18,3])[:,:2]
            kps_score[0, frame_id] = joint_annot_sample[2:].reshape([18,3])[:,2]
        else:
            pass
    processed_annot['keypoint'] = kps_np
    processed_annot['keypoint_score'] = kps_score
    processed_annot['frame_dir'] = name_w_personid
    processed_annot['total_frames'] = total_frame
    img_path = os.path.join(bold_dir, 'mmextract', val_annot[0], 'img_00001.jpg')
    width, height = imagesize.get(img_path)
    processed_annot['original_shape'] = (height, width)
    processed_annot['img_shape'] = (height, width)
    # get the label 
    emotion_cls = np.zeros(26)
    for idx in range(4, 4+26):
        emotion_cls[idx-4] = float(val_annot[idx])
    multi_label = np.argwhere(emotion_cls > 0.5)
    label = []
    for _label in multi_label:
        label.append(_label[0])
    processed_annot['label'] = label
    output['annotations'].append(processed_annot)
#%%
pkl_path = os.path.join(bold_dir, 'annotations', 'bold.pkl')
with open(pkl_path, 'wb') as f:
    annot = pickle.dump(output, f)
# %%
openpose_joint_set = ['Nose', 'Neck', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'R_Eye', 'L_Eye', 'R_Ear', 'L_Ear']
coco_joint_set = ['Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle']
openpose2coco = [openpose_joint_set.index(i) for i in coco_joint_set]
#%% construct the dict to store annotation
output = {'split':{'train':[], 'val':[]}, 'annotations':[]}
for val_id, val_annot in enumerate(val_list):
    person_id = int(val_annot[1])
    name_w_personid = val_annot[0] + '_personid{}'.format(person_id)
    output['split']['val'].append(name_w_personid)
    joint_annot_path = os.path.join(bold_dir, 'joints', val_annot[0][:-4]+'.npy')
    joint_annot = np.load(joint_annot_path)
    start_frame = int(joint_annot[:,0].min())
    total_frame = int(joint_annot[:,0].max() - start_frame + 1)
    processed_annot = {}
    kps_np = np.zeros([1,total_frame, 17, 2])
    kps_score = np.zeros([1, total_frame, 17])
    for joint_annot_sample in joint_annot:
        if int(joint_annot_sample[1]) == person_id:
            frame_id = int(joint_annot_sample[0] - start_frame)
            kps_np[0, frame_id] = joint_annot_sample[2:].reshape([18,3])[openpose2coco,:2]
            kps_score[0, frame_id] = joint_annot_sample[2:].reshape([18,3])[openpose2coco,2]
        else:
            pass
    processed_annot['keypoint'] = kps_np
    processed_annot['keypoint_score'] = kps_score
    processed_annot['frame_dir'] = name_w_personid
    processed_annot['total_frames'] = total_frame
    img_path = os.path.join(bold_dir, 'mmextract', val_annot[0], 'img_00001.jpg')
    width, height = imagesize.get(img_path)
    processed_annot['original_shape'] = (height, width)
    processed_annot['img_shape'] = (height, width)
    # get the label 
    emotion_cls = np.zeros(26)
    for idx in range(4, 4+26):
        emotion_cls[idx-4] = float(val_annot[idx])
    multi_label = np.argwhere(emotion_cls > 0.5)
    label = []
    for _label in multi_label:
        label.append(_label[0])
    processed_annot['label'] = label
    output['annotations'].append(processed_annot)
#%% load the annotation for train.csv
for val_id, val_annot in enumerate(train_list):
    person_id = int(val_annot[1])
    name_w_personid = val_annot[0] + '_personid{}'.format(person_id)
    output['split']['train'].append(name_w_personid)
    joint_annot_path = os.path.join(bold_dir, 'joints', val_annot[0][:-4]+'.npy')
    joint_annot = np.load(joint_annot_path)
    start_frame = int(joint_annot[:,0].min())
    total_frame = int(joint_annot[:,0].max() - start_frame + 1)
    processed_annot = {}
    kps_np = np.zeros([1,total_frame, 17, 2])
    kps_score = np.zeros([1, total_frame, 17])
    for joint_annot_sample in joint_annot:
        if int(joint_annot_sample[1]) == person_id:
            frame_id = int(joint_annot_sample[0] - start_frame)
            kps_np[0, frame_id] = joint_annot_sample[2:].reshape([18,3])[openpose2coco,:2]
            kps_score[0, frame_id] = joint_annot_sample[2:].reshape([18,3])[openpose2coco,2]
        else:
            pass
    processed_annot['keypoint'] = kps_np
    processed_annot['keypoint_score'] = kps_score
    processed_annot['frame_dir'] = name_w_personid
    processed_annot['total_frames'] = total_frame
    img_path = os.path.join(bold_dir, 'mmextract', val_annot[0], 'img_00001.jpg')
    width, height = imagesize.get(img_path)
    processed_annot['original_shape'] = (height, width)
    processed_annot['img_shape'] = (height, width)
    # get the label 
    emotion_cls = np.zeros(26)
    for idx in range(4, 4+26):
        emotion_cls[idx-4] = float(val_annot[idx])
    multi_label = np.argwhere(emotion_cls > 0.5)
    label = []
    for _label in multi_label:
        label.append(_label[0])
    processed_annot['label'] = label
    output['annotations'].append(processed_annot)
#%%
pkl_path = os.path.join(bold_dir, 'annotations', 'bold_cocoJ.pkl')
with open(pkl_path, 'wb') as f:
    annot = pickle.dump(output, f)
# %%
