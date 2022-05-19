#%%
import csv
BOLD_train_path = '/ocean/projects/iri180005p/chenyan/dataset/emotion_bold/BOLD_public/annotations/train.csv'
# BOLD_val_path = '/ocean/projects/iri180005p/chenyan/dataset/emotion_bold/BOLD_public/annotations/val.csv'
BOLD_annot_list = []
BOLD_video_list = []
BOLD_person_list = []

with open(BOLD_train_path, 'r') as csv_file:
    csvreader = csv.reader(csv_file)
    for row in csvreader:
        BOLD_annot_list.append(row)
#%% Get the previous LMA annotation
LMA_annot_path = '/ocean/projects/iri180005p/chenyan/dataset/emotion_bold/BOLD_public/annotations/LMA_coding_cleaned.csv'
lma_videl_list = []
with open(LMA_annot_path, 'r') as lma_csv_file:
    lmareader = csv.reader(lma_csv_file)
    for row in lmareader:
        lma_videl_list.append(row[0])
#%% check weather is happy or sad the put into the list
import numpy as np
get_label_list = []
peace_num = 0
for annot in BOLD_annot_list:
    if annot[0] in lma_videl_list:
        # print('exist')
        continue
    emotion_cls = np.zeros(26)
    for idx in range(4, 4+26):
        emotion_cls[idx-4] = float(annot[idx])
    # if emotion_cls[3+1] > 0.95 or emotion_cls[3+7] > 0.5 or emotion_cls[3+22] > 0.5:
    if emotion_cls[3+7] > 0.5 or emotion_cls[3+22] > 0.5:
        get_label_list.append(annot)
    # elif emotion_cls[3+1] > 0.5 and peace_num < 300:
    #     get_label_list.append(annot)
    #     peace_num += 1
print(len(get_label_list))
# %%
# select_idx = np.random.choice(1114, 1000, replace=False)
select_idx = np.random.choice(814, 814, replace=False)
final_label_list = [get_label_list[idx] for idx in select_idx]
#%%
neturl_num, happy_num, sad_num = 0,0,0
# for annot in get_label_list:
for annot in final_label_list:
    emotion_cls = np.zeros(26)
    for idx in range(4, 4+26):
        emotion_cls[idx-4] = float(annot[idx])
    if emotion_cls[3+1] > 0.5:
        neturl_num += 1
    if emotion_cls[3+7] > 0.5:
        happy_num += 1
    if emotion_cls[3+22] > 0.5:
        sad_num += 1
print(neturl_num, happy_num, sad_num)
# %% 
import os
new_annot_csv = open('new_annot_list.csv', 'w+', encoding='UTF8')
writer = csv.writer(new_annot_csv)
header = ['VideoPath', 'URL', 'PersonID']
writer.writerow(header)
joint_dir = '/ocean/projects/iri180005p/bvw546/data/BOLD_public/joints'
label_video_dir = '/ocean/projects/iri180005p/chenyan/dataset/emotion_bold/BOLD_public/upload_cma'
label_target_dir = '/ocean/projects/iri180005p/chenyan/dataset/emotion_bold/BOLD_public/target_cma'

path_list = []
for annot in final_label_list:
    # get the annot_video name
    video_name = annot[0]
    joint_path = os.path.join(joint_dir, video_name[:-3]+'npy')
    joint_npy =  np.load(joint_path)
    person_num = int(joint_npy[:, 1].max()) + 1
    frame_total = int(joint_npy[:, 0].max() - joint_npy[:, 0].min())
    label_video_name = video_name[:-4] + '_{:04d}_{:04d}.mp4'.format(frame_total, person_num)
    label_video_path = os.path.join(label_video_dir, video_name, label_video_name.split('/')[-1])
    if not os.path.exists(label_video_path):
        print(label_video_path)
    else:
        target_video_name = label_video_name[:-4] + '_Person' + str(annot[1]) + '.mp4'
        target_video_path = os.path.join(label_target_dir, target_video_name)
        target_video_dir = os.path.dirname(target_video_path)
        # if target_video_path in path_list:
        #     print(annot)
        # else:
        #     path_list.append(target_video_path)
        os.system('mkdir -p {}'.format(target_video_dir))
        os.system('cp {} {}'.format(label_video_path, target_video_path))
    # cp the annot_video to the target folder
        row = [target_video_name, 'https://cydar.ist.psu.edu/bodyemotionstudy/videos/' + label_video_name, str(annot[1])]
        writer.writerow(row)
new_annot_csv.close()

# %%
vido_check = []
for annot in final_label_list:
    vido_check.append(annot[0])
print(len(set(vido_check)))