#%% check weather the pkl file need to point out the person id
# then collect all the hrnet generated pose
import mmcv
pkl_path = '/ocean/projects/iri180005p/chenyan/coding/emotion_body/mmaction2/data/BOLD_public/annotations/bold_cocoJ.pkl'
# %%
annots = mmcv.load(pkl_path)
#%%
val_set = set(annots['split']['val'])
train_set = set(annots['split']['train'])
print(len(set(train_set)))
#%% after check the annots do not come with the person id
# load the csv annotation file and we need to add the personid
output_hrnet_val = '/ocean/projects/iri180005p/chenyan/coding/emotion_body/mmaction2/data/BOLD_public/annotations/bold_hrnet_val.pkl'
output_hrnet_train0 = '/ocean/projects/iri180005p/chenyan/coding/emotion_body/mmaction2/data/BOLD_public/annotations/bold_hrnet_train0.pkl'
output_hrnet_train1 = '/ocean/projects/iri180005p/chenyan/coding/emotion_body/mmaction2/data/BOLD_public/annotations/bold_hrnet_train1.pkl'
output_hrnet_train2 = '/ocean/projects/iri180005p/chenyan/coding/emotion_body/mmaction2/data/BOLD_public/annotations/bold_hrnet_train2.pkl'
output_hrnet_train3 = '/ocean/projects/iri180005p/chenyan/coding/emotion_body/mmaction2/data/BOLD_public/annotations/bold_hrnet_train3.pkl'
output_hrnet_train4 = '/ocean/projects/iri180005p/chenyan/coding/emotion_body/mmaction2/data/BOLD_public/annotations/bold_hrnet_train4.pkl'
output_hrnet_val = mmcv.load(output_hrnet_val)
output_hrnet_train0 = mmcv.load(output_hrnet_train0)
output_hrnet_train1 = mmcv.load(output_hrnet_train1)
output_hrnet_train2 = mmcv.load(output_hrnet_train2)
output_hrnet_train3 = mmcv.load(output_hrnet_train3)
output_hrnet_train4 = mmcv.load(output_hrnet_train4)
#%%
output_hrnet = {'split':{}}
output_hrnet['split']['val'] = output_hrnet_val['split']['val']
output_hrnet['split']['train'] = output_hrnet_train0['split']['train'] + output_hrnet_train1['split']['train'] + output_hrnet_train2['split']['train'] + output_hrnet_train3['split']['train'] + output_hrnet_train4['split']['train']
output_hrnet['annotations'] = output_hrnet_val['annotations'] + output_hrnet_train0['annotations'] + output_hrnet_train1['annotations'] + output_hrnet_train2['annotations']  + output_hrnet_train3['annotations'] + output_hrnet_train4['annotations'] 

#%%
import os
import csv
bold_dir = '/ocean/projects/iri180005p/chenyan/dataset/emotion_bold/BOLD_public'
train_csv_path = os.path.join(bold_dir, 'annotations', 'train.csv')
val_csv_path = os.path.join(bold_dir, 'annotations', 'val.csv')
split_val_id = 0
split_train_id = 0
annot_id = 0
with open(val_csv_path, 'r') as csv_file:
    csvreader = csv.reader(csv_file)
    for row in csvreader:
        assert row[0] == output_hrnet['split']['val'][split_val_id]
        assert row[0] == output_hrnet['annotations'][annot_id]['frame_dir']
        person_id = int(row[1])
        name_w_personid = row[0] + '_personid{}'.format(person_id)
        output_hrnet['split']['val'][split_val_id] = name_w_personid
        output_hrnet['annotations'][annot_id]['frame_dir'] = name_w_personid
        split_val_id += 1
        annot_id += 1
with open(train_csv_path, 'r') as csv_file:
    csvreader = csv.reader(csv_file)
    for row in csvreader:
        assert row[0] == output_hrnet['split']['train'][split_train_id]
        assert row[0] == output_hrnet['annotations'][annot_id]['frame_dir']
        person_id = int(row[1])
        name_w_personid = row[0] + '_personid{}'.format(person_id)
        output_hrnet['split']['train'][split_train_id] = name_w_personid
        output_hrnet['annotations'][annot_id]['frame_dir'] = name_w_personid
        split_train_id += 1
        annot_id += 1
# %%
mmcv.dump(output_hrnet, os.path.join(bold_dir, 'annotations', 'bold_hrnet_cocoJ.pkl'))