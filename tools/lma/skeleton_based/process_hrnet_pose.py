#%%
import mmcv
import os

bold_folder = '/ocean/projects/iri180005p/chenyan/coding/emotion_body/mmaction2/data/BOLD_public/'
hrnet_pkl_path = os.path.join(bold_folder, 'annotations', 'bold_hrnet_cocoJ.pkl')
coco_pkl_path = os.path.join(bold_folder, 'annotations', 'bold_cocoJ.pkl')
old_pose = mmcv.load(coco_pkl_path)
new_pose = mmcv.load(hrnet_pkl_path)
#%% check the name are the same for old_pose and new_pose
for idx in range(len(old_pose['annotations'])):
    assert old_pose['annotations'][idx]['frame_dir'] == new_pose['annotations'][idx]['frame_dir']
    assert old_pose['annotations'][idx]['label'] == new_pose['annotations'][idx]['label']
# %%
# import numpy as np
# for idx in range(len(old_pose['annotations'])):
#     num_person = len(new_pose['annotations'][idx]['keypoint'])
#     min_diff_value = 100000000000
#     for person_id in range(num_person):
#         diff = (old_pose['annotations'][idx]['keypoint'] - new_pose['annotations'][idx]['keypoint'][person_id:person_id+1])
#         diff_value = (diff * diff).mean()
#         if diff_value < min_diff_value:
#             min_diff_value = diff_value
#             matched_person_id = person_id
#     new_pose['annotations'][idx]['keypoint'] = new_pose['annotations'][idx]['keypoint'][matched_person_id:matched_person_id+1]
#     new_pose['annotations'][idx]['keypoint_score'] = new_pose['annotations'][idx]['keypoint_score'][matched_person_id:matched_person_id+1]
#%%
# mmcv.dump(new_pose, os.path.join(bold_folder, 'annotations', 'bold_hrnet_matched_cocoJ.pkl'))
#%%
import numpy as np
for idx in range(len(old_pose['annotations'])):
    num_person, num_frames, _, _ = new_pose['annotations'][idx]['keypoint'].shape
    refined_kps = np.zeros_like(old_pose['annotations'][idx]['keypoint'])
    refined_kps_score = np.zeros_like(old_pose['annotations'][idx]['keypoint_score'])
    for frame_id in range(num_frames):
        min_diff_value = 100000000000
        matched_person_id = 10000
        score = np.where(old_pose['annotations'][idx]['keypoint_score'][0,frame_id]>0.5, 1, 0)
        check_score = np.where(old_pose['annotations'][idx]['keypoint_score'][0,frame_id]>0.3, 1, 0)
        if check_score.sum() > 0:
            for person_id in range(num_person):
                diff = (old_pose['annotations'][idx]['keypoint'][0,frame_id] - new_pose['annotations'][idx]['keypoint'][person_id, frame_id])
                diff_value = (diff * diff * score[:, None]).mean()
                if diff_value < min_diff_value:
                    min_diff_value = diff_value
                    matched_person_id = person_id
            # if min_diff_value > 700:
            #     print(new_pose['annotations'][idx]['keypoint'][matched_person_id, frame_id])
            #     print('****')
            #     print(old_pose['annotations'][idx]['keypoint'][0,frame_id])
            #     print(min_diff_value)
            # print(min_diff_value, matched_person_id)
            refined_kps[0, frame_id] = new_pose['annotations'][idx]['keypoint'][matched_person_id, frame_id]
            refined_kps_score[0, frame_id] = new_pose['annotations'][idx]['keypoint_score'][matched_person_id, frame_id]
        else:
            # print(idx)
            pass
    new_pose['annotations'][idx]['keypoint'] = refined_kps
    new_pose['annotations'][idx]['keypoint_score'] = refined_kps_score
#%%
mmcv.dump(new_pose, os.path.join(bold_folder, 'annotations', 'bold_hrnet_matched_cocoJ_3.pkl'))
# %%
