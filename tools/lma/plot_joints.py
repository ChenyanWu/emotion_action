#%%
import cv2
import numpy as np
import os
dir_path = '/ocean/projects/iri180005p/chenyan/dataset/emotion_bold/BOLD_public/'
# meta_name = '003/x-6CtPWVi6E.mp4/0208'
meta_name = '003/IzvOYVMltkI.mp4/0114'
# meta_name = '003/y7ncweROe9U.mp4/0130'
# meta_name = '003/phrYEKv0rmw.mp4/0279.mp4'
# meta_name = '003/O_NYCUhZ9zw.mp4/0074.mp4'
# meta_name = '003/7YpF6DntOYw.mp4/0214.mp4'
if meta_name.endswith('mp4'):
    meta_name = meta_name[:-4]

joint_name = meta_name + '.npy'
pose_npy = np.load(os.path.join(dir_path, 'joints', joint_name))
video_name = meta_name + '.mp4'
video_path = os.path.join(dir_path, 'videos', video_name)
start_frame = pose_npy[:, 0].min()
def vis_2d_keypoints(img, kps_list, skt_graph, eps=0.4, radius=4, thickness=1):
    '''
    img: the input img
    kps_list: [(num_kps, 3),...]
    skt_graph: [(), (), ...]
    '''
    # Convert form plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(skt_graph))]
    colors = [(c[0] * 255, c[1] * 255, c[2] * 255) for c in colors]

    for kps in kps_list:
        for skt_id, skt in enumerate(skt_graph):
            p1 = (kps[skt[0], 0].astype(np.int32), kps[skt[0], 1].astype(np.int32))
            p2 = (kps[skt[1], 0].astype(np.int32), kps[skt[1], 1].astype(np.int32))
            
            cv2.circle(img, p1, radius=radius, color=colors[skt_id], thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(img, p2, radius=radius, color=colors[skt_id], thickness=-1, lineType=cv2.LINE_AA)

            if kps[skt[0], 2] > eps and kps[skt[1], 2] > eps:
                cv2.line(img, p1, p2, color=colors[skt_id], thickness=thickness, lineType=cv2.LINE_AA)
    return img

import matplotlib.pyplot as plt

limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
        [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
        [1, 16], [16, 18], [3, 17], [6, 18]]
limbSeq = [(lim[0]-1, lim[1]-1) for lim in limbSeq]

frame_dict = {}
for npy_id in range(len(pose_npy)):
    frame_id = int(pose_npy[npy_id, 0] - start_frame) + 1
    person_id = int(pose_npy[npy_id, 1])
    if frame_id in frame_dict:
        frame_dict[frame_id].append(pose_npy[npy_id, 2:].reshape([18, 3]))
    else:
        frame_dict[frame_id] = []
        frame_dict[frame_id].append(pose_npy[npy_id, 2:].reshape([18, 3]))

for frame_id in frame_dict:
    img_path = os.path.join(dir_path, 'mmextract', video_name, 'img_{:05d}.jpg'.format(frame_id))
    img = cv2.imread(img_path)
    img_draw = vis_2d_keypoints(img, frame_dict[frame_id], limbSeq)
    cv2.imwrite('plot_joints_all_people_folder/img_{:05d}.jpg'.format(frame_id), img_draw)
# %%
for npy_id in range(len(pose_npy)):
    frame_id = int(pose_npy[npy_id, 0] - start_frame) + 1
    person_id = int(pose_npy[npy_id, 1])
    img_path = os.path.join(dir_path, 'mmextract', video_name, 'img_{:05d}.jpg'.format(frame_id))
    img = cv2.imread(img_path)
    img_draw = vis_2d_keypoints(img, pose_npy[npy_id, 2:].reshape([1, 18, 3]), limbSeq)
    cv2.imwrite('plot_joints_folder/img_{:05d}_person_{}.jpg'.format(frame_id, person_id), img_draw)
