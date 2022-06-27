# Copyright (c) OpenMMLab. All rights reserved.
import copy
from curses import raw
import os
import os.path as osp
import csv
import torch
import numpy as np

from mmaction.datasets.pipelines import Resize
from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class LmaframeDataset(BaseDataset):
    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 filename_tmpl='img_{:05}.jpg',
                 with_offset=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB',
                 sample_by_class=False,
                 power=0.,
                 dynamic_length=False,
                 lma_annot_idx=4,
                 **kwargs):
        self.lma_annot_idx = lma_annot_idx
        self.filename_tmpl = filename_tmpl
        self.with_offset = with_offset
        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode,
            multi_class,
            num_classes,
            start_index,
            modality,
            sample_by_class=sample_by_class,
            power=power,
            dynamic_length=dynamic_length)
        self.short_cycle_factors = kwargs.get('short_cycle_factors',
                                              [0.5, 0.7071])
        self.default_s = kwargs.get('default_s', (224, 224))

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            raise
        elif self.ann_file.endswith('.csv'):
            video_infos = []
            # reading csv file
            with open(self.ann_file, 'r') as csvfile:
            # creating a csv reader object
                csvreader = csv.reader(csvfile)

                # extracting each data row one by one
                for row in csvreader:
                    if row[0] == 'vidID':
                        continue

                    video_info = {}
                    frame_dir = row[0]
                    if self.data_prefix is not None:
                        frame_dir = osp.join(self.data_prefix, frame_dir)
                    video_info['frame_dir'] = frame_dir

                    # idx for offset and total_frames
                    raw_total_frames = len(os.listdir(frame_dir))

                    # video_info['offset'] = int(row[2])
                    # video_info['total_frames'] = int(row[3]) - int(row[2]) + 1
                    # if use all the frames
                    video_info['offset'] = 0
                    video_info['total_frames'] = raw_total_frames

                    # get the lma label
                    lma_annot_idx = self.lma_annot_idx
                    try:
                        label = int(int(row[lma_annot_idx])>0)
                    except:
                        label = 0
                    # get the meta label
                    try:
                        meta_label = int(row[12])
                    except:
                        meta_label = 0
                    video_info['label'] = label
                    video_info['meta_label'] = meta_label

                    # compute the bbox for each frame
                    person_id = int(row[2])
                    joint_path = osp.join(self.data_prefix, '../joints', row[0][:-4] + '.npy')
                    joint_npy = np.load(joint_path)
                    
                    start_frame_id = int(joint_npy[:,0].min())
                    selected_frame = joint_npy[:, 1] == person_id
                    # use the unaggregate bbox
                    # joint_npy = joint_npy[selected_frame] # first two are frame number and entity id (num_frames, 56)
                    
                    # crop_bboxes = np.zeros([raw_total_frames, 4])
                    # for frame_joint in joint_npy:
                    #     frame_id = int(frame_joint[0] - start_frame_id)
                    #     joint = frame_joint[2:].reshape([18, 3])
                    #     x1 = joint[joint[:,2] > 1e-7, 0].min()
                    #     x2 = joint[joint[:,2] > 1e-7, 0].max()
                    #     y1 = joint[joint[:,2] > 1e-7, 1].min()
                    #     y2 = joint[joint[:,2] > 1e-7, 1].max()
                    #     crop_bboxes[frame_id, :] = np.array([(x2+x1)/2, (y2+y1)/2, (x2-x1)/200., (y2-y1)/200.])
                    # scale_w = crop_bboxes[:,2].max()
                    # scale_h = crop_bboxes[:,3].max()
                    # crop_bboxes[:,2] = scale_w
                    # crop_bboxes[:,3] = scale_h
                    # video_info['crop_bboxes'] = crop_bboxes

                    # use the aggregate bbox
                    joint_npy = joint_npy[selected_frame, 2:] # first two are frame number and entity id
                    joint_npy = joint_npy.reshape(joint_npy.shape[0], 18, 3)
                    x1 = joint_npy[joint_npy[:,:,2] > 1e-7, 0].min()
                    x2 = joint_npy[joint_npy[:,:,2] > 1e-7, 0].max()
                    y1 = joint_npy[joint_npy[:,:,2] > 1e-7, 1].min()
                    y2 = joint_npy[joint_npy[:,:,2] > 1e-7, 1].max()
                    # scale = max((x2-x1)/200., (y2-y1)/200.)
                    # bbox = np.array([(x2+x1)/2, (y2+y1)/2, scale, scale])
                    bbox = np.array([(x2+x1)/2, (y2+y1)/2, (x2-x1)/200., (y2-y1)/200.])
                    video_info['crop_bboxes'] = np.tile(bbox, [raw_total_frames, 1])

                    video_infos.append(video_info)

                    # if label == 1 and ('train' in self.ann_file):
                    #     for _ in range(3):
                    #         copy_video_info = copy.deepcopy(video_info)
                    #         video_infos.append(copy_video_info)
            return video_infos

        else:
            raise

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""

        def pipeline_for_a_sample(idx):
            results = copy.deepcopy(self.video_infos[idx])
            results['filename_tmpl'] = self.filename_tmpl
            results['modality'] = self.modality
            results['start_index'] = self.start_index

            # prepare tensor in getitem
            if self.multi_class:
                onehot = torch.zeros(self.num_classes)
                onehot[results['label']] = 1.
                results['label'] = onehot

            return self.pipeline(results)

        if isinstance(idx, tuple):
            index, short_cycle_idx = idx
            last_resize = None
            for trans in self.pipeline.transforms:
                if isinstance(trans, Resize):
                    last_resize = trans
            origin_scale = self.default_s
            long_cycle_scale = last_resize.scale

            if short_cycle_idx in [0, 1]:
                # 0 and 1 is hard-coded as PySlowFast
                scale_ratio = self.short_cycle_factors[short_cycle_idx]
                target_scale = tuple(
                    [int(round(scale_ratio * s)) for s in origin_scale])
                last_resize.scale = target_scale
            res = pipeline_for_a_sample(index)
            last_resize.scale = long_cycle_scale
            return res
        else:
            return pipeline_for_a_sample(idx)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        if self.multi_class:
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)
