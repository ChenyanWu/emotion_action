# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
import csv
import torch
import numpy as np

from mmaction.datasets.pipelines import Resize
from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class BoldframeDataset(BaseDataset):
    """Rawframe dataset for action recognition.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    the directory to frames of a video, total frames of the video and
    the label of a video, which are split with a whitespace.
    Example of a annotation file:

    .. code-block:: txt

        some/directory-1 163 1
        some/directory-2 122 1
        some/directory-3 258 2
        some/directory-4 234 2
        some/directory-5 295 3
        some/directory-6 121 3

    Example of a multi-class annotation file:


    .. code-block:: txt

        some/directory-1 163 1 3 5
        some/directory-2 122 1 2
        some/directory-3 258 2
        some/directory-4 234 2 4 6 8
        some/directory-5 295 3
        some/directory-6 121 3

    Example of a with_offset annotation file (clips from long videos), each
    line indicates the directory to frames of a video, the index of the start
    frame, total frames of the video clip and the label of a video clip, which
    are split with a whitespace.


    .. code-block:: txt

        some/directory-1 12 163 3
        some/directory-2 213 122 4
        some/directory-3 100 258 5
        some/directory-4 98 234 2
        some/directory-5 0 295 3
        some/directory-6 50 121 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str | None): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        filename_tmpl (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
        with_offset (bool): Determines whether the offset information is in
            ann_file. Default: False.
        multi_class (bool): Determines whether it is a multi-class
            recognition dataset. Default: False.
        num_classes (int | None): Number of classes in the dataset.
            Default: None.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
            Default: 'RGB'.
        sample_by_class (bool): Sampling by class, should be set `True` when
            performing inter-class data balancing. Only compatible with
            `multi_class == False`. Only applies for training. Default: False.
        power (float): We support sampling data with the probability
            proportional to the power of its label frequency (freq ^ power)
            when sampling data. `power == 1` indicates uniformly sampling all
            data; `power == 0` indicates uniformly sampling all classes.
            Default: 0.
        dynamic_length (bool): If the dataset length is dynamic (used by
            ClassSpecificDistributedSampler). Default: False.
    """

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
                 **kwargs):
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
            return self.load_json_annotations()
        elif self.ann_file.endswith('.csv'):
            video_infos = []
            # reading csv file
            with open(self.ann_file, 'r') as csvfile:
            # creating a csv reader object
                csvreader = csv.reader(csvfile)
                # extracting each data row one by one
                for row in csvreader:
                    video_info = {}
                    frame_dir = row[0]
                    if self.data_prefix is not None:
                        frame_dir = osp.join(self.data_prefix, frame_dir)
                    video_info['frame_dir'] = frame_dir

                    # idx for offset and total_frames
                    raw_total_frames = len(os.listdir(frame_dir))
                    video_info['offset'] = int(row[2])
                    video_info['total_frames'] = int(row[3]) - int(row[2]) + 1
                    # if use all the frames
                    # video_info['total_frames'] = len(os.listdir(frame_dir))

                    # idx for label[s]
                    emotion_cls = np.zeros(26)
                    for idx in range(4, 4+26):
                        emotion_cls[idx-4] = float(row[idx])
                    # emotion_label = np.argmax(emotion_cls)
                    # line += ' ' + str(emotion_label) + '\n'
                    multi_label = np.argwhere(emotion_cls > 0.5)
                    label = []
                    for _label in multi_label:
                        label.append(_label[0])
                    # assert label, f'missing label in line: {line}'
                    if self.multi_class:
                        assert self.num_classes is not None
                        video_info['label'] = label
                    else:
                        raise

                    # compute the bbox for each frame
                    person_id = int(row[1])
                    joint_path = osp.join(self.data_prefix, '../joints', row[0][:-4] + '.npy')
                    joint_npy = np.load(joint_path)
                    # aggregate = True
                    aggregate = False #without aggregate, the performance is better
                    if aggregate:
                        selected_frame = joint_npy[:, 1] == person_id
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
                    else:
                        start_frame_id = int(joint_npy[:,0].min())
                        selected_frame = joint_npy[:, 1] == person_id
                        joint_npy = joint_npy[selected_frame] # first two are frame number and entity id (num_frames, 56)
                        crop_bboxes = np.zeros([raw_total_frames, 4])
                        for frame_joint in joint_npy:
                            frame_id = int(frame_joint[0] - start_frame_id)
                            joint = frame_joint[2:].reshape([18, 3])
                            x1 = joint[joint[:,2] > 1e-7, 0].min()
                            x2 = joint[joint[:,2] > 1e-7, 0].max()
                            y1 = joint[joint[:,2] > 1e-7, 1].min()
                            y2 = joint[joint[:,2] > 1e-7, 1].max()
                            crop_bboxes[frame_id, :] = np.array([(x2+x1)/2, (y2+y1)/2, (x2-x1)/200., (y2-y1)/200.])
                        scale_w = crop_bboxes[:,2].max()
                        scale_h = crop_bboxes[:,3].max()
                        crop_bboxes[:,2] = scale_w
                        crop_bboxes[:,3] = scale_h
                        video_info['crop_bboxes'] = crop_bboxes
                    video_infos.append(video_info)
            return video_infos

        else:
            video_infos = []
            with open(self.ann_file, 'r') as fin:
                for line in fin:
                    line_split = line.strip().split()
                    video_info = {}
                    idx = 0
                    # idx for frame_dir
                    frame_dir = line_split[idx]
                    if self.data_prefix is not None:
                        frame_dir = osp.join(self.data_prefix, frame_dir)
                    video_info['frame_dir'] = frame_dir
                    idx += 1
                    if self.with_offset:
                        # idx for offset and total_frames
                        video_info['offset'] = int(line_split[idx])
                        video_info['total_frames'] = int(line_split[idx + 1])
                        idx += 2
                    else:
                        # idx for total_frames
                        video_info['total_frames'] = int(line_split[idx])
                        idx += 1
                    # idx for label[s]
                    label = [int(x) for x in line_split[idx:]]
                    # assert label, f'missing label in line: {line}'
                    if self.multi_class:
                        assert self.num_classes is not None
                        video_info['label'] = label
                    else:
                        assert len(label) == 1
                        video_info['label'] = label[0]
                    video_infos.append(video_info)

            return video_infos

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
