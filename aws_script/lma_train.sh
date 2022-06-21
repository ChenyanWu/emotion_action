# lma_idx=3
# CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/recognition/bold_lma/tsn_r101_1x1x10_100e_lma_rgb_frames_wo_extrabranch.py 2 --validate --deterministic --seed 0 --cfg-options data.train.lma_annot_idx=$lma_idx data.val.lma_annot_idx=$lma_idx data.test.lma_annot_idx=$lma_idx work_dir=./work_dirs/lma_rgb_$lma_idx

# lma_idx=5
# CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/recognition/bold_lma/tsn_r101_1x1x10_100e_lma_rgb_frames_wo_extrabranch.py 2 --validate --deterministic --seed 0 --cfg-options data.train.lma_annot_idx=$lma_idx data.val.lma_annot_idx=$lma_idx data.test.lma_annot_idx=$lma_idx work_dir=./work_dirs/lma_rgb_$lma_idx

lma_idx=6
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/recognition/bold_lma/tsn_r101_1x1x10_100e_lma_rgb_frames_wo_extrabranch.py 2 --validate --deterministic --seed 0 --cfg-options data.train.lma_annot_idx=$lma_idx data.val.lma_annot_idx=$lma_idx data.test.lma_annot_idx=$lma_idx work_dir=./work_dirs/lma_rgb_$lma_idx

# lma_idx=7
# CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/recognition/bold_lma/tsn_r101_1x1x10_100e_lma_rgb_frames_wo_extrabranch.py 2 --validate --deterministic --seed 0 --cfg-options data.train.lma_annot_idx=$lma_idx data.val.lma_annot_idx=$lma_idx data.test.lma_annot_idx=$lma_idx work_dir=./work_dirs/lma_rgb_$lma_idx

# lma_idx=8
# CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/recognition/bold_lma/tsn_r101_1x1x10_100e_lma_rgb_frames_wo_extrabranch.py 2 --validate --deterministic --seed 0 --cfg-options data.train.lma_annot_idx=$lma_idx data.val.lma_annot_idx=$lma_idx data.test.lma_annot_idx=$lma_idx work_dir=./work_dirs/lma_rgb_$lma_idx