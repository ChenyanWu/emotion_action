ln -s /mnt/efs/fs1/dataset/bold/* ./
tar -cvf flow.tar.gz /mnt/efs/fs1/dataset/bold/mmflow


# scp and extract data
scp chenyan@data.bridges2.psc.edu:/ocean/projects/iri180005p/chenyan/dataset/emotion_bold/BOLD_public/bold_frame.zip ./
unzip bold_frame.zip

# config environment from scratch (using Amazon Linux Pytorch)
conda activate pytorch_p39
pip install git+https://github.com/open-mmlab/mim.git
mim install mmcv-full==1.4.8
git clone https://github.com/ChenyanWu/emotion_action.git
cd emotion_action/
pip install -r requirements/build.txt
pip install -v -e .
mkdir -p data/BOLD_public/
ln -s /mnt/efs/fs1/dataset/bold/* data/BOLD_public/