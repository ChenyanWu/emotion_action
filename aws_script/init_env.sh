user@ip-172-31-80-121 emotion_action]$ cd data/BOLD_public/
(pytorch_p38) [ec2-user@ip-172-31-80-121 BOLD_public]$ ls
(pytorch_p38) [ec2-user@ip-172-31-80-121 BOLD_public]$ ln -s /mnt/efs/fs1/dataset/bold/* ./


tar -cvf flow.tar.gz /mnt/efs/fs1/dataset/bold/mmflow


# scp and extract data
scp -i /ocean/projects/iri180005p/chenyan/compvis_2.pem flow.tar.gz ec2-user@ec2-44-202-252-202.compute-1.amazonaws.com:/home/ec2-user/
tar -xvf flow.tar.gz

# config environment from scratch (using Amazon Linux Pytorch)
source activate pytorch
pip install git+https://github.com/open-mmlab/mim.git
mim install mmcv-full==1.4.8
git clone https://github.com/ChenyanWu/emotion_action.git
cd emotion_action/
pip install -r requirements/build.txt
pip install -v -e .
mkdir -p data/BOLD_public/
ln -s /mnt/efs/fs1/dataset/bold/* data/BOLD_public/