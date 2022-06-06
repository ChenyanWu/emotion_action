user@ip-172-31-80-121 emotion_action]$ cd data/BOLD_public/
(pytorch_p38) [ec2-user@ip-172-31-80-121 BOLD_public]$ ls
(pytorch_p38) [ec2-user@ip-172-31-80-121 BOLD_public]$ ln -s /mnt/efs/fs1/dataset/bold/* ./


tar -cvf flow.tar.gz /mnt/efs/fs1/dataset/bold/mmflow
