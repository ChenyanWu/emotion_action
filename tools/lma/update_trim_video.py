#%% load the annotations and then generate the dict
BOLD_train_path = '/home/chenyan/dataset/body_langage/BOLD_public/annotations/train.csv'

annot_dict = {}
with open(BOLD_train_path, 'r') as csv_file:
    csvreader = csv.reader(csv_file)
    for row in csvreader:
        video_key = row[0][:-4] + '_Person' + str(row[1]) + '.mp4'
        annot_dict[video_key] = row

#%% update the new_annot_list.csv to add the file size
import csv
import os
import cv2
csv_path = 'new_annot_list.csv'
update_csv_path = 'update_annot_list.csv'
# video_folder = '/ocean/projects/iri180005p/chenyan/dataset/emotion_bold/BOLD_public/target_cma'
video_folder = '/home/chenyan/dataset/body_langage/BOLD_public/give2label'
target_folder = '/home/chenyan/dataset/body_langage/BOLD_public/updategive2label'

new_annot_csv = open(update_csv_path, 'w+', encoding='UTF8')
writer = csv.writer(new_annot_csv)

num_oversize = 0
with open(csv_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        video_name = row[0]
        video_path = os.path.join(video_folder, video_name)
        if os.path.exists(video_path):
            target_path = os.path.join(target_folder, video_name)
            os.system('mkdir -p {}'.format(os.path.dirname(target_path)))
            size = os.path.getsize(video_path) / (1024*1024)
            if size > 1:
                key_name = video_name[:-22] + video_name[-12:]
                cap=cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                start_time = int(annot_dict[key_name][2]) / fps
                end_time = int(annot_dict[key_name][3]) / fps
                ffmpeg_extract_subclip(video_path, start_time, end_time, target_path)
                size = os.path.getsize(target_path) / (1024*1024)
            else:
                os.system('cp {} {}'.format(video_path, target_path))

            if size > 1:
                num_oversize += 1
            row.append('{:.4f}'.format(size))
            writer.writerow(row)
        else:
            print(video_path)
            row.append('size(MB)')
            writer.writerow(row)
new_annot_csv.close()
print(num_oversize)
# %%
import cv2
sample_video_path = '/home/chenyan/dataset/body_langage/BOLD_public/give2label/003/2qQs3Y9OJX0.mp4/0543_0207_0001_Person0.mp4'
cap=cv2.VideoCapture(sample_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
start_time = 0
end_time = 1
ffmpeg_extract_subclip(sample_video_path, start_time, end_time, targetname="test.mp4")
# %%
