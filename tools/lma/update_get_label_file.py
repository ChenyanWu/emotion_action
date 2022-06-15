#%% update the new_annot_list.csv to add the file size
import csv
import os
csv_path = 'new_annot_list.csv'
update_csv_path = 'update_annot_list.csv'
video_folder = '/ocean/projects/iri180005p/chenyan/dataset/emotion_bold/BOLD_public/target_cma'

new_annot_csv = open(update_csv_path, 'w+', encoding='UTF8')
writer = csv.writer(new_annot_csv)

num_oversize = 0
with open(csv_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        video_name = row[0]
        video_path = os.path.join(video_folder, video_name)
        if os.path.exists(video_path):
            size = os.path.getsize(video_path) / (1024*1024)
            if size > 1:
                num_oversize += 1
            row.append('{:.4f}'.format(size))
            writer.writerow(row)
        else:
            row.append('size(MB)')
            writer.writerow(row)
new_annot_csv.close()
print(num_oversize)
# %%
