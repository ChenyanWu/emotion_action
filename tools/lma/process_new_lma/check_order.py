#%% compare file order
import csv
import pandas as pd
chenyan_csv = 'update_sorted_annot_list.csv'
cma_xlsx = 'new_annot_list.xlsx'
xls = pd.ExcelFile(cma_xlsx)
df1 = pd.read_excel(xls, 'new_annot_list')
#%%
idx_video = 0
with open(chenyan_csv, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        chenyan_video_name = row[0]
        if chenyan_video_name == 'VideoPath':
            pass
        else:
            cma_video_name = df1['VideoPath'][idx_video]
            if cma_video_name != chenyan_video_name:
                print(cma_video_name, chenyan_video_name, idx_video)
            idx_video += 1
# %% read the new 2022 coding xlsx
out_csv_file = 'LMA_coding_2022_cleaned.csv'
new_annot_csv = open(out_csv_file, 'w+', encoding='UTF8')
writer = csv.writer(new_annot_csv)

new_lma_coding = 'LMA_coding_2022_raw.csv'
with open(new_lma_coding, 'r') as new_lma_csv:
    csv_reader = csv.reader(new_lma_csv)
    for idx,row in enumerate(csv_reader):
        if row[1] not in row[0]:
            print(row[0], idx)
        row2write = []
        if row[0] == 'vidID':
            writer.writerow(["vidID","clip_num","personID","passive_weight","arms_to_upper_body" ,"sink","head_drop" ,"jump","rhythmicity","spread","free_flow","light_weight","up_or_rise","rotation","emotion"])
        else:
            try:
                analized_flag = int(row[5])
                if analized_flag == 1:
                    vID_name = row[0][:-22] + '.mp4'
                    row2write.append(vID_name)
                    row2write.append(row[0].split('/')[-1][:14])
                    row2write.append(row[6])
                    row2write.extend(row[8:8+11])
                    row2write.append(row[7])
                    writer.writerow(row2write)
                else:
                    pass
            except:
                pass
new_annot_csv.close()
#%%
