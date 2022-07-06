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
# %%
