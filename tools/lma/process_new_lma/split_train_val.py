#%% 
import csv
enlarge_lma_annot = 'LMA_coding_cleaned_enlarge.csv'

check_dict = {}
with open(enlarge_lma_annot, 'r') as new_lma_csv:
    csv_reader = csv.reader(new_lma_csv)
    for idx,row in enumerate(csv_reader):
        key_csv = row[0] + str(row[2])
        check_dict[key_csv] = row[1:]
#%%
lma_processed_annot = 'LMA_coding_cleaned_enlarge_val.csv'
with open(lma_processed_annot, 'r') as new_lma_csv:
    csv_reader = csv.reader(new_lma_csv)
    for idx,row in enumerate(csv_reader):
        key_csv = row[0] + str(row[2])
        if check_dict[key_csv] != row[1:]:
            print('error')