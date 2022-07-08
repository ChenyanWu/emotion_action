#%% 
import csv
enlarge_lma_annot = '/ocean/projects/iri180005p/chenyan/dataset/emotion_bold/BOLD_public/annotations/LMA_coding_cleaned_enlarge.csv'

new_train_coding = 'LMA_coding_cleaned_enlarge_train.csv'
new_val_coding = 'LMA_coding_cleaned_enlarge_val'

new_train_csv = open(new_train_coding, 'w+', encoding='UTF8')
writer_1 = csv.writer(new_train_csv)
new_val_csv = open(new_val_coding, 'w+', encoding='UTF8')
writer_2 = csv.writer(new_val_csv)

with open(enlarge_lma_annot, 'r') as new_lma_csv:
    csv_reader = csv.reader(new_lma_csv)
    for idx,row in enumerate(csv_reader):
        if row[0] == 'vidID':
            writer_2.