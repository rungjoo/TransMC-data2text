### BERT score with human reference
from bert_score import BERTScorer
scorer = BERTScorer(lang="en",  rescale_with_baseline=False)

from bert_score import score
import glob
human_files = "/data/private/E2E/dataset/f_test.txt"

human_open = open(human_files, "r")
human_dataset = human_open.readlines()
human_open.close()

human_references = []

temp_reference = []
for i in range(len(human_dataset)):
    if human_dataset[i] == '\n':
        human_references.append(temp_reference)
        temp_reference = []
    else:
        temp_reference.append(human_dataset[i].strip())
human_references.append(temp_reference)
human_compare = []
for i in range(len(human_references)):
    for k in range(len(human_references[i])):
        human_compare.append(human_references[i][k])

# output_path = "/data/private/E2E/predictions/final/*"
# output_path = "/data/private/E2E/predictions/reproduce/try_2/*"
output_path = "/data/private/E2E/predictions/no_pretrained/try_1/*"
pred_files = glob.glob(output_path)
# pred_files = ["/project/work/E2E/predictions/final/sampling30_1.txt"]

score_list = []
for i in range(len(pred_files)):    
    cands = []
    pred_data_open = open(pred_files[i], "r")
    pred_data_dataset = pred_data_open.readlines()
    pred_len = len(pred_data_dataset)
    pred_data_open.close()
    
    for k in range(len(pred_data_dataset)):
        out_sen = pred_data_dataset[k].strip()
        repeat_num = len(human_references[k])
        for _ in range(repeat_num):
            cands.append(out_sen)

#     P, R, F1 = score(cands, human_compare, lang='en', verbose=True)
    P, R, F1 = scorer.score(cands, human_compare)
    
    F1_list=list(F1.numpy())
    BERT_score = sum(F1_list)/len(F1_list)
    
    score_list.append(BERT_score)    
    
for i in range(len(pred_files)):
    print(pred_files[i])
    print(score_list[i])