from bert_score import BERTScorer
scorer = BERTScorer(lang="en",  rescale_with_baseline=False)

import glob
human_files = "/data/private/WebNLG-models/prediction/challenge/reference.txt"

human_open = open(human_files, "r")
human_dataset = human_open.readlines()
human_open.close()

output_path = "/data/private/WebNLG-models/prediction/challenge/compare/*"
# output_path = "/data/private/WebNLG-models/prediction/challenge/my_output/*"
pred_files = glob.glob(output_path)

score_list = []
for i in range(len(pred_files)):    
    cands = []
    pred_data_open = open(pred_files[i], "r")
    pred_data_dataset = pred_data_open.readlines()
    pred_data_open.close()
    
    P, R, F1 = scorer.score(human_dataset, pred_data_dataset)
    
    F1_list=list(F1.numpy())
    BERT_score = sum(F1_list)/len(F1_list)
    
    score_list.append(BERT_score)    
    
for i in range(len(pred_files)):
    print(pred_files[i])
    print(score_list[i])