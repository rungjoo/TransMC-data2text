# Transforming Multi-Conditioned Generation from Meaning Representation (RANLP 2021)
- https://arxiv.org/pdf/2101.04257.pdf 

## Requirements
1. Pytorch 1.2+
2. Python 3.5+
3. [Huggingface Transformer](https://github.com/huggingface/transformers)
4. nltk library

## Datasets & Evaluation
1. [E2E Dataset](https://github.com/tuetschek/e2e-dataset)
2. [e2e-metrics](https://github.com/tuetschek/e2e-metrics)
3. [BERTScore](https://pypi.org/project/bert-score/)

## Train
For All dataset
```bash
python3 train.py
```

For sampling dataset
```bash
python3 train_sampling.py
```

## Inference
Run inference.py to generate pred.txt with the trained model.

## Evaluation
Refer to evaluation.ipynb file for e2e-metrics (BLEU, NIST, METEOR, ROUGE_L, CIDEr)
```bash
./e2e-metrics/measure_scores.py ./dataset/f_test.txt {prediction.txt}
```

For BERT score, edit the prediction file in the eval_BERTscore.py and
```bash
python3 eval_BERTscore.py
```


