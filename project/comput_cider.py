from evaluate import CiderD
import csv
from utils import to_device, Checkpoint, Step, Smoother, Logger
import os
from tqdm import tqdm

def comput_cider_forcsv(pred_csv,target_csv):
    samples_pred =[]
    samples_target =[]
    res, gts = [], {}
    with open(pred_csv, 'r') as fp:
        reader = csv.reader(fp)
        for row in reader:
            samples_pred.append(row[1])
    with open(target_csv, 'r') as fp:
        reader = csv.reader(fp)
        for row in reader:
            samples_target.append(row[2])

    for i in range(len(samples_pred)):
        res.append({'image_id':i, 'caption': [samples_pred[i]]})
        gts[i] = [samples_target[i]]
    CiderD_scorer = CiderD(df='corpus', sigma=15)
    cider_score, cider_scores = CiderD_scorer.compute_score(gts , res)
    print('cider:',cider_score)