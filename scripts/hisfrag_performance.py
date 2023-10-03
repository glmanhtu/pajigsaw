import argparse
import pickle

import pandas as pd

from misc import wi19_evaluate

parser = argparse.ArgumentParser('Hisfrag20 testing script', add_help=False)
parser.add_argument('--path', type=str, required=True)
args = parser.parse_args()

with open(args.path, 'rb') as f:
    similarity_map = pickle.load(f)

matrix = pd.DataFrame.from_dict(similarity_map, orient='index').sort_index()
matrix = matrix.reindex(sorted(matrix.columns), axis=1)

m_ap, top1, pr_a_k10, pr_a_k100 = wi19_evaluate.get_metrics(matrix, lambda x: x.split("_")[0])

print(f'mAP {m_ap:.3f}\t' f'Top 1 {top1:.3f}\t' f'Pr@k10 {pr_a_k10:.3f}\t' f'Pr@k100 {pr_a_k100:.3f}')