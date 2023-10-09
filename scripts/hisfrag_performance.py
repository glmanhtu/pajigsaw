import argparse
import pickle

import pandas as pd

from misc import wi19_evaluate

parser = argparse.ArgumentParser('Hisfrag20 testing script', add_help=False)
parser.add_argument('--pair-maps', type=str, required=True, nargs='+')
args = parser.parse_args()

similarity_map = {}

for file_path in args.pair_maps:
    with open(file_path, 'rb') as f:
        pair_map = pickle.load(f)
    for first_img in pair_map:
        if first_img not in similarity_map:
            similarity_map[first_img] = {}
        for second_img in pair_map[first_img]:
            if second_img not in similarity_map[first_img]:
                similarity_map[first_img][second_img] = pair_map[first_img][second_img]
            else:
                similarity_map[first_img][second_img] += pair_map[first_img][second_img]
                similarity_map[first_img][second_img] /= 2.

matrix = pd.DataFrame.from_dict(similarity_map, orient='index').sort_index()
matrix = matrix.reindex(sorted(matrix.columns), axis=1)

total_cells = matrix.size
cells_with_data = matrix.notnull().sum().sum()
print('Total cells missing data:', total_cells - cells_with_data)

m_ap, top1, pr_a_k10, pr_a_k100 = wi19_evaluate.get_metrics(matrix, lambda x: x.split("_")[0])

print(f'mAP {m_ap:.3f}\t' f'Top 1 {top1:.3f}\t' f'Pr@k10 {pr_a_k10:.3f}\t' f'Pr@k100 {pr_a_k100:.3f}')