import argparse
import os.path
import pickle

import pandas as pd
import tqdm

from misc import wi19_evaluate

parser = argparse.ArgumentParser('Hisfrag20 testing script', add_help=False)
parser.add_argument('--pair-maps', type=str, required=True, nargs='+')
parser.add_argument('--output-file', type=str, required=True, default='similarity_matrix.csv')
args = parser.parse_args()

similarity_map = {}
max_value = 10000
if not os.path.isfile(args.output_file):
    for file_path in args.pair_maps:
        print(f'Loading file {file_path}')
        with open(file_path, 'rb') as f:
            pair_map = pickle.load(f)
        for first_img in tqdm.tqdm(pair_map.keys()):
            if first_img not in similarity_map:
                similarity_map[first_img] = {}
            for second_img in pair_map[first_img]:
                value = int(pair_map[first_img][second_img][0] * max_value)
                if second_img not in similarity_map[first_img]:
                    similarity_map[first_img][second_img] = value
                else:
                    similarity_map[first_img][second_img] += value
                    similarity_map[first_img][second_img] = int(similarity_map[first_img][second_img] / 2.)

    print('Creating Dataframe...')
    similarity_map = pd.DataFrame.from_dict(similarity_map, orient='index').sort_index()
    similarity_map = similarity_map.reindex(sorted(similarity_map.columns), axis=1)

    print('To CSV...')
    similarity_map.to_csv(args.output_file, chunksize=1000)
else:
    similarity_map = pd.read_csv(args.output_file, index_col=0)

total_cells = similarity_map.size
cells_with_data = similarity_map.notnull().sum().sum()
print('Total cells missing data:', total_cells - cells_with_data)

print('Starting to calculate performance...')
m_ap, top1, pr_a_k10, pr_a_k100 = wi19_evaluate.get_metrics(similarity_map, lambda x: x.split("_")[0],
                                                            max_value=max_value)

print(f'mAP {m_ap:.3f}\t' f'Top 1 {top1:.3f}\t' f'Pr@k10 {pr_a_k10:.3f}\t' f'Pr@k100 {pr_a_k100:.3f}')
