import logging

import numpy as np


def calc_map_prak(distances, labels, positive_pairs, negative_pairs, prak=(1, 5)):
    avg_precision = []
    prak_res = [[] for _ in prak]

    for i in range(0, len(distances)):
        cur_dists = distances[i, :]
        idxs = np.argsort(cur_dists).flatten()
        sorted_labels = labels[idxs]
        pos_labels = positive_pairs[labels[i]]
        neg_labels = negative_pairs[labels[i]]
        filtered_labels = []
        # filtered_labels = sorted_labels
        for label in sorted_labels:
            if label in pos_labels or label in neg_labels:
                filtered_labels.append(label)

        cur_sum = []
        pos_count = 1
        correct_count = []
        for idx, label in enumerate(filtered_labels):
            if idx == 0:
                continue    # First img is original image
            if label in pos_labels:
                cur_sum.append(float(pos_count) / idx)
                pos_count += 1
                correct_count.append(1)
            else:
                correct_count.append(0)

        for i, k in enumerate(prak):
            val = sum(correct_count[:k]) / min(sum(correct_count), k)
            prak_res[i].append(val)


        ap = sum(cur_sum) / len(cur_sum)
        avg_precision.append(ap)


    m_ap = sum(avg_precision) / len(avg_precision)
    for i, k in enumerate(prak):
        prak_res[i] = sum(prak_res[i]) / len(prak_res[i])

    return m_ap, tuple(prak_res)

