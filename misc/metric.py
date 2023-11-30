import numpy as np


def calc_map_prak(distances, labels, positive_pairs, negative_pairs=None, prak=(1, 5)):
    avg_precision = []
    prak_res = [[] for _ in prak]

    for i in range(0, len(distances)):
        cur_dists = distances[i, :]
        idxs = np.argsort(cur_dists).flatten()
        sorted_labels = labels[idxs].tolist()
        pos_labels = positive_pairs[labels[i]]
        if negative_pairs is not None:
            neg_labels = negative_pairs[labels[i]]
            for li, label in reversed(list(enumerate(sorted_labels))):
                if label not in pos_labels and label not in neg_labels:
                    del sorted_labels[li]

        cur_sum = []
        pos_count = 1
        correct_count = []
        for idx, label in enumerate(sorted_labels):
            if idx == 0:
                continue    # First img is original image
            if label in pos_labels:
                cur_sum.append(float(pos_count) / idx)
                pos_count += 1
                correct_count.append(1)
            else:
                correct_count.append(0)

        if sum(correct_count) == 0:
            # If there is no positive pair, there should be a problem in GT
            # Ignore for now
            continue

        for i, k in enumerate(prak):
            val = sum(correct_count[:k]) / min(sum(correct_count), k)
            prak_res[i].append(val)


        ap = sum(cur_sum) / len(cur_sum)
        avg_precision.append(ap)


    m_ap = sum(avg_precision) / len(avg_precision)
    for i, k in enumerate(prak):
        prak_res[i] = sum(prak_res[i]) / len(prak_res[i])

    return m_ap, tuple(prak_res)

