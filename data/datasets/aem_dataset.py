import glob
import os

import imagesize
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pytorch_metric_learning import samplers

from data.samplers import MPerClassSampler


class AEMDataLoader:

    class FakeSampler:
        def set_epoch(self, _):
            return

    def __init__(self, datasets, batch_size, m, numb_workers, pin_memory):
        mini_batch_size = batch_size // len(datasets)
        max_dataset_length = max([len(x) for x in datasets]) * 10
        self.dataloaders = []
        for dataset in datasets:
            sampler = MPerClassSampler(dataset.data_labels, m=m, length_before_new_iter=max_dataset_length)
            dataloader = DataLoader(dataset, sampler=sampler, pin_memory=pin_memory, batch_size=mini_batch_size,
                                    drop_last=True, num_workers=numb_workers)
            self.dataloaders.append(dataloader)
        self.sampler = AEMDataLoader.FakeSampler()

    def __len__(self):
        return max([len(x) for x in self.dataloaders])

    def __iter__(self):
        for samples in zip(*self.dataloaders):
            images, targets = None, None
            for item in samples:
                item_images, item_targets = item
                if images is None:
                    images = item_images
                    targets = item_targets
                else:
                    images = torch.cat([images, item_images], dim=0)
                    targets = torch.cat([targets, item_targets], dim=0)

            yield images, targets


class AEMLetterDataset(Dataset):
    def __init__(self, dataset_path: str, transforms, letter):
        self.dataset_path = dataset_path
        image_pattern = os.path.join(dataset_path, '**', '*.png')
        files = glob.glob(image_pattern, recursive=True)

        tms = {}
        for file in files:
            file_name_components = os.path.basename(file).split('_')
            curr_letter, tm = file_name_components[0], file_name_components[1]
            tms.setdefault(tm, [])  # Ensure that we have all TMS for consistency between letters

            if curr_letter != letter:
                continue
            width, height = imagesize.get(file)
            if width < 32 or height < 32:
                # Ignore extreme small images
                continue
            if '_ex.png' in file and os.path.exists(file.replace('_ex', '')):
                # Ignore duplicate samples
                continue

            tms.setdefault(tm, []).append(file)

        for tm in list(tms.keys()):
            if len(tms[tm]) < 2:
                tms[tm] = []

        self.labels = sorted(tms.keys())
        self.__label_idxes = {k: i for i, k in enumerate(self.labels)}

        self.data = []
        self.data_labels = []
        for tm in self.labels:
            for anchor in sorted(tms[tm]):
                self.data.append((tm, anchor))
                self.data_labels.append(self.__label_idxes[tm])

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def read_distance_matrix(self, distance_file):
        similarity_matrix = pd.read_csv(distance_file, index_col=0)
        similarity_matrix.index = similarity_matrix.index.map(str)
        similarity_matrix.index = similarity_matrix.index.map(self.__label_idxes)
        similarity_matrix.columns = similarity_matrix.columns.map(str)
        similarity_matrix.columns = similarity_matrix.columns.map(self.__label_idxes)
        results = {}
        for tm_id in similarity_matrix.columns:
            tm_similarity = similarity_matrix[tm_id]
            positives = tm_similarity[tm_similarity < -0.7].keys().to_numpy()
            negatives = tm_similarity[tm_similarity > -0.5].keys().to_numpy()
            results[tm_id] = torch.from_numpy(positives).cuda(), torch.from_numpy(negatives).cuda()
        return results

    def __getitem__(self, idx):
        (tm, anchor) = self.data[idx]

        with Image.open(anchor) as img:
            anchor_img = self.transforms(img.convert('RGB'))

        label = self.__label_idxes[tm]
        return anchor_img, label
