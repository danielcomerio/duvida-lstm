
import os
import time
from typing import Dict, List, Sequence, Tuple, Union
import numpy as np
import torch
from pathlib import Path
from sklearn import preprocessing


def mkdirs_if_not_exists(dir: Union[Path, str]) -> None:
    if type(dir) is str:
        dir = Path(dir)

    if not dir.exists():
        os.makedirs(dir)


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def group_by_column(data: Union[np.ndarray, Sequence],
                    column_idx: int
                    ) -> Dict[str, Sequence]:
    groups = dict()
    for line in data:
        label = line[column_idx]
        if label not in groups:
            groups[label] = []
        groups[label].append(line)
    return groups


def train_val_test_split(lines: Union[np.ndarray, Sequence],
                         label_idx: int,
                         train_percentage: float = 0.7,
                         val_percentage: float = 0.15,
                         ) -> Tuple[List, List, List]:

    # group the lines of the csv by the class
    groups = group_by_column(lines, label_idx)

    train = []
    val = []
    test = []

    for group in groups.values():
        # randomize the order of samples in the groups
        np.random.shuffle(group)

        # add a subset of the samples for each set in a stratified way
        train_limit = int(len(group) * train_percentage)
        val_limit = train_limit + int(len(group) * val_percentage)

        train += group[:train_limit]
        val += group[train_limit: val_limit]
        test += group[val_limit:]

    return train, val, test


class LabelToInt:
    def __init__(self, labels: Union[np.ndarray, Sequence]):
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit(labels)

    def __call__(self, label):
        if isinstance(label, str):
            return self.label_encoder.transform([label])[0]
        else:
            return self.label_encoder.transform(label)


class SingleStrLabelEncoder:
    def __init__(self, labels):
        self.encoder = LabelToInt(labels)

    def __call__(self, label):
        return int(self.encoder(label)[0])


class SingleIntLabelEncoder:
    def __call__(self, label):
        return int(label[0])


class MultiIntLabelEncoder:
    def __call__(self, label):
        return np.array(label).astype(np.int8)


class FeatureLabelSplitter:
    def __init__(self,
                 feature_indices: Sequence[int],
                 labels_indices: Sequence[int]):
        self.feature_indices = feature_indices
        self.labels_indices = labels_indices

    def __call__(self, line: Sequence):
        features = []
        labels = []

        for idx in self.feature_indices:
            features.append(line[idx])

        for idx in self.labels_indices:
            labels.append(line[idx])

        return features, labels


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self._start_time = time.perf_counter()

    def ellapsed(self):
        return time.perf_counter() - self._start_time