
import numpy as np
from typing import Callable, Optional, Sequence, Tuple, Any, Union
from torch.utils.data import Dataset

from utils import FeatureLabelSplitter
from balancing import balance_by_oversampling


class CsvDataset(Dataset):
    """Dataset for tabular data stored in csv format"""

    def __init__(self,
                 csv_path: str,
                 feature_label_split_fn: Callable,
                 delimiter=";",
                 feature_transforms: Optional[Callable] = None,
                 label_transforms: Optional[Callable] = None,
                 balance_by_column: int = -1):

        self.feature_label_split_fn = feature_label_split_fn
        self.feature_transforms = feature_transforms
        self.label_transforms = label_transforms

        self.data = CsvFunctions.load(csv_path, delimiter=delimiter)

        if balance_by_column >= 0:
            self.data = balance_by_oversampling(
                self.data,
                balance_by_column
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        features, label = self.feature_label_split_fn(self.data[idx])

        if self.feature_transforms is not None:
            features = self.feature_transforms(features)

        if self.label_transforms is not None:
            label = self.label_transforms(label)

        return features, label


class CsvFunctions:
    @staticmethod
    def save(data: Union[np.ndarray, Sequence], path: str, delimiter: str = ';'):
        with open(path, "w") as f:
            for line in data:
                f.write(delimiter.join(line))
                f.write("\n")

    @staticmethod
    def load(path: str, delimiter: Optional[str] = ';', discard_header: bool = True):
        with open(path, "r") as f:
            lines = f.readlines()
        if discard_header:
            lines = lines[1:]
        lines = [l.strip().split(delimiter) for l in lines]
        return lines


def datasets_from_csvs(train_path: str,
                       val_path: str,
                       test_path: str,
                       input_size: int,
                       feature_indices: Sequence[int],
                       labels_indices: Sequence[int],
                       label_to_int_fn: Optional[Callable] = None,
                       balance_by_column: int = -1,
                       delimiter: Optional[str] = ';',
                       train_transforms: Optional[Callable] = None,
                       eval_transforms: Optional[Callable] = None,
                       ):

    input_label_splitter_fn = FeatureLabelSplitter(
        feature_indices, labels_indices)

    train_ds = CsvDataset(
        csv_path=train_path,
        feature_label_split_fn=input_label_splitter_fn,
        feature_transforms=train_transforms,
        label_transforms=label_to_int_fn,
        balance_by_column=balance_by_column,
        delimiter=delimiter
    )

    val_ds = CsvDataset(
        csv_path=val_path,
        feature_label_split_fn=input_label_splitter_fn,
        feature_transforms=eval_transforms,
        label_transforms=label_to_int_fn,
        delimiter=delimiter
    )

    test_ds = CsvDataset(
        csv_path=test_path,
        feature_label_split_fn=input_label_splitter_fn,
        feature_transforms=eval_transforms,
        label_transforms=label_to_int_fn,
        delimiter=delimiter
    )

    return train_ds, val_ds, test_ds