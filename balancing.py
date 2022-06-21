
import numpy as np
from typing import Sequence, Union

from utils import group_by_column


def balance_by_oversampling(data: Union[np.ndarray, Sequence], label_idx: int):
    """Build a new version of the dataset with 50% of healthy cases and 50%
    hemorrhages. To do so, for each healthy scan, we add a sample from the
    set of hemorrhages. Multiple scans from the set of hemorrhages will be 
    repeated, but data augmentation will add some leven of variability.
    Args:
        data (np.ndarray): Dataset represented as a numpy array. Rows are 
            samples and columns are features.
        label_idx (np.ndarray): Index of the feature (column) that contains
            the label. This feature will be used for balancing the dataset.
    """
    groups = group_by_column(data, label_idx).values()
    max_group_size = max([len(group) for group in groups])

    balanced_data = []

    for i in range(max_group_size):
        for group in groups:
            balanced_data.append(group[i % len(group)])

    return balanced_data