from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

@dataclass
class DatasetConfig:
    """Configuration for a dataset.

    Attributes:
        dataset_name: The name of the dataset.
        data_path: The path to the dataset.
        synthetic_path: The path to the synthetic data.
        subset_root: The root directory of the subsets.
        subsets: A list of subsets to use.
        extensions: A list of file extensions to use.
        extension: A single file extension to use.
        label_map: A mapping from label names to label values.
        group_extraction: The method to use for group extraction.
        group_level: The level of the group in the directory structure.
        filename_delimiter: The delimiter to use for parsing filenames.
        n_neighbors: The number of neighbors to use for k-NN.
        task2_thirds: Magnitude thirds to use for Task 2 feature selection.
        synthetic_folder: The name of the folder containing synthetic data.
    """
    dataset_name: str
    data_path: str
    synthetic_path_residual: str
    synthetic_path_variational: str
    subset_root: Optional[str] = ""
    subsets: Optional[List[str]] = field(default_factory=lambda: [""])
    extensions: Optional[List[str]] = field(default_factory=list)
    extension: Optional[str] = ""
    label_map: Dict[str, int] = field(default_factory=lambda: {"real": 0, "spoof": 1, "synthetic": 2})
    group_extraction: str = "dir"
    group_level: int = -2
    filename_delimiter: str = "_"
    n_neighbors: int = 5
    task2_thirds: Optional[List[int]] = None
    synthetic_folder: str = "spoof"  # Name of the synthetic data folder


def extractor(dataset_name, filename):
    """Extracts user and finger IDs from a filename.

    Args:
        dataset_name: The name of the dataset.
        filename: The filename to parse.

    Returns:
        A tuple containing the user and finger IDs.
    """
    separator = "_"
    if dataset_name == "PLUS":
        parts = filename.split(separator)
        uid = parts[2]
        fid = parts[4]
        return uid, fid
    elif dataset_name == "IDIAP":
        parts = filename.split(separator)
        uid = parts[0]
        fid = parts[1]
        return uid, fid
    elif dataset_name == "SCUT":
        parts = filename.split(separator)
        uid = parts[0]
        fid = parts[1]
        return uid, fid
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def synthetic_extractor(dataset_name, filename) -> int:
    """Extracts user and finger IDs from a filename.

    Args:
        dataset_name: The name of the dataset.
        filename: The filename to parse.

    Returns:
        A tuple containing the user and finger IDs.
    """
    separator = "_"
    if dataset_name == "PLUS":
        parts = filename.split(separator)
        uid = parts[1]
        return uid
    elif dataset_name == "IDIAP":
        parts = filename.split(separator)
        uid = parts[1]
        return uid
    elif dataset_name == "SCUT":
        parts = filename.split(separator)
        uid = parts[1]
        return uid
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def filter_users_with_both_classes(labels: np.ndarray, groups: np.ndarray):
    """Removes users (groups) that do not have samples from both classes.

    Args:
        labels: The labels of the samples.
        groups: The groups of the samples.

    Returns:
        A boolean mask indicating which samples to keep.
    """
    valid_users = []
    unique_groups = np.unique(groups)

    for g in unique_groups:
        user_labels = labels[groups == g]
        if len(np.unique(user_labels)) > 1:
            valid_users.append(g)

    valid_mask = np.isin(groups, valid_users)
    return valid_mask
