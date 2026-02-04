import os
import glob
from typing import List, Tuple
import cv2
import numpy as np
from src.data_utils import DatasetConfig, extractor, synthetic_extractor

def _find_image_files(folder_path: str, extensions: List[str]) -> List[str]:
    """Finds all image files in a folder with the given extensions."""
    img_files = []
    for ext in extensions:
        pattern = os.path.join(folder_path, "**", f"*.{ext}")
        img_files.extend(glob.glob(pattern, recursive=True))
    return sorted(list(set(img_files)))

def _load_images_from_path(
    folder_path: str,
    extensions: List[str],
    label: int,
    dataset_name: str,
    synthetic: bool = False,
) -> Tuple[List[np.ndarray], List[int], List[int]]:
    """Loads images from a folder and assigns labels and groups."""
    images, labels, groups = [], [], []
    if not os.path.isdir(folder_path):
        return images, labels, groups

    img_files = _find_image_files(folder_path, extensions)
    for img_path in img_files:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        images.append(img)
        labels.append(label)
        filename = os.path.splitext(os.path.basename(img_path))[0]
        if synthetic:
            user_id = synthetic_extractor(dataset_name, filename)
        else:
            user_id, _ = extractor(dataset_name, filename)
        groups.append(int(user_id))
    return images, labels, groups

def get_data(config: DatasetConfig) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Loads images from a dataset."""
    images, labels, groups = [], [], []
    extensions = config.extensions or ["png", "jpg", "jpeg", "bmp"]
    extensions = [ext.lstrip(".") for ext in extensions]

    for subset in config.subsets:
        subset_parts = [part for part in (config.subset_root, subset) if part]

        for label_name, label_value in config.label_map.items():
            path_parts = [config.data_path] + subset_parts + [label_name]
            folder_path = os.path.join(*path_parts)
            imgs, lbls, grps = _load_images_from_path(
                folder_path, extensions, label_value, config.dataset_name
            )
            images.extend(imgs)
            labels.extend(lbls)
            groups.extend(grps)

    return images, np.array(labels), np.array(groups)

def get_data_with_synthetic(config: DatasetConfig):
    """Loads images from a dataset, including synthetic samples for Task 2."""
    extensions = config.extensions or ["png", "jpg", "jpeg", "bmp"]
    extensions = [ext.lstrip(".") for ext in extensions]
    
    real_images, real_labels, real_groups = [], [], []
    fake_images, fake_labels, fake_groups = [], [], []
    synthetic_images_residual, synthetic_labels_residual, synthetic_groups_residual = [], [], []
    synthetic_images_variational, synthetic_labels_variational, synthetic_groups_variational = [], [], []

    for subset in config.subsets:
        subset_parts = [part for part in (config.subset_root, subset) if part]

        # Load REAL samples
        real_path = os.path.join(config.data_path, *subset_parts, "real")
        imgs, lbls, grps = _load_images_from_path(
            real_path, extensions, 0, config.dataset_name
        )
        real_images.extend(imgs)
        real_labels.extend(lbls)
        real_groups.extend(grps)

        # Load FAKE/SPOOF samples
        fake_path = os.path.join(config.data_path, *subset_parts, "spoof")
        imgs, lbls, grps = _load_images_from_path(
            fake_path, extensions, 1, config.dataset_name
        )
        fake_images.extend(imgs)
        fake_labels.extend(lbls)
        fake_groups.extend(grps)

        # Load SYNTHETIC samples
        synthetic_path = os.path.join(
            config.synthetic_path_residual, "", config.synthetic_folder
        )
        imgs, lbls, grps = _load_images_from_path(
            synthetic_path, extensions, 2, config.dataset_name, True
        )
        synthetic_images_residual.extend(imgs)
        synthetic_labels_residual.extend(lbls)
        synthetic_groups_residual.extend(grps)

        synthetic_path = os.path.join(
            config.synthetic_path_variational, "", config.synthetic_folder
        )
        imgs, lbls, grps = _load_images_from_path(
            synthetic_path, extensions, 2, config.dataset_name, True
        )
        synthetic_images_variational.extend(imgs)
        synthetic_labels_variational.extend(lbls)
        synthetic_groups_variational.extend(grps)

    has_synthetic_data = len(synthetic_images_variational) > 0 and len(synthetic_images_residual) > 0

    return (
        real_images, np.array(real_labels), np.array(real_groups),
        fake_images, np.array(fake_labels), np.array(fake_groups),
        synthetic_images_residual, np.array(synthetic_labels_residual), np.array(synthetic_groups_residual),
        synthetic_images_variational, np.array(synthetic_labels_variational), np.array(synthetic_groups_variational),
        has_synthetic_data,
    )