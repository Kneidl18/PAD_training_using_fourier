import os
import sys
import numpy as np
from typing import Any

from src.data_utils import filter_users_with_both_classes, DatasetConfig
from src.evaluate_task1 import evaluate_task1
from src.evaluate_task2 import evaluate_task2
from src.fourier_feature_extraction import (
    format_task2_thirds,
    make_thirds_feature_extractor,
    normalize_task2_thirds,
)
from src.get_data import get_data, get_data_with_synthetic
from src.cli import parse_args
from src.config import get_dataset_configs

from dotenv import load_dotenv

load_dotenv(".env")

def _create_plots_dir():
    """Creates the plots directory if it doesn't exist."""
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

def _run_task1_evaluation(
    name: str,
    images: list,
    labels: np.ndarray,
    groups: np.ndarray,
    dataset_config: DatasetConfig,
    args: Any,
):
    """Runs Task 1 evaluation for a single dataset."""
    print(f"--- Dataset: {name} ---")
    if len(images) == 0:
        print("No images found, skipping.")
        return

    print(f"Found total of {len(np.unique(groups))} users")

    mask = filter_users_with_both_classes(labels, groups)
    images = [img for img, keep in zip(images, mask) if keep]
    labels = labels[mask]
    groups = groups[mask]

    print(f"{len(np.unique(groups))} users with real & spoof data ({mask.sum()} Samples)")
    print()

    feature_modes = [
        ("Magnitude-only", "magnitude"),
        ("Phase-only", "phase"),
        ("Combined (Magnitude + Phase)", "combined"),
    ]

    for config_name, feature_mode in feature_modes:
        print(f"=== {config_name} Features ===")

        results = evaluate_task1(
            images,
            labels,
            groups,
            dataset_config.n_neighbors,
            feature_mode=feature_mode,
            print_corr=args.print_corr
        )

        print(
            f"Baseline (4/5): APCER: {results.get('4/5')[0]:.4f} | BPCER: {results.get('4/5')[1]:.4f} | ACER: {results.get('4/5')[2]:.4f}"
        )
        print(
            f"Step 1   (2/5): APCER: {results.get('2/5')[0]:.4f} | BPCER: {results.get('2/5')[1]:.4f} | ACER: {results.get('2/5')[2]:.4f}"
        )
        print(
            f"Step 2   (1/5): APCER: {results.get('1/5')[0]:.4f} | BPCER: {results.get('1/5')[1]:.4f} | ACER: {results.get('1/5')[2]:.4f}"
        )
        print()

def _run_task2_evaluation(
    name: str,
    dataset_config: DatasetConfig,
    args: Any,
):
    """Runs Task 2 evaluation for a single dataset."""
    print(f"--- Dataset: {name} ---")

    (real_images, real_labels, real_groups,
     fake_images, fake_labels, fake_groups,
     synthetic_images_residual, synthetic_labels_residual, synthetic_groups_residual,
     synthetic_images_variational, synthetic_labels_variational, synthetic_groups_variational,
     has_synthetic_data) = get_data_with_synthetic(dataset_config)

    if len(real_images) == 0 or len(fake_images) == 0:
        print("No real or fake images found, skipping.")
        print()
        return

    if not has_synthetic_data or len(synthetic_images_residual) == 0 or len(synthetic_images_variational) == 0:
        print("⚠️  WARNING: No synthetic data found!")
        print(f"   Expected locations: {os.path.join(dataset_config.synthetic_path_residual, dataset_config.synthetic_folder)} and {os.path.join(dataset_config.synthetic_path_variational, dataset_config.synthetic_folder)}")
        print()
        return

    print(f"Loaded {len(real_images)} real, {len(fake_images)} fake, {len(synthetic_images_residual)} residual synthetic samples and {len(synthetic_images_variational)} variational synthetic samples.")

    # Create a combined dataset from real and fake images to identify users with both classes
    all_images = real_images + fake_images
    all_labels = np.concatenate([real_labels, fake_labels])
    all_groups = np.concatenate([real_groups, fake_groups])

    # Generate a mask for users who have both real and fake samples
    user_mask = filter_users_with_both_classes(all_labels, all_groups)
    valid_users = np.unique(all_groups[user_mask])
    
    # Create a filter mask for each dataset based on the valid users
    real_mask = np.isin(real_groups, valid_users)
    fake_mask = np.isin(fake_groups, valid_users)
    synthetic_mask_residual = np.isin(synthetic_groups_residual, valid_users)
    synthetic_mask_variational = np.isin(synthetic_groups_variational, valid_users)
    
    # Apply the masks to filter the datasets
    real_images = [img for img, keep in zip(real_images, real_mask) if keep]
    real_labels = real_labels[real_mask]
    real_groups = real_groups[real_mask]
    
    fake_images = [img for img, keep in zip(fake_images, fake_mask) if keep]
    fake_labels = fake_labels[fake_mask]
    fake_groups = fake_groups[fake_mask]
    
    synthetic_images_residual = [img for img, keep in zip(synthetic_images_residual, synthetic_mask_residual) if keep]
    synthetic_labels_residual = synthetic_labels_residual[synthetic_mask_residual]
    synthetic_groups_residual = synthetic_groups_residual[synthetic_mask_residual]

    synthetic_images_variational = [img for img, keep in zip(synthetic_images_variational, synthetic_mask_variational) if keep]
    synthetic_labels_variational = synthetic_labels_variational[synthetic_mask_variational]
    synthetic_groups_variational = synthetic_groups_variational[synthetic_mask_variational]

    print(f"Filtered to {len(valid_users)} users with real & spoof data.")
    print(f"Samples: {len(real_images)} real, {len(fake_images)} fake, {len(synthetic_images_residual)} residual, {len(synthetic_images_variational)} variational.")
    print()

    task2_thirds = normalize_task2_thirds(dataset_config.task2_thirds)
    print(f"Using magnitude thirds: {format_task2_thirds(task2_thirds)}")

    feature_extractor_func = make_thirds_feature_extractor(task2_thirds)

    results_residual, results_variational = evaluate_task2(
        real_images, real_labels, real_groups,
        fake_images, fake_labels, fake_groups,
        synthetic_images_residual, synthetic_labels_residual, synthetic_groups_residual,
        synthetic_images_variational, synthetic_labels_variational, synthetic_groups_variational,
        feature_extractor_func,
        dataset_config.n_neighbors,
    )

    print(f"=== Residual Synthetic Features ===")
    results_task2 = results_residual
    print(
        f"Step 3 (1/5 Real+Fake + 1/5 Real+Synth): APCER: {results_task2.get('Step 3')[0]:.4f} | BPCER: {results_task2.get('Step 3')[1]:.4f} | ACER: {results_task2.get('Step 3')[2]:.4f}"
    )
    print(
        f"Step 4 (1/5 Real+Fake + 3/5 Real+Synth): APCER: {results_task2.get('Step 4')[0]:.4f} | BPCER: {results_task2.get('Step 4')[1]:.4f} | ACER: {results_task2.get('Step 4')[2]:.4f}"
    )
    print(
        f"Step 5 (4/5 Real+Synth, no Fakes):      APCER: {results_task2.get('Step 5')[0]:.4f} | BPCER: {results_task2.get('Step 5')[1]:.4f} | ACER: {results_task2.get('Step 5')[2]:.4f}"
    )
    print()

    print(f"=== Variational Synthetic Features ===")
    results_task2 = results_variational
    print(
        f"Step 3 (1/5 Real+Fake + 1/5 Real+Synth): APCER: {results_task2.get('Step 3')[0]:.4f} | BPCER: {results_task2.get('Step 3')[1]:.4f} | ACER: {results_task2.get('Step 3')[2]:.4f}"
    )
    print(
        f"Step 4 (1/5 Real+Fake + 3/5 Real+Synth): APCER: {results_task2.get('Step 4')[0]:.4f} | BPCER: {results_task2.get('Step 4')[1]:.4f} | ACER: {results_task2.get('Step 4')[2]:.4f}"
    )
    print(
        f"Step 5 (4/5 Real+Synth, no Fakes):      APCER: {results_task2.get('Step 5')[0]:.4f} | BPCER: {results_task2.get('Step 5')[1]:.4f} | ACER: {results_task2.get('Step 5')[2]:.4f}"
    )
    print()


def main():
    """Main function."""
    args = parse_args()

    if args.plots:
        _create_plots_dir()

    if args.dataset == "ALL":
        dataset_names = ["PLUS", "IDIAP", "SCUT"]
    else:
        dataset_names = [args.dataset]

    try:
        dataset_configs = get_dataset_configs()
    except RuntimeError as err:
        print(f"Configuration error: {err}", file=sys.stderr)
        print(
            "Set REAL_DATA_ROOT and SYNTHETIC_DATA_ROOT_RESIDUAL and SYNTHETIC_DATA_ROOT_VARIATIONAL in your .env file.",
            file=sys.stderr,
        )
        return 1

    # Determine which tasks to run based on --task flag
    run_task1 = args.task in ["1", "both"]
    run_task2 = args.task in ["2", "both"]

    # Load data for Task 1 if needed
    datasets = {}
    if run_task1:
        for name in dataset_names:
            datasets[name] = get_data(dataset_configs[name])

    # Run Task 1 if requested
    if run_task1:
        print("\n" + "="*60)
        print("TASK 1: Baseline Evaluation (Real + Fake only)")
        print("="*60 + "\n")

        for name, (images, labels, groups) in datasets.items():
            _run_task1_evaluation(name, images, labels, groups, dataset_configs[name], args)

    # Run Task 2 if requested
    if run_task2:
        print("\n" + "="*60)
        print("TASK 2: Synthetic PAI Sample Integration")
        print("="*60 + "\n")

        for name in dataset_names:
            _run_task2_evaluation(name, dataset_configs[name], args)

    return None


if __name__ == "__main__":
    main()
