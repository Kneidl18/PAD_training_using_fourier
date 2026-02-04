from typing import Any, Callable, Tuple, List, Dict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier

from src.calculate_metrics import metrics, mean_tuple
from src.plotting import plot_cv_indices #TODO: implement here


def _evaluate_step3(X_real, y_real, g_real, X_fake, y_fake, g_fake, X_synthetic, y_synthetic, g_synthetic, X_test, y_test, train_users_full, target_users_1_5, target_users_2_5, n_neighbors, attack_label, random_state):
    rng = np.random.default_rng(random_state)
    n_users_1_5 = min(len(train_users_full), target_users_1_5)
    users_1_5 = rng.choice(train_users_full, size=n_users_1_5, replace=False)

    n_users_2_5 = min(len(train_users_full), target_users_2_5)
    users_2_5 = rng.choice(train_users_full, size=n_users_2_5, replace=False)
    users_additional = np.setdiff1d(users_2_5, users_1_5)

    mask_1_5_real = np.isin(g_real, users_1_5)
    mask_1_5_fake = np.isin(g_fake, users_1_5)
    X_train_step3_base = np.vstack([X_real[mask_1_5_real], X_fake[mask_1_5_fake]])
    y_train_step3_base = np.concatenate([y_real[mask_1_5_real], y_fake[mask_1_5_fake]])

    mask_add_real = np.isin(g_real, users_additional)
    mask_add_synth = np.isin(g_synthetic, users_additional)
    X_train_step3_aug = np.vstack([X_real[mask_add_real], X_synthetic[mask_add_synth]])
    y_train_step3_aug = np.concatenate([y_real[mask_add_real], y_synthetic[mask_add_synth]])

    X_train_step3 = np.vstack([X_train_step3_base, X_train_step3_aug])
    y_train_step3 = np.concatenate([y_train_step3_base, y_train_step3_aug])

    # TODO: check
    # replace label 2 with attack_label 
    # ?is this correct?
    y_train_step3[y_train_step3 == 2] = attack_label

    clf3 = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf3.fit(X_train_step3, y_train_step3)
    preds3 = clf3.predict(X_test)

    return metrics(y_test, preds3, attack_label)


def _evaluate_step4(X_real, y_real, g_real, X_fake, y_fake, g_fake, X_synthetic, y_synthetic, g_synthetic, X_test, y_test, train_users_full, target_users_1_5, n_neighbors, attack_label, random_state):
    rng = np.random.default_rng(random_state)
    n_users_1_5 = min(len(train_users_full), target_users_1_5)
    users_1_5 = rng.choice(train_users_full, size=n_users_1_5, replace=False)

    users_remaining_3_5 = np.setdiff1d(train_users_full, users_1_5)

    mask_1_5_real = np.isin(g_real, users_1_5)
    mask_1_5_fake = np.isin(g_fake, users_1_5)
    X_train_step4_base = np.vstack([X_real[mask_1_5_real], X_fake[mask_1_5_fake]])
    y_train_step4_base = np.concatenate([y_real[mask_1_5_real], y_fake[mask_1_5_fake]])

    mask_3_5_real = np.isin(g_real, users_remaining_3_5)
    mask_3_5_synth = np.isin(g_synthetic, users_remaining_3_5)
    X_train_step4_aug = np.vstack([X_real[mask_3_5_real], X_synthetic[mask_3_5_synth]])
    y_train_step4_aug = np.concatenate([y_real[mask_3_5_real], y_synthetic[mask_3_5_synth]])

    X_train_step4 = np.vstack([X_train_step4_base, X_train_step4_aug])
    y_train_step4 = np.concatenate([y_train_step4_base, y_train_step4_aug])

    y_train_step4[y_train_step4 == 2] = attack_label

    clf4 = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf4.fit(X_train_step4, y_train_step4)
    preds4 = clf4.predict(X_test)

    return metrics(y_test, preds4, attack_label)


def _evaluate_step5(X_real, y_real, g_real, X_synthetic, y_synthetic, g_synthetic, X_test, y_test, train_users_full, n_neighbors, attack_label):
    mask_train_real = np.isin(g_real, train_users_full)
    mask_train_synth = np.isin(g_synthetic, train_users_full)
    X_train_step5 = np.vstack([X_real[mask_train_real], X_synthetic[mask_train_synth]])
    y_train_step5 = np.concatenate([y_real[mask_train_real], y_synthetic[mask_train_synth]])

    y_train_step5[y_train_step5 == 2] = attack_label

    clf5 = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf5.fit(X_train_step5, y_train_step5)
    preds5 = clf5.predict(X_test)

    return metrics(y_test, preds5, attack_label)


def evaluate_task2(
        real_images: List[np.ndarray],
        real_labels: np.ndarray,
        real_groups: np.ndarray,
        fake_images: List[np.ndarray],
        fake_labels: np.ndarray,
        fake_groups: np.ndarray,
        synthetic_images_residual: List[np.ndarray],
        synthetic_labels_residual: np.ndarray,
        synthetic_groups_residual: np.ndarray,
        synthetic_images_variational: List[np.ndarray],
        synthetic_labels_variational: np.ndarray,
        synthetic_groups_variational: np.ndarray,
        feature_extractor_func: Callable[[np.ndarray], np.ndarray],
        n_neighbors: int,
        attack_label: int = 1,
        random_state: int = 42,
        plot_splits: bool = True,
) -> tuple[dict[str, tuple[float, float, float]], dict[str, tuple[float, float, float]]]:
    """Performs the Task 2 evaluation with synthetic PAI samples.

    This function implements Steps 3, 4, and 5 from the Task 2 instructions:
      - Step 3: 1/5 Real+Fake + 1/5 Real+Synthetic (size of Step 1 from Task 1)
      - Step 4: 1/5 Real+Fake + 3/5 Real+Synthetic (size of baseline)
      - Step 5: 4/5 Real+Synthetic (no fake samples in training)

    Important:
    - The test set always consists of Real + Fake samples (never synthetic).
    - The same folds as in Task 1 are used (same random_state).
    - The reduction is user-based (entire subjects).

    Args:
        real_images: A list of real/bona fide images.
        real_labels: The labels for the real images (should all be 0).
        real_groups: The subject IDs for the real images.
        fake_images: A list of fake/spoof images.
        fake_labels: The labels for the fake images (should all be 1).
        fake_groups: The subject IDs for the fake images.
        synthetic_images_residual: A list of synthetic images.
        synthetic_labels_residual: The labels for the synthetic images (should all be 1).
        synthetic_groups_residual: The subject IDs for the synthetic images.
        synthetic_images_variational: A list of synthetic images.
        synthetic_labels_variational: The labels for the synthetic images (should all be 1).
        synthetic_groups_variational: The subject IDs for the synthetic images.
        feature_extractor_func: The feature extraction function.
        n_neighbors: The number of neighbors for k-NN.
        attack_label: The label for attack samples (default: 1).
        random_state: The random seed for reproducibility (default: 42).
        plot_splits: Whether to plot the cross-validation splits (default: True).

    Returns:
        A dictionary with the results for Step 3, 4, and 5.
    """

    rng = np.random.default_rng(random_state)

    print("  Extracting features for real samples...")
    X_real = np.vstack([feature_extractor_func(img) for img in real_images])

    print("  Extracting features for fake samples...")
    X_fake = np.vstack([feature_extractor_func(img) for img in fake_images])

    print("  Extracting features for synthetic samples...")
    X_synthetic_residual = np.vstack([feature_extractor_func(img) for img in synthetic_images_residual])
    X_synthetic_variational = np.vstack([feature_extractor_func(img) for img in synthetic_images_variational])

    y_real = np.array(real_labels)
    g_real = np.array(real_groups)
    y_fake = np.array(fake_labels)
    g_fake = np.array(fake_groups)
    y_synthetic_residual = np.array(synthetic_labels_residual)
    g_synthetic_residual = np.array(synthetic_groups_residual)
    y_synthetic_variational = np.array(synthetic_labels_variational)
    g_synthetic_variational = np.array(synthetic_groups_variational)


    # 2) Kombiniere Real + Fake für Test-Set (wie in Task 1)
    X_real_fake = np.vstack([X_real, X_fake])
    y_real_fake = np.concatenate([y_real, y_fake])
    g_real_fake = np.concatenate([g_real, g_fake])

    # Gesamtzahl an Usern (basierend auf Real+Fake wie in Task 1)
    total_users = len(np.unique(g_real_fake))
    target_users_2_5 = int(round(total_users * 0.40))  # 2/5 gesamt
    target_users_1_5 = int(round(total_users * 0.20))  # 1/5 gesamt

    # 3) 5-Fold-Partition der USER (GLEICHE wie in Task 1!)
    results_residual = _perform_cross_validation_task2(X_real_fake, y_real_fake, g_real_fake, X_real, y_real, g_real, X_fake, y_fake, g_fake, X_synthetic_residual, y_synthetic_residual, g_synthetic_residual, n_neighbors, attack_label, random_state, plot_splits)
    results_variational = _perform_cross_validation_task2(X_real_fake, y_real_fake, g_real_fake, X_real, y_real, g_real, X_fake, y_fake, g_fake, X_synthetic_variational, y_synthetic_variational, g_synthetic_variational, n_neighbors, attack_label, random_state, plot_splits)

    return results_residual, results_variational


def _perform_cross_validation_task2(
    X_real_fake: np.ndarray, y_real_fake: np.ndarray, g_real_fake: np.ndarray,
    X_real: np.ndarray, y_real: np.ndarray, g_real: np.ndarray,
    X_fake: np.ndarray, y_fake: np.ndarray, g_fake: np.ndarray,
    X_synthetic: np.ndarray, y_synthetic: np.ndarray, g_synthetic: np.ndarray,
    n_neighbors: int, attack_label: int, random_state: int,
    plot_splits: bool = True
) -> Dict[str, Tuple[float, float, float]]:
    """Performs 5-fold cross-validation for task 2 and returns the results."""
    total_users = len(np.unique(g_real_fake))
    target_users_2_5 = int(round(total_users * 0.40))  # 2/5 gesamt
    target_users_1_5 = int(round(total_users * 0.20))  # 1/5 gesamt

    gkf = GroupKFold(n_splits=5)

    res_step3, res_step4, res_step5 = [], [], []

    for fold_idx, (train_idx_full, test_idx) in enumerate(gkf.split(X_real_fake, y_real_fake, groups=g_real_fake)):
        print(f"  Processing fold {fold_idx + 1}/5...")

        X_test = X_real_fake[test_idx]
        y_test = y_real_fake[test_idx]

        train_users_full = np.unique(g_real_fake[train_idx_full])

        # ---- Step 3: 1/5 Real+Fake + 1/5 Real+Synthetic ----
        res_step3.append(_evaluate_step3(X_real, y_real, g_real, X_fake, y_fake, g_fake, X_synthetic, y_synthetic, g_synthetic, X_test, y_test, train_users_full, target_users_1_5, target_users_2_5, n_neighbors, attack_label, random_state))

        # ---- Step 4: 1/5 Real+Fake + 3/5 Real+Synthetic ----
        res_step4.append(_evaluate_step4(X_real, y_real, g_real, X_fake, y_fake, g_fake, X_synthetic, y_synthetic, g_synthetic, X_test, y_test, train_users_full, target_users_1_5, n_neighbors, attack_label, random_state))

        # ---- Step 5: 4/5 Real+Synthetic (KEINE Fake Samples!) ----
        res_step5.append(_evaluate_step5(X_real, y_real, g_real, X_synthetic, y_synthetic, g_synthetic, X_test, y_test, train_users_full, n_neighbors, attack_label))

    return {
        "Step 3": mean_tuple(res_step3),
        "Step 4": mean_tuple(res_step4),
        "Step 5": mean_tuple(res_step5),
    }