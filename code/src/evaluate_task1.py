from typing import Any, Callable, Tuple, List
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier

from src.plotting import plot_cv_indices
from src.calculate_metrics import mean_tuple, metrics, calculate_correlation
from src.fourier_feature_extraction import extract_fourier_features, get_feature_extractor


def evaluate_task1(
    images: List[np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    n_neighbors: int,
    attack_label: int = 1,
    random_state: int = 42,
    feature_mode: str = "combined",
    print_corr: bool = False,
    plot_splits: bool = False,
    **feature_kwargs,
) -> dict[str, tuple[Any]]:
    """Performs a 5-fold cross-validation evaluation of the feature extractor.

    The evaluation is performed in three variants:
      - '4/5': Training on 4/5 of the users, testing on 1/5 (baseline).
      - '2/5': Same as '4/5', but training is reduced to 2/5 of the total users.
      - '1/5': Same as '4/5', but training is reduced to 1/5 of the total users.

    The reduction is user-based, so the test users remain the same in all variants.

    Args:
        images: The images to evaluate.
        labels: The labels of the images.
        groups: The groups of the images.
        n_neighbors: The number of neighbors to use for k-NN.
        attack_label: The label of the attack class.
        random_state: The random state to use for the evaluation.
        feature_mode: The feature mode to use.
        print_corr: Whether to print the correlation between magnitude and phase features.
        **feature_kwargs: Additional keyword arguments for the feature extractor.

    Returns:
        A dictionary containing the results of the evaluation.
    """

    feature_extractor = get_feature_extractor(feature_mode, **feature_kwargs)

    features_list = []
    magnitude_list = []
    phase_list = []

    for img in images:
        if feature_mode == "combined":
            magnitude, phase = extract_fourier_features(img, **feature_kwargs)
            features_list.append(np.concatenate([magnitude, phase]))
            magnitude_list.append(magnitude)
            phase_list.append(phase)
        else:
            features_list.append(feature_extractor(img))

    if print_corr and feature_mode == "combined":
        corr = calculate_correlation(magnitude_list, phase_list)
        print(f"Correlation: {corr:.3f}")

    X = np.vstack(features_list)
    y = np.array(labels)
    g = np.array(groups)

    sort_idx = np.lexsort((y, g))
    X = X[sort_idx]
    y = y[sort_idx]
    g = g[sort_idx]

    return _perform_cross_validation(X, y, g, n_neighbors, attack_label, random_state, plot_splits)


def _perform_cross_validation(X, y, g, n_neighbors, attack_label, random_state, plot_splits: bool = True):
    """Performs 5-fold cross-validation and returns the results."""
    rng = np.random.default_rng(random_state)
    total_users = len(np.unique(g))
    target_users_2_5 = int(round(total_users * 0.40))  # 2/5 gesamt
    target_users_1_5 = int(round(total_users * 0.20))  # 1/5 gesamt

    gkf = GroupKFold(n_splits=5, shuffle=False)
    res_full, res_2_5, res_1_5 = [], [], []

    if plot_splits:
        # Create figures for the plots
        fig_full, ax_full = plt.subplots(figsize=(15, 5))
        fig_2_5, ax_2_5 = plt.subplots(figsize=(15, 5))
        fig_1_5, ax_1_5 = plt.subplots(figsize=(15, 5))

    for fold, (train_idx_full, test_idx) in enumerate(gkf.split(X, y, groups=g)):
        X_test, y_test, g_test = X[test_idx], y[test_idx], g[test_idx]

        # ---- Variante '4/5' (Baseline: voller Trainingsfold) ----
        X_train_full, y_train_full, g_train_full = (
            X[train_idx_full],
            y[train_idx_full],
            g[train_idx_full],
        )

        if np.intersect1d(np.unique(g[train_idx_full]), np.unique(g[test_idx])).size > 0:
            raise ValueError(f"Fold {fold + 1}: Train and test groups overlap!")

        if plot_splits:
            # Plot the '4/5' split
            plot_cv_indices(X, y, g, train_idx_full, test_idx, n_splits=5, ax=ax_full, fold_number=fold, title="4/5 Split")

        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(X_train_full, y_train_full)
        preds = clf.predict(X_test)
        res_full.append(metrics(y_test, preds, attack_label))

        # ---- Variante '2/5' (absolut 2/5 Gesamt-User) ----
        train_groups_full = np.unique(g_train_full)
        # pick 2/5 of the *total* unique groups that are present in the current train fold
        rng = np.random.default_rng(random_state)
        train_groups_2_5 = rng.choice(
            train_groups_full,
            size=min(target_users_2_5, len(train_groups_full)),
            replace=False
        )

        mask_train_2_5 = np.isin(g, train_groups_2_5)
        train_idx_2_5 = np.where(mask_train_2_5)[0]
        X_train_2_5, y_train_2_5 = X[train_idx_2_5], y[train_idx_2_5]

        if plot_splits:
            # PLOT
            excluded_idx_2_5 = np.setdiff1d(train_idx_full, train_idx_2_5)
            plot_cv_indices(X, y, g, train_idx_2_5, test_idx, excluded_idx=excluded_idx_2_5, n_splits=5, ax=ax_2_5,
                                     fold_number=fold, title="2/5 Split")

        clf2 = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf2.fit(X_train_2_5, y_train_2_5)
        preds2 = clf2.predict(X_test)
        res_2_5.append(metrics(y_test, preds2, attack_label))

        # ---- Variante '1/5' (absolut 1/5 Gesamt-User) ----
        train_groups_full = np.unique(g_train_full)
        # pick 1/5 of the *total* unique groups that are present in the current train fold
        train_groups_1_5 = rng.choice(
            train_groups_full,
            size=min(target_users_1_5, len(train_groups_full)),
            replace=False
        )

        mask_train_1_5 = np.isin(g, train_groups_1_5)
        train_idx_1_5 = np.where(mask_train_1_5)[0]
        X_train_1_5, y_train_1_5 = X[train_idx_1_5], y[train_idx_1_5]

        if plot_splits:
            # PLOT
            excluded_idx_1_5 = np.setdiff1d(train_idx_full, train_idx_1_5)
            plot_cv_indices(X, y, g, train_idx_1_5, test_idx, excluded_idx=excluded_idx_1_5, n_splits=5, ax=ax_1_5,
                                     fold_number=fold, title="1/5 Split")

        clf3 = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf3.fit(X_train_1_5, y_train_1_5)
        preds3 = clf3.predict(X_test)
        res_1_5.append(metrics(y_test, preds3, attack_label))

    if plot_splits:
        # Save the plots
        fig_full.savefig("cv_split_4_5.png")
        fig_2_5.savefig("cv_split_2_5.png")
        fig_1_5.savefig("cv_split_1_5.png")
        plt.close(fig_full)
        plt.close(fig_2_5)
        plt.close(fig_1_5)

    return {
        "4/5": mean_tuple(res_full),
        "2/5": mean_tuple(res_2_5),
        "1/5": mean_tuple(res_1_5),
    }