import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

def show_fourier_spectrum(image, save_path: str = None):
    """Creates and shows a plot of the Fourier spectrum of an image.

    Args:
        image: The image to plot.
        save_path: The path to save the plot to. If None, the plot is shown.
    """
    # 2D FFT
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(
        1 + np.abs(f_transform_shifted)
    )  # log für bessere Sichtbarkeit

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    plt.figure(figsize=(6, 6))
    plt.imshow(magnitude_spectrum, cmap="gray")
    # plt.scatter(ccol, crow, color='red', s=50, label='Zentrum (Low-Frequency)')
    # plt.title("Fourier Magnitude Spectrum mit Zentrum")
    # plt.legend()
    plt.axis("off")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

rng = np.random.RandomState(42)

def plot_cv_indices(X, y, groups, train_idx, test_idx, excluded_idx=None, n_splits=5, ax=None, lw=10, fold_number=1, dataset_name="", title=""):
    """
    Visualizes the *actual* CV split used in training/testing.
    Keeps the same style as sklearn's cross-validation plot example.

    Parameters
    ----------
    X : array-like, shape (n_samples, ...)
        Feature matrix (used only for length).
    y : array-like, shape (n_samples,)
        Class labels.
    groups : array-like, shape (n_samples,)
        Group identifiers.
    train_idx : array-like
        Indices of training samples for this fold.
    test_idx : array-like
        Indices of test samples for this fold.
    n_splits : int
        Total number of CV iterations (for consistent axis layout).
    ax : matplotlib axis
        Optional axis for plotting.
    lw : int
        Line width for markers.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))

    train_color = "#6699CC"  # lighter blue
    excluded_color = "#EAEAEA"  # very light gray
    test_color = "#CC6677"  # muted red

    # Create legend handles
    legend_elements = [
        Patch(facecolor=train_color, edgecolor='black', label='Training'),
        Patch(facecolor=test_color, edgecolor='black', label='Test'),
        Patch(facecolor=excluded_color, edgecolor='black', label='Excluded'),
    ]

    cmap_data = plt.cm.get_cmap("tab20c", 20)

    cmap_cv = ListedColormap(["#6699CC", "#EAEAEA", "#CC6677"])
    # cmap_cv = plt.cm.get_cmap("coolwarm")
    n_colors = 20

    n_samples = len(X)

    # initialize all indices as NaN (unassigned)
    indices = np.full(n_samples, np.nan)
    indices[train_idx] = 0  # training samples
    indices[test_idx] = 1   # test samples
    if excluded_idx is not None:
        indices[excluded_idx] = 0.5  # mid-gray for excluded samples

    # Draw the fold split (one row)
    ax.scatter(
        range(n_samples),
        [fold_number + 0.5] * n_samples,
        c=indices,
        marker="_",
        lw=lw,
        cmap=cmap_cv,
        vmin=-0.2,
        vmax=1.2,
    )

    if fold_number == n_splits-1:

        # Class color row
        ax.scatter(
            range(n_samples),
            [n_splits + 0.5] * n_samples,
            c=y,
            marker="_",
            lw=lw,
            cmap=cmap_data,
        )

        # Group color row
        ax.scatter(
            range(n_samples),
            [n_splits + 1.5] * n_samples,
            c=np.mod(groups, n_colors),
            marker="_",
            lw=lw,
            cmap=cmap_data,
        )

        # Axis formatting (keep original look)
        yticklabels = [f"Fold {i+1}" for i in range(n_splits)] + ["class", "group"]
        ax.set(
            yticks=np.arange(n_splits + 2) + 0.5,
            yticklabels=yticklabels,
            xlabel="Sample index",
            ylim=[n_splits + 2.2, -0.2],
            xlim=[0, n_samples],
        )

        legend_elements = [
            Patch(facecolor=train_color, edgecolor='black', label='Training'),
            Patch(facecolor=test_color, edgecolor='black', label='Test'),
        ]
        if excluded_idx is not None:
            legend_elements.append(
                Patch(facecolor=excluded_color, edgecolor='black', label='Excluded')
            )
        ax.legend(handles=legend_elements, loc='upper center',bbox_to_anchor=(0.5, -0.15),ncol=len(legend_elements), frameon=True, title="Legend")

        ax.set_title(f"{dataset_name}: {title}", fontsize=15)


    return ax

def plot_feature_vectors_as_spiderchart(feature_vectors: list, labels: list = None, save_path: str = None, thirds_lines: bool = False):
    '''
    Plots each feature vector overlapped in different colors as a spider/radar chart.

    Args:
        feature_vectors: List of feature vectors to plot
        labels: Optional list of labels for each feature vector (e.g., ['Magnitude', 'Phase'])
        save_path: Optional path to save the plot. If None, the plot is shown.
    '''
    num_vars = len(feature_vectors[0])
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # complete the loop

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Default labels if none provided
    if labels is None:
        labels = ['Magnitude', 'Phase'] if len(feature_vectors) == 2 else [f'Vector {i + 1}' for i in
                                                                           range(len(feature_vectors))]
    
    for i, vector in enumerate(feature_vectors):
        values = vector.tolist()
        values += values[:1]  # close loop
        color = color_cycle[i % len(color_cycle)]
        label = labels[i] if i < len(labels) else f'Vector {i + 1}'
        ax.plot(angles, values, linewidth=1.5, linestyle='solid', color=color, alpha=0.7, label=label)
        ax.fill(angles, values, color=color, alpha=0.2)

    if thirds_lines:
        split_features = [0, 10, 20]
        r0, r1 = 0, max(feature_vectors[0]) + 0.03

        for i in split_features:
            theta = angles[i]
            ax.plot([theta, theta], [r0, r1], linewidth=3, color="red", alpha=1)   # add alpha=... if you want

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"F{i + 1}" for i in range(num_vars)])
    #ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    ax.legend(
        loc="center",                 # anchor point of the legend box
        bbox_to_anchor=(1.055, 0.75),   # (x, y) in figure coords (0–1)
        bbox_transform=fig.transFigure,
    )

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_heatmap(data, output_path, title="", xlabel="", ylabel=""):
    """Generates and saves a heatmap.

    Args:
        data (dict): A dictionary where keys are tuples of (x, y) and values are the data to plot.
        output_path (str): The path to save the heatmap to.
        title (str, optional): The title of the heatmap. Defaults to "".
        xlabel (str, optional): The label for the x-axis. Defaults to "".
        ylabel (str, optional): The label for the y-axis. Defaults to "".
    """
    x_labels = sorted(set(x for x, _ in data.keys()))
    y_labels = sorted(set(y for _, y in data.keys()))

    acer_matrix = np.full((len(y_labels), len(x_labels)), np.nan)
    for (x, y), value in data.items():
        i = y_labels.index(y)
        j = x_labels.index(x)
        acer_matrix[i, j] = value

    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.imshow(acer_matrix, cmap='viridis', vmin=0.07, vmax=0.37)
    
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            val = acer_matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val > 0.25 else "black")

    plt.colorbar(c, ax=ax, label="ACER")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_feature_distribution(features_list, labels, output_path, title=""):
    """Visualizes the feature distributions for a given dataset."""
    data = {
        "label": [],
        "feature": [],
        "value": [],
    }
    for i, features in enumerate(features_list):
        for j, value in enumerate(features):
            data["label"].append("real" if labels[i] == 0 else "spoof")
            data["feature"].append(f"feature_{j}")
            data["value"].append(value)

    plt.figure(figsize=(20, 10))
    sns.boxplot(x="feature", y="value", hue="label", data=data)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_variance_violin(
    data_records,
    output_path,
    title="",
    hue=None,
    col=None,
    col_wrap=4,
    order=None,
    hue_order=None,
    col_order=None,
):
    """Plot variance distributions as violins with optional faceting."""
    if not data_records:
        print("Warning: No variance data provided for plotting.")
        return

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("plot_variance_violin requires pandas to be installed.") from exc

    df = pd.DataFrame(data_records)
    if df.empty:
        print("Warning: Variance dataframe is empty; skipping plot.")
        return

    plot_kwargs = dict(
        data=df,
        x="distance",
        y="dataset",
        hue=hue,
        kind="violin",
        order=order,
        hue_order=hue_order,
        cut=0,
        inner="quartile",
        density_norm="width",
        height=3.2,
        aspect=1.2,
        sharey=True,
        orient="h",
    )
    if col:
        plot_kwargs.update(col=col, col_wrap=col_wrap, col_order=col_order)

    g = sns.catplot(**plot_kwargs)
    g.set_axis_labels("L2 distance", "Dataset")
    if col:
        g.set_titles("{col_name}")
    if title:
        g.fig.suptitle(title, y=1.03)

    g.fig.tight_layout()
    g.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(g.fig)

def plot_two_feature_vectors_as_spiderchart(feature_vectors: list[list], labels: list = None, save_path: str = None):
    '''
    Plots each feature vector overlapped in different colors as a spider/radar chart.

    Args:
        feature_vectors: List of feature vectors to plot
        labels: Optional list of labels for each feature vector (e.g., ['Magnitude', 'Phase'])
        save_path: Optional path to save the plot. If None, the plot is shown.
    '''
    num_vars = len(feature_vectors[0])
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # complete the loop

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Default labels if none provided
    if labels is None:
        labels = ['Magnitude', 'Phase'] if len(feature_vectors) == 2 else [f'Vector {i + 1}' for i in
                                                                           range(len(feature_vectors))]

    for i, vector in enumerate(feature_vectors):
        values = vector.tolist()
        values += values[:1]  # close loop
        color = color_cycle[i % len(color_cycle)]
        label = labels[i] if i < len(labels) else f'Vector {i + 1}'
        ax.plot(angles, values, linewidth=1.5, linestyle='solid', color=color, alpha=0.7, label=label)
        ax.fill(angles, values, color=color, alpha=0.2)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"F{i + 1}" for i in range(num_vars)])
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
