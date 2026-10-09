from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

matplotlib.use("agg")


def _require_plotnine():
    from plotnine import aes, geom_boxplot, geom_point, ggplot, ggtitle

    return aes, geom_boxplot, geom_point, ggplot, ggtitle


def _require_umap():
    import umap

    return umap


def umapPlot_latent_check(real_latent, fake_latent, path_file_save=None):
    """
    Creates a UMAP plot to visually compare real and fake latent embeddings. This function reduces the dimensionality
    of the concatenated real and fake latent vectors and plots them in two-dimensional space, with points colored by
    category.

    Parameters:
    - real_latent (np.array): Array of latent vectors representing real data.
    - fake_latent (np.array): Array of latent vectors representing fake data.
    - path_file_save (str, optional): File path to save the generated UMAP plot image. If None, the plot will not be
      saved. Defaults to None.

    Returns:
    - chart_pr (ggplot): A ggplot object representing the UMAP plot, with points colored by 'Real' or 'Fake' category.

    """

    all_latent = np.concatenate([fake_latent, real_latent], axis=0)
    cat_t = ["Real"] * real_latent.shape[0]
    cat_g = ["Fake"] * fake_latent.shape[0]
    cat_rf_gt = np.append(cat_g, cat_t)
    umap = _require_umap()
    aes, _geom_boxplot, geom_point, ggplot, ggtitle = _require_plotnine()
    trans = umap.UMAP(random_state=42, min_dist=0.5, n_neighbors=30).fit(all_latent)
    X_embedded_pr = trans.transform(all_latent)
    df = X_embedded_pr.copy()
    df = pd.DataFrame(df)
    df["x-umap"] = X_embedded_pr[:, 0]
    df["y-umap"] = X_embedded_pr[:, 1]
    df["category"] = cat_rf_gt

    chart_pr = (
        ggplot(df, aes(x="x-umap", y="y-umap", colour="category"))
        + geom_point(size=0.5, alpha=0.5)
        + ggtitle("UMAP dimensions")
    )

    if path_file_save is not None:
        chart_pr.save(path_file_save, width=12, height=8, dpi=144)
    return chart_pr


def boxplot_metrics(model_dict, metric_key, path_file_save=None):
    """
    Generates a box plot to compare a specified metric across multiple models. The function takes a dictionary of
    model names and their corresponding DataFrames containing performance metrics, then plots the metric of interest
    for each model side-by-side in a box plot format.

    Parameters:
    - model_dict (dict): Dictionary where keys are model names and values are DataFrames containing model metrics. Each
      DataFrame should include columns for 'perturbation' and the specified `metric_key`.
    - metric_key (str): Name of the metric column to be used for the box plot comparison across models.
    - path_file_save (str, optional): File path to save the generated box plot image. If None, the plot will not be
      saved. Defaults to None.

    Returns:
    - chart_pr (ggplot): A ggplot object representing the box plot for the specified metric across models.

    """

    shared_cols = ["perturbation", metric_key]
    df_list = []
    for model, results in model_dict.items():
        results = results[shared_cols]
        results["model"] = np.repeat(model, results.shape[0])
        df_list.append(results)
    df = pd.concat(df_list, ignore_index=True)

    aes, geom_boxplot, _geom_point, ggplot, _ggtitle = _require_plotnine()
    chart_pr = ggplot(df, aes(x="model", y=metric_key, fill="model")) + geom_boxplot()
    if path_file_save is not None:
        chart_pr.save(path_file_save, width=12, height=8, dpi=144)
    return chart_pr


def contourplot_space_mapping(
    embeddings_cell,
    embeddings_pert,
    background_pert,
    background_cell,
    highlight_labels,
    colors,
    random_state=42,
    bandwidth=0.2,
    save_path=None,
    dpi=300,
    figsize=(12, 6),
    save_embed=False,
    save_path_embed=None,
    Y_embedded=None,
    Z_embedded=None,
):
    """
    Generates a contour plot mapping of embeddings for perturbation and cellular representations. The function uses
    UMAP for dimensionality reduction and visualizes both background and highlighted embeddings, optionally saving the
    embeddings and plot.

    Parameters:
    - embeddings_cell (np.array): Array of embeddings representing cell data to be plotted with highlights.
    - embeddings_pert (np.array): Array of embeddings representing perturbation data to be plotted with highlights.
    - background_pert (np.array): Background data for perturbation embeddings to provide additional context in the plot.
    - background_cell (np.array): Background data for cell embeddings to provide additional context in the plot.
    - highlight_labels (list): List of labels to highlight in the plot, differentiating these points from the background.
    - colors (list): List of colors corresponding to each highlight label for visual distinction.
    - random_state (int, optional): Seed for UMAP dimensionality reduction to ensure reproducibility. Defaults to 42.
    - bandwidth (float, optional): Bandwidth parameter for Gaussian KDE used in contouring highlighted points. Defaults to 0.2.
    - save_path (str, optional): Path to save the generated plot image. If None, the plot will not be saved. Defaults to None.
    - dpi (int, optional): Dots per inch for the saved plot image. Only used if save_path is specified. Defaults to 300.
    - figsize (tuple, optional): Size of the figure (width, height) in inches. Defaults to (12, 6).
    - save_embed (bool, optional): Whether to save the UMAP-transformed embeddings to files. Defaults to False.
    - save_path_embed (str, optional): Path to save UMAP-transformed embeddings if save_embed is True. Defaults to None.
    - Y_embedded (np.array, optional): Precomputed UMAP embeddings for perturbation data. If None, embeddings are computed.
    - Z_embedded (np.array, optional): Precomputed UMAP embeddings for cell data. If None, embeddings are computed.

    """

    embeddings_cell_all = np.concatenate([background_cell, embeddings_cell])
    embeddings_pert_all = np.concatenate([background_pert, embeddings_pert])

    cat_pert = ["Other"] * background_pert.shape[0] + [
        label
        for label in highlight_labels
        for _ in range(embeddings_pert.shape[0] // len(highlight_labels))
    ]
    cat_cell = ["Other"] * background_cell.shape[0] + [
        label
        for label in highlight_labels
        for _ in range(embeddings_cell.shape[0] // len(highlight_labels))
    ]

    # Create UMAP transformers and transform data
    if Y_embedded == None and Z_embedded == None:
        umap = _require_umap()
        trans_pert = umap.UMAP(
            random_state=random_state, min_dist=0.5, n_neighbors=30
        ).fit(embeddings_pert_all)
        trans_cell = umap.UMAP(
            random_state=random_state, min_dist=0.5, n_neighbors=30
        ).fit(embeddings_cell_all)
        Y_embedded = trans_pert.transform(embeddings_pert_all)
        if save_embed:
            np.save(save_path_embed + "Y_embedded.npy", Y_embedded)
        Z_embedded = trans_cell.transform(embeddings_cell_all)
        if save_embed:
            np.save(save_path_embed + "Z_embedded.npy", Z_embedded)

    fig = plt.figure(figsize=(12, 6))

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    # Define a plotting function for each subplot
    def plot_with_contours(
        ax, data, categories, title, highlights, colors, add_contour=True
    ):
        for highlight, color in zip(highlights, colors):
            highlight_data = data[categories == highlight]
            other_data = data[categories != highlight]

            # Plot background data
            ax.scatter(
                other_data[:, 0], other_data[:, 1], color="gray", s=1, label="Other"
            )

            # Plot highlight data
            ax.scatter(
                highlight_data[:, 0],
                highlight_data[:, 1],
                color=color,
                s=1,
                label=highlight,
            )

            if add_contour and highlight_data.size > 0:
                x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
                y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])
                x_grid = np.linspace(x_min, x_max, 100)
                y_grid = np.linspace(y_min, y_max, 100)
                X, Y = np.meshgrid(x_grid, y_grid)
                kde = gaussian_kde(highlight_data.T, bw_method=bandwidth)
                Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                ax.contour(X, Y, Z, levels=5, colors=color)

        ax.set_xlim(np.min(data[:, 0]), np.max(data[:, 0]))
        ax.set_ylim(np.min(data[:, 1]), np.max(data[:, 1]))
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    # Plotting for each representation
    plot_with_contours(
        ax1,
        Y_embedded,
        np.array(cat_pert),
        "Perturbation Representation",
        highlight_labels,
        colors,
        add_contour=False,
    )
    plot_with_contours(
        ax2,
        Z_embedded,
        np.array(cat_cell),
        "Cellular Representation",
        highlight_labels,
        colors,
    )

    # Draw lines for highlighted points
    transFigure = fig.transFigure.inverted()
    highlight_indices = np.where(np.isin(np.array(cat_pert), highlight_labels))[0]

    if len(highlight_indices) > 30:
        highlight_indices = np.random.choice(
            highlight_indices, 30, replace=False
        )  # Randomly pick 30 indices

    pert_indices = highlight_indices
    if background_cell.shape[0] > background_pert.shape[0]:
        Y_indices = (
            highlight_indices - background_cell.shape[0] + background_pert.shape[0]
        )
        Z_indices = highlight_indices

    elif background_cell.shape[0] < background_pert.shape[0]:
        Z_indices = (
            highlight_indices - background_pert.shape[0] + background_cell.shape[0]
        )
        Y_indices = highlight_indices
    else:
        Y_indices = highlight_indices
        Z_indices = highlight_indices

    for i in range(len(highlight_indices)):
        index = highlight_indices[i]
        xy1 = transFigure.transform(ax1.transData.transform(Y_embedded[Y_indices[i]]))
        xy2 = transFigure.transform(ax2.transData.transform(Z_embedded[Z_indices[i]]))
        line_color = colors[highlight_labels.index(cat_pert[pert_indices[i]])]
        line = matplotlib.lines.Line2D(
            (xy1[0], xy2[0]),
            (xy1[1], xy2[1]),
            transform=fig.transFigure,
            color=line_color,
            linewidth=0.5,
        )
        fig.lines.append(line)

    handles1, labels1 = ax1.get_legend_handles_labels()
    unique_handles1, unique_labels1 = [], []
    for handle, label in zip(handles1, labels1):
        if label not in unique_labels1:
            unique_handles1.append(handle)
            unique_labels1.append(label)
    ax1.legend(unique_handles1, unique_labels1, loc="lower left")

    handles2, labels2 = ax2.get_legend_handles_labels()
    unique_handles2, unique_labels2 = [], []
    for handle, label in zip(handles2, labels2):
        if label not in unique_labels2:
            unique_handles2.append(handle)
            unique_labels2.append(label)
    ax2.legend(unique_handles2, unique_labels2, loc="lower right")

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()
