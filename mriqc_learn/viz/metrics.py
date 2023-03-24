# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2022 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Peeking into IQMs (image quality metrics)."""
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def plot_batches(X, cols=None, out_file=None, site_labels="left"):
    """Plot a matrix of images x metrix clustered by sites."""
    sort_by = ["site"]
    if "database" in X.columns:
        sort_by.insert(0, "database")

    if "rating" in X.columns:
        sort_by.append("rating")

    X = X.sort_values(by=sort_by).copy()
    sites = X.site.values.reshape(-1).tolist()

    # Select features
    numdata = X.select_dtypes([np.number]) if cols is None else X[cols]
    colmin = numdata.min()
    numdata = numdata - colmin
    colmax = numdata.max()
    numdata = numdata / colmax

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.imshow(
        numdata.values,
        cmap=plt.cm.viridis,
        interpolation="nearest",
        aspect="auto",
    )

    locations = []
    spines = []
    X["index"] = range(len(X))
    for site in list(set(sites)):
        indices = X.loc[X.site == site, "index"].values.reshape(-1).tolist()
        locations.append(int(np.average(indices)))
        spines.append(indices[0])

    if site_labels == "right":
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

    plt.xticks(
        range(numdata.shape[1]),
        numdata.columns,
        rotation="vertical",
    )
    plt.yticks(locations, list(set(sites)))
    for line in spines[1:]:
        plt.axhline(y=line, color="w", linestyle="-")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.grid(False)

    ticks_font = FontProperties(
        style="normal",
        size=14,
        weight="normal",
        stretch="normal",
    )
    for label in ax.get_yticklabels():
        label.set_fontproperties(ticks_font)

    ticks_font = FontProperties(
        style="normal",
        size=12,
        weight="normal",
        stretch="normal",
    )
    for label in ax.get_xticklabels():
        label.set_fontproperties(ticks_font)

    if out_file is not None:
        fig.savefig(out_file, bbox_inches="tight", pad_inches=0, dpi=300)
    return fig


def plot_histogram(X, X_scaled, metric=None):
    """Plot a histogram with different hues for different sites in X."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(20, 8), sharey=False, constrained_layout=True
    )
    fig.set_constrained_layout_pads(w_pad=0.1, wspace=0)

    colors = list(plt.cm.get_cmap("tab20").colors)
    colors = colors[::2] + colors[1::2]
    ax1.set_prop_cycle(color=colors)
    ax2.set_prop_cycle(color=colors)

    groups = Counter(X.site.values.squeeze().tolist()).most_common()
    nstd_iqm_min, nstd_iqm_max = X[metric].min(), X[metric].max()
    nstd_bin_sides = np.linspace(nstd_iqm_min, nstd_iqm_max, 101, endpoint=True)
    nstd_width = nstd_bin_sides[1] - nstd_bin_sides[0]
    nstd_bin_centers = nstd_bin_sides[:-1] + 0.5 * nstd_width

    std_iqm_min, std_iqm_max = X_scaled[metric].min(), X_scaled[metric].max()
    std_bin_sides = np.linspace(std_iqm_min, std_iqm_max, 101, endpoint=True)
    std_width = std_bin_sides[1] - std_bin_sides[0]
    std_bin_centers = std_bin_sides[:-1] + 0.5 * std_width

    full_nstd_hist, _ = np.histogram(
        X[metric].values.squeeze().tolist(),
        bins=100,
        range=(nstd_iqm_min, nstd_iqm_max),
        density=True,
    )
    full_std_hist, _ = np.histogram(
        X_scaled[metric].values.squeeze().tolist(),
        bins=100,
        range=(std_iqm_min, std_iqm_max),
        density=True,
    )

    for site, n in groups:
        nstd_site_data = X[X.site.str.contains(site)]
        nstd_iqm = nstd_site_data[metric].values.squeeze().tolist()
        nstd_hist, _ = np.histogram(
            nstd_iqm,
            bins=100,
            # weights=[1 / len(X)] * len(nstd_iqm),
            range=(nstd_iqm_min, nstd_iqm_max),
        )

        std_site_data = X_scaled[X_scaled.site.str.contains(site)]
        std_iqm = std_site_data[metric].values.squeeze().tolist()
        std_hist, _ = np.histogram(
            std_iqm,
            bins=100,
            # weights=[1 / len(X)] * len(std_iqm),
            range=(std_iqm_min, std_iqm_max),
        )

        ax1.bar(
            nstd_bin_centers,
            height=nstd_hist * full_nstd_hist,
            label=f"{site} ({n})",
            width=0.75 * nstd_width,
        )
        ax2.bar(
            std_bin_centers,
            height=std_hist * full_std_hist,
            label=f"{site} ({n})",
            width=0.75 * std_width,
        )

    ax2.legend(prop={"size": 18})
    return fig


def plot_corrmat(
    data,
    col_labels=None,
    row_labels=None,
    ax=None,
    cbar_kw={},
    cbarlabel="",
    symmetric=True,
    figsize=None,
    sort=False,
    **kwargs,
):
    """
    Create a heatmap from a pandas/numpy array.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    sort
        Flag to perform hierachical clustering on the correlation plot
    **kwargs
        All other arguments are forwarded to `imshow`.

    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # Cluster rows (if arguments enabled)
    if sort:
        import pandas as pd
        from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

        Z = linkage(data, 'complete', optimal_ordering=True)

        dendrogram(Z, labels=data.columns, no_plot=True)

        # Clusterize the data
        threshold = 0.1
        labels = fcluster(Z, threshold, criterion='distance')
        # Keep the indices to sort labels
        labels_order = np.argsort(labels)

        # Build a new dataframe with the sorted columns
        for idx, i in enumerate(data.columns[labels_order]):
            if idx == 0:
                clustered = pd.DataFrame(data[i])
            else:
                df_to_append = pd.DataFrame(data[i])
                clustered = pd.concat([clustered, df_to_append], axis=1)
        data = clustered

    if hasattr(data, "columns"):
        col_labels = data.columns
        data = data.values

    if figsize is not None:
        plt.figure(figsize=figsize)

    if not ax:
        ax = plt.gca()

    # If matrix is symmetric, keep only lower triangle
    if symmetric:
        data[np.triu(np.ones(data.shape, dtype=bool))] = np.nan

    kwargs["cmap"] = kwargs.pop("cmap", "PuOr_r")
    kwargs["vmin"] = kwargs.pop("vmin", -1.0)
    kwargs["vmax"] = kwargs.pop("vmax", 1.0)

    # Inset axis for the colorbar
    axins1 = inset_axes(
        ax,
        width="50%",  # width = 50% of parent_bbox width
        height="2%",  # height : 5%
        loc="upper right",
    )

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = plt.gcf().colorbar(im, cax=axins1, orientation="horizontal", **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(col_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", va="center", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar
