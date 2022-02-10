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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


def plot_batches(X, cols=None, out_file=None, site_labels="left"):
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
        family="FreeSans",
        style="normal",
        size=14,
        weight="normal",
        stretch="normal",
    )
    for label in ax.get_yticklabels():
        label.set_fontproperties(ticks_font)

    ticks_font = FontProperties(
        family="FreeSans",
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
