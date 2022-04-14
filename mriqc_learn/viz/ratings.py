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
"""Visualizing human ratings."""
import os.path as op
from math import pi
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import FontProperties
from matplotlib.patches import RegularPolygon


def plot_raters(dataframe, ax=None, width=101, size=0.40, default="whitesmoke"):
    raters = sorted(dataframe.columns.tolist())
    dataframe = dataframe.sort_values(by=raters, ascending=True)
    matrix = dataframe.values
    nsamples, nraters = dataframe.shape

    palette = {1: "limegreen", 0: "dimgray", -1: "tomato"}

    ax = ax if ax is not None else plt.gca()

    ax.set_aspect("equal", "box")
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    nrows = ((nsamples - 1) // width) + 1
    xlims = (-14.0, width)
    ylims = (-0.07 * nraters, nrows * nraters + nraters * 0.07 + (nrows - 1))

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    offset = 0.5 * (size / 0.40)

    for row in range(nrows):
        submatrix = matrix[(row * width) :, :]
        submatrix = submatrix[: min(width, len(submatrix)), :]
        for (x, y), w in np.ndenumerate(submatrix):
            color = palette.get(w, default)
            rect = RegularPolygon(
                [x + offset, y + offset + row * (nraters + 1)],
                (y + 2) * 2,
                radius=size,
                orientation=-0.5 * pi,
                facecolor=color,
                edgecolor=color,
            )
            ax.add_patch(rect)

    text_x = ((nsamples - 1) % width) + 6.5
    text_x = -8.5
    for i, rname in enumerate(raters):
        nsamples = sum(~dataframe[rname].isnull())
        good = 100 * sum(dataframe[rname] == 1) / nsamples
        bad = 100 * sum(dataframe[rname] == -1) / nsamples

        text_y = 1.5 * i + (nrows - 1) * 2.0
        ax.text(
            text_x,
            text_y,
            "%2.0f%%" % good,
            color="limegreen",
            weight=1000,
            size=16,
            horizontalalignment="right",
            verticalalignment="center",
            transform=ax.transData,
        )
        ax.text(
            text_x + 3.50,
            text_y,
            "%2.0f%%" % max((0.0, 100 - good - bad)),
            color="dimgray",
            weight=1000,
            size=16,
            horizontalalignment="right",
            verticalalignment="center",
            transform=ax.transData,
        )
        ax.text(
            text_x + 7.0,
            text_y,
            "%2.0f%%" % bad,
            color="tomato",
            weight=1000,
            size=16,
            horizontalalignment="right",
            verticalalignment="center",
            transform=ax.transData,
        )

    # ax.autoscale_view()
    ax.invert_yaxis()
    plt.grid(False)

    # Remove and redefine spines
    for side in ["top", "right", "bottom"]:
        # Toggle the spine objects
        ax.spines[side].set_color("none")
        ax.spines[side].set_visible(False)

    ax.spines["left"].set_linewidth(1.5)
    ax.spines["left"].set_color("dimgray")
    # ax.spines["left"].set_position(('data', xlims[0]))

    ax.set_yticks([0.5 * (ylims[0] + ylims[1])])
    ax.tick_params(axis="y", which="major", pad=15)

    ticks_font = FontProperties(size=20)
    for label in ax.get_yticklabels():
        label.set_fontproperties(ticks_font)

    return ax


def raters_variability_plot(
    mdata,
    figsize=(22, 22),
    width=101,
    out_file=None,
    raters=["rater_1", "rater_2", "rater_3"],
    only_overlap=False,
    rater_names=["Rater 1", "Rater 2a", "Rater 2b"],
):
    if only_overlap:
        mdata = mdata[np.all(~np.isnan(mdata[raters]), axis=1)]
    # Swap raters 2 and 3
    # i, j = cols.index('rater_2'), cols.index('rater_3')
    # cols[j], cols[i] = cols[i], cols[j]
    # mdata.columns = cols

    sites_list = sorted(set(mdata.site.values.ravel().tolist()))
    sites_len = []
    for site in sites_list:
        sites_len.append(len(mdata.loc[mdata.site == site]))

    sites_len, sites_list = zip(*sorted(zip(sites_len, sites_list)))

    blocks = [(slen - 1) // width + 1 for slen in sites_len]
    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = GridSpec(
        len(sites_list), 1, width_ratios=[1], height_ratios=blocks, hspace=0.05
    )

    for s, gsel in zip(sites_list, gs):
        ax = plt.subplot(gsel)
        plot_raters(
            mdata.loc[mdata.site == s, raters],
            ax=ax,
            width=width,
            size=0.40 if len(raters) == 3 else 0.80,
        )
        ax.set_yticklabels([s])

    # ax.add_line(Line2D([0.0, width], [8.0, 8.0], color='k'))
    # ax.annotate(
    #     '%d images' % width, xy=(0.5 * width, 8), xycoords='data',
    #     xytext=(0.5 * width, 9), fontsize=20, ha='center', va='top',
    #     arrowprops=dict(arrowstyle='-[,widthB=1.0,lengthB=0.2', lw=1.0)
    # )

    # ax.annotate('QC Prevalences', xy=(0.1, -0.15), xytext=(0.5, -0.1), xycoords='axes fraction',
    #         fontsize=20, ha='center', va='top',
    #         arrowprops=dict(arrowstyle='-[, widthB=3.0, lengthB=0.2', lw=1.0))

    newax = plt.axes([0.6, 0.65, 0.25, 0.16])
    newax.grid(False)
    newax.set_xticklabels([])
    newax.set_xticks([])
    newax.set_yticklabels([])
    newax.set_yticks([])

    nsamples = len(mdata)
    for i, rater in enumerate(raters):
        nsamples = len(mdata) - sum(np.isnan(mdata[rater].values))
        good = 100 * sum(mdata[rater] == 1.0) / nsamples
        bad = 100 * sum(mdata[rater] == -1.0) / nsamples

        text_x = 0.92
        text_y = 0.5 - 0.17 * i
        newax.text(
            text_x - 0.37,
            text_y,
            "%2.1f%%" % good,
            color="limegreen",
            weight=500,
            size=25,
            horizontalalignment="right",
            verticalalignment="center",
            transform=newax.transAxes,
        )
        newax.text(
            text_x - 0.18,
            text_y,
            "%2.1f%%" % max((0.0, 100 - good - bad)),
            color="dimgray",
            weight=500,
            size=25,
            horizontalalignment="right",
            verticalalignment="center",
            transform=newax.transAxes,
        )
        newax.text(
            text_x + 0.02,
            text_y,
            "%2.1f%%" % bad,
            color="tomato",
            weight=500,
            size=25,
            horizontalalignment="right",
            verticalalignment="center",
            transform=newax.transAxes,
        )

        newax.add_patch(
            RegularPolygon(
                [text_x - 0.62, text_y], (i + 2) * 2, radius=0.06, color="lightgray"
            )
        )
        newax.text(
            1 - text_x,
            text_y,
            "Rater",
            color="k",
            size=25,
            horizontalalignment="left",
            verticalalignment="center",
            transform=newax.transAxes,
        )
        newax.text(
            text_x - 0.6365,
            text_y,
            f"{i + 1}",
            color="w",
            size=25,
            horizontalalignment="left",
            verticalalignment="center",
            transform=newax.transAxes,
        )

    newax.text(
        0.5,
        0.95,
        "Imbalance of ratings",
        color="k",
        size=25,
        horizontalalignment="center",
        verticalalignment="top",
        transform=newax.transAxes,
    )
    newax.text(
        0.5,
        0.85,
        "(ABIDE, aggregated)",
        color="k",
        size=25,
        horizontalalignment="center",
        verticalalignment="top",
        transform=newax.transAxes,
    )

    if out_file is True:
        out_file = "raters.svg"

    if out_file:
        fname, ext = op.splitext(out_file)
        if ext[1:] not in ["pdf", "svg", "png"]:
            ext = ".svg"
            out_file = fname + ".svg"

        fig.savefig(
            op.abspath(out_file), format=ext[1:], bbox_inches="tight", pad_inches=0, dpi=300
        )
    return fig
