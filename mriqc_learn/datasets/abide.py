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
"""Public API for the ABIDE dataset."""
from pathlib import Path
from pkg_resources import resource_filename as pkgrf

import numpy as np
from numpy.random import default_rng
import pandas as pd


def load_data(
    path=None,
    split_strategy="random",
    test_split=0.2,
    seed=None,
    site=None,
):
    """
    Load the ABIDE dataset.

    The loaded data are split into training and test datasets, and training
    and test are also divided in features and targets.

    Parameters
    ----------
    path : :obj:`os.pathlike`
        Whether to indicate a custom path were data are written in a TSV file.
    split_strategy : ``None`` or :obj:`str`
        How the data must be split into train and test subsets.
        Possible values are: ``"random"`` (default), ``"site"``, or ``None``/``"none"``.
    test_split : :obj:`float`
        Fraction of the dataset that will be split as test set when the
        split strategy is ``"random"``.
    seed : :obj:`int`
        A number to fix the seed of the random number generator
    site : :obj:`str`
        A site label indicating a particular site to be left out as test set.

    Returns
    -------
    (train_x, train_y), (test_x, test_y)
        The requested splits of the data

    """

    if site is not None:
        split_strategy = "site"

    if path is None:
        path = Path(pkgrf("mriqc_learn.datasets", "abide.tsv"))

    dataframe = pd.read_csv(path, index_col=None, delimiter=r"\s+")

    if split_strategy is None or split_strategy.lower() == "none":
        return (
            dataframe[dataframe.columns[5:]],
            dataframe[dataframe.columns[:5]]
        ), (None, None)

    n = len(dataframe)
    rng = default_rng(seed)

    if split_strategy.lower() == "random":
        sample_idx = rng.integers(n, size=int(np.round(test_split * n)))
        test_df = dataframe.iloc[sample_idx]
        train_df = dataframe.drop(sample_idx)
    else:
        if site is None:
            sites = sorted(set(dataframe.site.unique()))
            site = sites[rng.integers(len(sites), size=1)[0]]

        sample = dataframe.site.str.contains(site)
        test_df = dataframe[sample]
        train_df = dataframe[~sample]

    return (
        train_df[dataframe.columns[5:]],
        train_df[dataframe.columns[:5]]
    ), (
        test_df[dataframe.columns[5:]],
        test_df[dataframe.columns[:5]]
    )
