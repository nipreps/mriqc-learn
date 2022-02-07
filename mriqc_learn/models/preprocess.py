# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
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
# STATEMENT OF CHANGES: This file is derived from the sources of scikit-learn 0.19,
# which licensed under the BSD 3-clause.
# This file contains extensions and modifications to the original code.
"""Preprocessing transformers."""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.utils.validation import (
    check_is_fitted,
    FLOAT_DTYPES,
)


class DropColumns(BaseEstimator, TransformerMixin):
    """
    Wraps a data transformation to run only in specific
    columns [`source <https://stackoverflow.com/a/41461843/6820620>`_].

    Example
    -------

        >>> from mriqc_learn.models.preproces import DropColumns
        >>> tfm = DropColumns(columns=['duration', 'num_operations'])
        >>> # scaled = tfm.fit_transform(churn_d)

    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.drop(self.columns, axis=1)


class SiteRobustScaler(RobustScaler):

    def __init__(
        self,
        *,
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
        copy=True,
        unit_variance=False,
        groupby="site",
    ):
        self.groupby = groupby
        super().__init__(
            with_centering=with_centering,
            with_scaling=with_scaling,
            quantile_range=quantile_range,
            copy=copy,
            unit_variance=unit_variance,
        )

    def fit(self, X, y=None):
        self.groups_ = X.groupby(self.groupby).groups
        columns = X.columns.tolist()
        self.columns_ = list(range(len(columns)))
        self.columns_.remove(columns.index(self.groupby))
        self.scalers_ = {}
        for group, indexes in self.groups_.items():
            self.scalers_[group] = RobustScaler(
                with_centering=self.with_centering,
                with_scaling=self.with_scaling,
                quantile_range=self.quantile_range,
                copy=self.copy,
                unit_variance=self.unit_variance,
            ).fit(X.iloc[indexes, self.columns_])
        return self

    def transform(self, X):
        if not self.scalers_:
            self.fit(X)

        if self.copy:
            X = X.copy()

        for group, indexes in self.groups_.items():
            X.iloc[indexes, self.columns_] = self.scalers_[group].transform(
                X.iloc[indexes, self.columns_]
            )
        return X
