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
import logging
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, OneHotEncoder


LOG = logging.getLogger("mriqc_learn")
rng = np.random.default_rng()


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


class NoiseWinnowFeatSelect(BaseEstimator, TransformerMixin):
    """
    Remove features with less importance than a noise feature
    https://gist.github.com/satra/c6eb113055810f19709fa7c5ebd23de8

    """

    def __init__(
        self,
        n_winnow=10,
        use_classifier=False,
        n_estimators=1000,
        disable=False,
        k=1,
        ignore=("site", ),
    ):
        self.disable = disable
        self.n_winnow = n_winnow
        self.use_classifier = use_classifier
        self.n_estimators = n_estimators
        self.k = k
        self.ignore = ignore
        self.importances_ = None
        self.importances_snr_ = None
        self.mask_ = None

    def fit(self, X, y=None, n_jobs=1):
        """Fit the model with X.
        This is the workhorse function.
        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        self : object
            Returns the instance itself.
        self.mask_ : array
            Logical array of features to keep
        """
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

        if self.disable:
            self.mask_ = np.ones(X.shape[1], dtype=bool)
            return self

        X_input = X.copy()

        # Drop metadata columns (e.g., site)
        if self.ignore:
            columns_ = X_input.columns.tolist()
            dropped_ = []
            for col in set(self.ignore).intersection(columns_):
                X_input = X_input.drop(col, axis=1)
                dropped_.append(columns_.index(col))
            self.dropped_ = sorted(dropped_)

        n_sample, n_feature = np.shape(X_input)
        self.mask_ = np.ones(n_feature, dtype=bool)

        if y is None:
            y = X[["site"]].copy().values
            self.use_classifier = True

        if self.use_classifier:
            y = OrdinalEncoder().fit_transform(y)

        if hasattr(y, "columns"):
            y = y.values.squeeze()

        counter = 0
        noise_flag = True
        while noise_flag:
            counter = counter + 1
            noise_feature = _generate_noise(n_sample, y, self.use_classifier)

            # Add noise feature
            X = X_input.loc[:, self.mask_].copy()
            X["noise"] = noise_feature

            # Initialize estimator
            clf = ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                criterion="gini",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features="sqrt",
                max_leaf_nodes=None,
                min_impurity_decrease=1e-07,
                bootstrap=True,
                oob_score=False,
                n_jobs=n_jobs,
                random_state=None,
                verbose=0,
                warm_start=False,
                class_weight="balanced",
            ) if self.use_classifier else ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                criterion="squared_error",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features="auto",
                max_leaf_nodes=None,
                min_impurity_decrease=1e-07,
                bootstrap=False,
                oob_score=False,
                n_jobs=n_jobs,
                random_state=None,
                verbose=0,
                warm_start=False,
            )

            clf.fit(X, y.reshape(-1))
            LOG.debug("done fitting once")
            importances = clf.feature_importances_
            drop_features = importances[:-1] <= (self.k * importances[-1])

            if np.all(~drop_features):
                LOG.warn(
                    "All features (%d) are better than noise", self.mask_.sum()
                )
            elif np.all(drop_features):
                LOG.warn("No features are better than noise")
                # noise better than all features aka no feature better than noise
            else:
                LOG.warn(
                    "Removing feature less relevant than noise: "
                    f"{', '.join(X.columns[:-1][drop_features])}."
                )
                self.mask_[self.mask_] = ~drop_features

            # fail safe
            if counter >= self.n_winnow:
                noise_flag = False

        self.importances_ = importances[:-1]
        self.importances_snr_ = importances[:-1] / importances[-1]
        LOG.warn(
            "Feature selection: %d of %d features better than noise feature",
            self.mask_.sum(),
            len(self.mask_),
        )
        return self

    # def fit_transform(self, X, y=None):
    #     """Fit the model with X and apply the dimensionality reduction on X.
    #     Parameters
    #     ----------
    #     X : array-like, shape (n_samples, n_features)
    #         Training data, where n_samples is the number of samples
    #         and n_features is the number of features.
    #     Returns
    #     -------
    #     X_new : array-like, shape (n_samples, n_components)
    #     """
    #     return self.fit(X, y).transform(X, y)

    def transform(self, X, y=None):
        """Apply dimensionality reduction to X.
        X is masked.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, ["mask_"], all_or_any=all)
        if self.dropped_:
            self.mask_ = np.insert(self.mask_, self.dropped_, True)
        return X.loc[:, self.mask_]


class SiteCorrelationSelector(BaseEstimator, TransformerMixin):
    """
    Remove features with less importance than a noise feature
    https://gist.github.com/satra/c6eb113055810f19709fa7c5ebd23de8

    """

    def __init__(
        self,
        target_auc=0.6,
        n_estimators=1000,
        disable=False,
        max_iter=None,
        max_remove=0.7,
        site_col="site",
    ):
        self.disable = disable
        self.target_auc = target_auc
        self.n_estimators = n_estimators
        self.mask_ = None
        self.max_remove = max_remove if max_remove > 0 else None
        self.max_iter = max_iter
        self.site_col = site_col

    def fit(self, X, y, n_jobs=1):
        """Fit the model with X.
        This is the workhorse function.
        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        self : object
            Returns the instance itself.
        self.mask_ : array
            Logical array of features to keep
        """
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.model_selection import train_test_split

        n_feature = np.shape(X)[1]
        self.mask_ = np.ones(n_feature, dtype=bool)

        if self.disable:
            return self

        sites = X[[self.site_col]]
        if len(set(sites.values.squeeze().tolist())) == 1:
            return self

        site_index = X.columns.tolist().index(self.site_col)
        X_input = X.copy()

        self.mask_[site_index] = False  # Always remove site
        n_feature -= 1  # Remove site

        y_input = OrdinalEncoder().fit_transform(sites)

        X_train, X_test, y_train, y_test = train_test_split(
            X_input, y_input, test_size=0.33, random_state=42
        )

        max_remove = n_feature - 5
        if self.max_remove < 1.0:
            max_remove = int(self.max_remove * n_feature)
        elif int(self.max_remove) < n_feature:
            max_remove = int(self.max_remove)

        removed_names = []
        min_score = 1.0
        i = 0
        while True:
            clf = ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                criterion="gini",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features="sqrt",
                max_leaf_nodes=None,
                min_impurity_decrease=1e-07,
                bootstrap=True,
                oob_score=False,
                n_jobs=n_jobs,
                random_state=None,
                verbose=0,
                warm_start=False,
                class_weight="balanced",
            ).fit(X_train.loc[:, self.mask_], y_train.reshape(-1))

            y_predicted = OneHotEncoder(sparse=False).fit(y_input).transform(
                clf.predict(X_test.loc[:, self.mask_]).reshape(-1, 1).astype("uint8")
            )

            score = roc_auc_score(
                y_test,
                y_predicted,
                average="macro",
                multi_class="ovr",
                sample_weight=None,
            )

            if score < self.target_auc:
                break
            if np.sum(~self.mask_) >= max_remove:
                break
            if self.max_iter is not None and i >= self.max_iter:
                break

            importances = np.zeros(self.mask_.shape)
            importances[self.mask_] = clf.feature_importances_
            rm_feat = np.argmax(importances)

            # Remove feature
            self.mask_[rm_feat] = False
            removed_names.append(X.columns[rm_feat])
            if score < min_score:
                min_score = score

            LOG.warn(f"Removing [{i}] {X.columns[rm_feat]}")

            i += 1

        LOG.warn(f"""\
Feature selection:
- {self.mask_.sum()} Kept: {', '.join(X.columns[self.mask_])}.
- {len(removed_names)} Removed: {', '.join(removed_names)}.""")
        return self

    def fit_transform(self, X, y=None, n_jobs=1):
        """Fit the model with X and apply the dimensionality reduction on X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        return self.fit(X, y, n_jobs=n_jobs).transform(X, y)

    def transform(self, X, y=None):
        """Apply dimensionality reduction to X.
        X is masked.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, ["mask_"], all_or_any=all)
        if self.site_col in X.columns.tolist():
            self.mask_[X.columns.tolist().index(self.site_col)] = True
        return X.loc[:, self.mask_]


def _generate_noise(n_sample, y, clf_flag=True):
    """
    Generates a random noise sample that is not correlated (<0.05)
    with the output y. Uses correlation if regression, and ROC AUC
    if classification
    """
    if clf_flag:
        return np.random.normal(loc=0, scale=1, size=(n_sample, 1))

    noise_corr = 1.0
    while noise_corr > 0.05:
        noise_feature = np.random.normal(loc=0, scale=10.0, size=(n_sample, 1))
        noise_corr = np.abs(np.corrcoef(noise_feature, y[:, np.newaxis], rowvar=0)[0][1])

    return noise_feature
