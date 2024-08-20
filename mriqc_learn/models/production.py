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
"""Create a pipeline for nested cross-validation."""

from pkg_resources import resource_filename as pkgrf

from joblib import load
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor
from mriqc_learn.models import preprocess as pp


def load_model():
    return load(pkgrf("mriqc_learn.data", "classifier.joblib"))


def init_pipeline():
    return init_pipeline_rfc()


def init_pipeline_rfc():
    """
    Initialize a pipeline running a random forest classifier.

    Parameters
    ----------
    model_type : str
        The model to use. Only 'rfc' and 'xgboost' are supported.
    """

    steps = [
        (
            "drop_ft",
            pp.DropColumns(
                drop=[f"size_{ax}" for ax in "xyz"] + [f"spacing_{ax}" for ax in "xyz"]
            ),
        ),
        (
            "scale",
            pp.SiteRobustScaler(
                with_centering=True,
                with_scaling=True,
                unit_variance=True,
            ),
        ),
        ("site_pred", pp.SiteCorrelationSelector()),
        ("winnow", pp.NoiseWinnowFeatSelect(use_classifier=True)),
        ("drop_site", pp.DropColumns(drop=["site"])),
        (
            "model",
            RFC(
                bootstrap=True,
                class_weight=None,
                criterion="gini",
                max_depth=10,
                max_features="sqrt",
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                min_samples_leaf=10,
                min_samples_split=10,
                min_weight_fraction_leaf=0.0,
                n_estimators=400,
                oob_score=True,
            ),
        ),
    ]
    return Pipeline(steps)


def init_pipeline_xgboost(
    n_estimators=50,
    max_depth=2,
    eta=0.1,
    subsample=1.0,
    learning_rate=0.1,
    colsample_bytree=1.0,
):
    steps = [
        (
            "drop_ft",
            pp.DropColumns(
                drop=[f"size_{ax}" for ax in "xyz"] + [f"spacing_{ax}" for ax in "xyz"]
            ),
        ),
        ("winnow", pp.NoiseWinnowFeatSelect(use_classifier=True)),
        (
            "model",
            XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                eta=eta,
                subsample=subsample,
                learning_rate=learning_rate,
                colsample_bytree=colsample_bytree,
                n_jobs=1,
            ),
        ),
    ]
    return Pipeline(steps)


def init_pipeline_naive(strategy="mean"):
    steps = [
        (
            "drop_ft",
            pp.DropColumns(
                drop=[f"size_{ax}" for ax in "xyz"] + [f"spacing_{ax}" for ax in "xyz"]
            ),
        ),
        ("winnow", pp.NoiseWinnowFeatSelect(use_classifier=True)),
        (
            "model",
            DummyRegressor(strategy=strategy),
        ),
    ]
    return Pipeline(steps)
