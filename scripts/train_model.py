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
"""Trains and exports the model."""
from os import cpu_count
import numpy as np
import pandas as pd
from joblib import dump
from mriqc_learn.datasets import load_dataset
from mriqc_learn.models.production import init_pipeline


def main():
    """Generate a model file."""
    # Load ABIDE
    (train_x, train_y), (_, _) = load_dataset(split_strategy="none")
    train_x["site"] = train_y.site

    # Load DS030
    (test_x, test_y), (_, _) = load_dataset("ds030", split_strategy="none")
    test_x["site"] = test_y.site

    # Merged dataset
    merged_x = pd.concat((train_x, test_x))
    merged_y = np.hstack(
        (
            train_y.rater_3.values.squeeze().astype(int),
            test_y.rater_1.values.squeeze().astype(int),
        )
    )
    merged_y[merged_y < 1] = 0

    fit_params = {
        "winnow__n_jobs": cpu_count(),
        "site_pred__n_jobs": cpu_count(),
    }

    model = init_pipeline().fit(
        X=merged_x,
        y=merged_y,
        **fit_params,
    )
    dump(model, "classifier.joblib")


if __name__ == "__main__":
    main()
