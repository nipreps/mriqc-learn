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
"""Access to the MRIQC Web-API."""
from functools import partial
import random
from time import sleep
import json
import asyncio
import concurrent.futures
import requests
import tqdm


def get_page(page, endpoint="T1w", max_results=50, progress=None, jitter=True):
    """Get a given page."""
    if jitter:
        sleep(random.random() * 3.0)
    if not (r := requests.get(f"https://mriqc.nimh.nih.gov/api/v1/{endpoint}", params={"page": page, "max_results": max_results})).ok:
        return

    data = json.loads(r.text)
    if not data["_items"]:
        return

    if progress is not None:
        progress.update(1)

    return data

def get_npages(endpoint="T1w", max_results=50):
    """Get number of pages in a given endpoint."""
    if (n := get_page(1, endpoint=endpoint, max_results=50)) is not None:
        return int(n["_links"]["last"]["href"].rsplit("page=")[-1])

    raise RuntimeError("Could not get number of pages - perhaps inexistent endpoint?")


def fetch(endpoint="T1w", njobs=None):
    """Fetch all entries from an endpoint."""
    total_pages = get_npages(endpoint)
    pbar = tqdm.tqdm(total=total_pages)
    callback = partial(get_page, endpoint=endpoint, progress=pbar)

    with concurrent.futures.ThreadPoolExecutor(max_workers=njobs) as executor:
        return executor.map(callback, range(1, total_pages + 1))
