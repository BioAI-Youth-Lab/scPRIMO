from __future__ import annotations

import os
import pickle

import numpy as np

from primo.paths import DATA_DIR


def load_kfold_split(
    fold_num, split_dir=None, k=5
):
    split_dir = split_dir or str(DATA_DIR / "splits")
    with open(os.path.join(split_dir, f"kfold_{k}_split_{fold_num}.pkl"), "rb") as f:
        split = pickle.load(f)
    return split


def load_kfold_split_testsize(
    fold_num, split_dir=None, k=5, test_size=0.3
):
    split_dir = split_dir or str(DATA_DIR / "splits")
    with open(
        os.path.join(split_dir, f"kfold_{k}_split_{fold_num}_testsize_{test_size}.pkl"),
        "rb",
    ) as f:
        split = pickle.load(f)
    return split


def load_data_testsize(
    keys, load_dir="./sciplex_example/cached_data/", idx=1, test_size=0.3
):
    data = {}

    for key in keys:
        if os.path.exists(os.path.join(load_dir, f"{key}_{test_size}_{idx}.npy")):
            data[key] = np.load(os.path.join(load_dir, f"{key}_{test_size}_{idx}.npy"))
        elif os.path.exists(os.path.join(load_dir, f"{key}_{test_size}_{idx}.pkl")):
            with open(
                os.path.join(load_dir, f"{key}_{test_size}_{idx}.pkl"), "rb"
            ) as f:
                data[key] = pickle.load(f)
        else:
            return None

    print("数据加载完成！")
    return data
