import os
import glob


def get_cache_dir(dir, dataset_name):
    if dir is None:
        return dir
    if dir == "memory":
        return ""
    # We need to check for trailing lockfiles here:
    # https://github.com/tensorflow/tensorflow/issues/28798
    for i in range(3):
        cache_dir = os.path.join(dir, dataset_name if i == 0 else f"{dataset_name}_{i}")
        if not glob.glob(f"{cache_dir}/*.lockfile"):
            os.makedirs(cache_dir, exist_ok=True)
            return cache_dir
    raise RuntimeError(
        f"Out of retries! Cache lockfile already exists ({cache_dir}). "
        "If you are sure no other running TF computations are using this cache prefix, "
        "delete the lockfile and restart training."
    )
