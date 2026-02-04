import os
from functools import lru_cache
from typing import Dict

from src.data_utils import DatasetConfig


def _require_env(var_name: str) -> str:
    """Fetch an environment variable or raise a helpful error."""
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(f"{var_name} is not set")
    return value


@lru_cache(maxsize=1)
def get_dataset_configs() -> Dict[str, DatasetConfig]:
    """Build dataset configuration objects from environment variables."""
    data_root = _require_env("REAL_DATA_ROOT")
    synthetic_root_residual = _require_env("SYNTHETIC_DATA_ROOT_RESIDUAL")
    synthetic_root_variational = _require_env("SYNTHETIC_DATA_ROOT_VARIATIONAL")

    return {
        "PLUS": DatasetConfig(
            dataset_name="PLUS",
            data_path=os.path.join(data_root, "PLUS"),
            synthetic_path_residual=os.path.join(synthetic_root_residual, "PLUS_matched"),
            synthetic_path_variational=os.path.join(synthetic_root_variational, "PLUS_matched"),
            extensions=["png"],
            n_neighbors=6,
            task2_thirds=[1, 2, 3],
        ),
        "IDIAP": DatasetConfig(
            dataset_name="IDIAP",
            data_path=os.path.join(data_root, "IDIAP"),
            synthetic_path_residual=os.path.join(synthetic_root_residual, "IDIAP"),
            synthetic_path_variational=os.path.join(synthetic_root_variational, "IDIAP"),
            subset_root="full",
            subsets=["train", "test", "dev"],
            extensions=["png"],
            n_neighbors=1,
            task2_thirds=[1, 2],
        ),
        "SCUT": DatasetConfig(
            dataset_name="SCUT",
            data_path=os.path.join(data_root, "SCUT"),
            synthetic_path_residual=os.path.join(synthetic_root_residual, "SCUT"),
            synthetic_path_variational=os.path.join(synthetic_root_variational, "SCUT"),
            subset_root="full",
            subsets=["train", "test", "dev"],
            extensions=["bmp", "png"],
            n_neighbors=1,
            task2_thirds=[1],
        ),
    }
