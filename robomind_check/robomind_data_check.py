"""RoboMINDv1 HDF5 data quality check library.

Checks puppet end-effector and joint-position data for "stuck" values
(nearly constant across all frames in an episode).
"""

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
import h5py


# ---------------------------------------------------------------------------
# Key candidates per robot type (auto-detected at runtime)
# ---------------------------------------------------------------------------
PUPPET_EE_CANDIDATES = [
    "puppet/end_effector",          # ur, franka, tienkung_xsens, franka_fr3_dual
    "puppet/end_effector_left",     # agilex dual-arm
    "franka/end_effector",          # simulation
]

PUPPET_JP_CANDIDATES = [
    "puppet/joint_position",        # most robot types
    "puppet/joint_position_left",   # agilex dual-arm
    "franka/joint_position",        # simulation
]

MASTER_JP_CANDIDATES = [
    "master/joint_position",        # most robot types
    "master/joint_position_left",   # agilex dual-arm
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ArrayQualityResult:
    """Quality metrics for a single numeric array."""
    key: str
    exists: bool
    num_frames: int = 0
    num_dims: int = 0
    max_range: float = 0.0          # max of per-dim (max - min)
    range_per_dim: Optional[List[float]] = None
    is_stuck: bool = False          # max_range < tolerance
    unique_count: int = 0
    unique_ratio: float = 0.0


@dataclass
class EpisodeQualityResult:
    """Quality check results for one trajectory.hdf5 file."""
    file_path: str
    benchmark: str = ""
    robot_type: str = ""
    task_name: str = ""
    split: str = ""                 # "train", "val", or "unsplit"
    episode_id: str = ""
    num_frames: int = 0
    puppet_ee: Optional[ArrayQualityResult] = None
    puppet_jp: Optional[ArrayQualityResult] = None
    master_jp: Optional[ArrayQualityResult] = None
    error: Optional[str] = None

    @property
    def has_stuck_ee(self) -> bool:
        return self.puppet_ee is not None and self.puppet_ee.is_stuck

    @property
    def has_stuck_jp(self) -> bool:
        return self.puppet_jp is not None and self.puppet_jp.is_stuck


@dataclass
class QualityCheckConfig:
    """Configurable thresholds."""
    ee_tolerance: float = 1e-3
    jp_tolerance: float = 1e-3
    compute_range_per_dim: bool = True


# ---------------------------------------------------------------------------
# Key detection
# ---------------------------------------------------------------------------
def detect_keys(h5file) -> dict:
    """Return first matching key for each role, or None."""
    result = {}
    for role, candidates in [
        ("puppet_ee", PUPPET_EE_CANDIDATES),
        ("puppet_jp", PUPPET_JP_CANDIDATES),
        ("master_jp", MASTER_JP_CANDIDATES),
    ]:
        found = None
        for key in candidates:
            if key in h5file:
                found = key
                break
        result[role] = found
    return result


# ---------------------------------------------------------------------------
# Path parsing
# ---------------------------------------------------------------------------
def _parse_path_into(result: EpisodeQualityResult):
    """Extract metadata from file path using 'success_episodes' as anchor."""
    parts = result.file_path.split("/")
    try:
        se_idx = parts.index("success_episodes")
    except ValueError:
        result.error = "No 'success_episodes' in path"
        return

    # benchmark / robot_type / task_name are before the anchor
    if se_idx >= 4:
        result.benchmark = parts[se_idx - 4]
        result.robot_type = parts[se_idx - 3]
        result.task_name = parts[se_idx - 2]
    elif se_idx >= 3:
        result.robot_type = parts[se_idx - 3]
        result.task_name = parts[se_idx - 2]

    # split and episode_id are after the anchor
    after_se = parts[se_idx + 1] if se_idx + 1 < len(parts) else ""
    if after_se in ("train", "val"):
        result.split = after_se
        result.episode_id = parts[se_idx + 2] if se_idx + 2 < len(parts) else ""
    else:
        result.split = "unsplit"
        result.episode_id = after_se


# ---------------------------------------------------------------------------
# Array quality check
# ---------------------------------------------------------------------------
def check_array_quality(
    h5file,
    key: str,
    tolerance: float,
    compute_range: bool = True,
) -> ArrayQualityResult:
    """Check a single numeric array for stuck values."""
    if key not in h5file:
        return ArrayQualityResult(key=key, exists=False)

    arr = h5file[key][:]
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n_frames, n_dims = arr.shape

    if n_frames == 0:
        return ArrayQualityResult(
            key=key, exists=True, num_frames=0, num_dims=n_dims,
            is_stuck=True,
        )

    # Core: per-dimension range
    range_per_dim = arr.max(axis=0) - arr.min(axis=0)
    max_range = float(range_per_dim.max())
    is_stuck = max_range < tolerance

    # Auxiliary: unique count via void-view trick
    arr_c = np.ascontiguousarray(arr)
    void_view = arr_c.view(np.dtype((np.void, arr_c.dtype.itemsize * n_dims)))
    unique_count = len(np.unique(void_view))
    unique_ratio = unique_count / n_frames

    return ArrayQualityResult(
        key=key,
        exists=True,
        num_frames=n_frames,
        num_dims=n_dims,
        max_range=max_range,
        range_per_dim=range_per_dim.tolist() if compute_range else None,
        is_stuck=is_stuck,
        unique_count=unique_count,
        unique_ratio=unique_ratio,
    )


# ---------------------------------------------------------------------------
# Episode-level check (main entry point)
# ---------------------------------------------------------------------------
def check_episode(
    file_path: str,
    config: QualityCheckConfig = None,
) -> EpisodeQualityResult:
    """Check data quality for a single trajectory.hdf5 file.

    Safe for multiprocessing — opens/closes file internally.
    """
    if config is None:
        config = QualityCheckConfig()

    result = EpisodeQualityResult(file_path=file_path)
    _parse_path_into(result)

    try:
        with h5py.File(file_path, "r") as f:
            keys = detect_keys(f)

            if keys["puppet_ee"]:
                result.puppet_ee = check_array_quality(
                    f, keys["puppet_ee"],
                    config.ee_tolerance,
                    config.compute_range_per_dim,
                )
                result.num_frames = result.puppet_ee.num_frames

            if keys["puppet_jp"]:
                result.puppet_jp = check_array_quality(
                    f, keys["puppet_jp"],
                    config.jp_tolerance,
                    config.compute_range_per_dim,
                )
                if result.num_frames == 0 and result.puppet_jp:
                    result.num_frames = result.puppet_jp.num_frames

            if keys["master_jp"]:
                result.master_jp = check_array_quality(
                    f, keys["master_jp"],
                    config.jp_tolerance,
                    config.compute_range_per_dim,
                )
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"

    return result
