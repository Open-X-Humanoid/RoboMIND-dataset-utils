
from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pinocchio as pin

# Pose类的简单实现，只有xyz+xyzw的格式
class Pose:
    __slots__ = ("position", "quat_xyzw")

    def __init__(self, position: np.ndarray, quat_xyzw: np.ndarray):
        self.position = np.asarray(position, dtype=np.float64).reshape(3)
        self.quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)

_URDF_ROOT = Path(__file__).resolve().parent / "urdf"

_MODEL_FRAME_CONFIG = {
    "agilex": {
        "base": "body_Link",
        "left_tcp": "fl_link6",
        "right_tcp": "fr_link6",
        "urdf_filename": "aloha_new.urdf",
        "left_arm_joints": [
            "fl_joint1", "fl_joint2", "fl_joint3", "fl_joint4", "fl_joint5", "fl_joint6",
        ],
        "right_arm_joints": [
            "fr_joint1", "fr_joint2", "fr_joint3", "fr_joint4", "fr_joint5", "fr_joint6",
        ],
    },
    "arx": {
        "base": "base_link",
        "left_tcp": "left_arm_link6",
        "right_tcp": "right_arm_link6",
        "left_arm_joints": [
            "left_arm_joint1", "left_arm_joint2", "left_arm_joint3",
            "left_arm_joint4", "left_arm_joint5", "left_arm_joint6",
        ],
        "right_arm_joints": [
            "right_arm_joint1", "right_arm_joint2", "right_arm_joint3",
            "right_arm_joint4", "right_arm_joint5", "right_arm_joint6",
        ],
    },
    "tiangong_station": {
        "base": "body",
        "left_tcp": "left_tcp_link",
        "right_tcp": "right_tcp_link",
        "left_arm_joints": [
            "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
            "elbow_pitch_l_joint", "elbow_yaw_l_joint",
            "wrist_pitch_l_joint", "wrist_roll_l_joint",
        ],
        "right_arm_joints": [
            "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
            "elbow_pitch_r_joint", "elbow_yaw_r_joint",
            "wrist_pitch_r_joint", "wrist_roll_r_joint",
        ],
    },
    "tiansuo1": {
        "base": "base",
        "left_tcp": "wrist_roll_l_link",
        "right_tcp": "wrist_roll_r_link",
        "left_arm_joints": [
            "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
            "elbow_pitch_l_joint", "elbow_yaw_l_joint",
            "wrist_pitch_l_joint", "wrist_roll_l_joint",
        ],
        "right_arm_joints": [
            "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
            "elbow_pitch_r_joint", "elbow_yaw_r_joint",
            "wrist_pitch_r_joint", "wrist_roll_r_joint",
        ],
    },
    "tianyi2": {
        "base": "base",
        "left_tcp": "left_tcp_link",
        "right_tcp": "right_tcp_link",
        "left_arm_joints": [
            "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
            "elbow_pitch_l_joint", "elbow_yaw_l_joint",
            "wrist_pitch_l_joint", "wrist_roll_l_joint",
        ],
        "right_arm_joints": [
            "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
            "elbow_pitch_r_joint", "elbow_yaw_r_joint",
            "wrist_pitch_r_joint", "wrist_roll_r_joint",
        ],
    },
    "tiangong2dex": {
        "base": "pelvis",
        "left_tcp": "left_tcp_link",
        "right_tcp": "right_tcp_link",
        "urdf_filename": "tiangong2dex.urdf",
        "left_arm_joints": [
            "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
            "elbow_pitch_l_joint", "elbow_yaw_l_joint",
            "wrist_pitch_l_joint", "wrist_roll_l_joint",
        ],
        "right_arm_joints": [
            "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
            "elbow_pitch_r_joint", "elbow_yaw_r_joint",
            "wrist_pitch_r_joint", "wrist_roll_r_joint",
        ],
    },
    "tiangong2pro": {
        "base": "pelvis",
        "left_tcp": "left_tcp_link",
        "right_tcp": "right_tcp_link",
        "left_arm_joints": [
            "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
            "elbow_pitch_l_joint", "elbow_yaw_l_joint",
            "wrist_pitch_l_joint", "wrist_roll_l_joint",
        ],
        "right_arm_joints": [
            "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
            "elbow_pitch_r_joint", "elbow_yaw_r_joint",
            "wrist_pitch_r_joint", "wrist_roll_r_joint",
        ],
    },
    "franka": {
        "single_arm": True,
        "base": "panda_link0",
        "tcp": "panda_hand_tcp",
        "urdf_filename": "panda.urdf",
        "arm_joints": [
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
            "panda_joint5", "panda_joint6", "panda_joint7",
        ],
    },
    "ur5e": {
        "single_arm": True,
        "base": "base_link",
        "tcp": "ee_link",
        "urdf_filename": "ur5_robot.urdf",
        "arm_joints": [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        ],
    },
    "tiangong1pro": {
        "base": "pelvis",
        "left_tcp": "L_hand_base_link",
        "right_tcp": "R_hand_base_link",
        "urdf_filename": "humanoid.urdf",
        "left_arm_joints": [
            "left_joint1", "shoulder_roll_l_joint", "left_joint3", "elbow_l_joint",
            "left_joint5", "left_joint6", "left_joint7",
        ],
        "right_arm_joints": [
            "right_joint1", "shoulder_roll_r_joint", "right_joint3", "elbow_r_joint",
            "right_joint5", "right_joint6", "right_joint7",
        ],
    },
}

_MODEL_FOLDER_ALIAS = {
    "tiangong1": "tiangong1pro",
    "tianyi1": "tiangong1pro",
}

def _find_urdf_for_model(
    model_name: str, preferred_urdf_filename: Optional[str] = None
) -> Tuple[Path, Path]:
    base = _URDF_ROOT / model_name
    if not base.exists():
        raise FileNotFoundError(f"Wrong model name: {base}")

    candidates: List[Path] = []
    for sub in ("urdf", "."):
        search = base / sub if sub != "." else base
        if not search.exists():
            continue
        for p in search.iterdir():
            if p.suffix.lower() == ".urdf":
                candidates.append(p)
    if preferred_urdf_filename:
        for p in candidates:
            if p.name == preferred_urdf_filename:
                return p, base
    return candidates[0], base


def _resolve_package_in_urdf(urdf_path: Path, pkg_root: Path, urdf_content: str) -> str:
    pattern = re.compile(r"package://([^/]+)/")
    abs_root = str(pkg_root.resolve()).replace("\\", "/")

    def repl(m: re.Match) -> str:
        return f"{abs_root}/"

    return pattern.sub(repl, urdf_content)


def _pick_frame_id(model: pin.Model, names: List[str]) -> int:
    for name in names:
        fid = model.getFrameId(name)
        if fid < model.nframes:
            return fid


def _get_arm_from_joint_list(
    model: pin.Model,
    joint_names: List[str],
) -> List[Tuple[str, int]]:
    name_to_idx: List[Tuple[str, int]] = []
    for jid in range(1, model.njoints):
        name = model.names[jid]
        if name not in joint_names:
            continue
        j = model.joints[jid]
        if j.nq > 0:
            name_to_idx.append((name, j.idx_q))
    # 按 joint_names 的顺序排的
    order = {n: i for i, n in enumerate(joint_names)}
    name_to_idx.sort(key=lambda x: order.get(x[0], 9999))
    return name_to_idx


class JointToEePose:
    def __init__(
        self,
        urdf_path: Path,
        model_name: Optional[str] = None,
    ):
        """
        Args:
            urdf_path: URDF 文件路径。
            model_name: 机型名，与 urdf 文件夹中的每个文件夹名一致。
        """
        urdf_path = Path(urdf_path).resolve()
        pkg_root = urdf_path.parent.parent if urdf_path.parent.name == "urdf" else urdf_path.parent

        content = urdf_path.read_text(encoding="utf-8", errors="replace")
        content = _resolve_package_in_urdf(urdf_path, pkg_root, content)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".urdf", delete=False, dir=str(pkg_root), encoding="utf-8"
        ) as f:
            f.write(content)
            tmp_path = f.name
        try:
            self.pin_model = pin.buildModelFromUrdf(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        self.data = self.pin_model.createData()

        cfg = _MODEL_FRAME_CONFIG.get(model_name, {}) if model_name else {}
        self._single_arm = bool(cfg.get("single_arm"))

        base_names = cfg.get("base")
        base_names = [base_names] if isinstance(base_names, str) else base_names
        self._base_fid = _pick_frame_id(self.pin_model, base_names)
        self._base_name = self.pin_model.frames[self._base_fid].name

        if self._single_arm:
            tcp_names = cfg.get("tcp")
            tcp_names = [tcp_names] if isinstance(tcp_names, str) else tcp_names
            self._left_tcp_fid = self._right_tcp_fid = _pick_frame_id(self.pin_model, tcp_names)
            self._left_tcp_name = self._right_tcp_name = self.pin_model.frames[self._left_tcp_fid].name
            arm_joints = cfg.get("arm_joints", [])
            single_list = _get_arm_from_joint_list(self.pin_model, arm_joints)
            self._left_arm = self._right_arm = single_list
        else:
            left_tcp_names = cfg.get("left_tcp")
            right_tcp_names = cfg.get("right_tcp")
            left_tcp_names = [left_tcp_names] if isinstance(left_tcp_names, str) else left_tcp_names
            right_tcp_names = [right_tcp_names] if isinstance(right_tcp_names, str) else right_tcp_names
            self._left_tcp_fid = _pick_frame_id(self.pin_model, left_tcp_names)
            self._right_tcp_fid = _pick_frame_id(self.pin_model, right_tcp_names)
            self._left_tcp_name = self.pin_model.frames[self._left_tcp_fid].name
            self._right_tcp_name = self.pin_model.frames[self._right_tcp_fid].name
            left_joints = cfg.get("left_arm_joints", [])
            right_joints = cfg.get("right_arm_joints", [])
            self._left_arm = _get_arm_from_joint_list(self.pin_model, left_joints)
            self._right_arm = _get_arm_from_joint_list(self.pin_model, right_joints)

        self.q = np.zeros(self.pin_model.nq)
        pin.forwardKinematics(self.pin_model, self.data, self.q)
        pin.updateFramePlacements(self.pin_model, self.data)

    def get_left_arm_joint_names(self) -> List[str]:
        return [name for name, _ in self._left_arm]

    def get_right_arm_joint_names(self) -> List[str]:
        return [name for name, _ in self._right_arm]

    def get_arm_joint_names(self) -> List[str]:
        return [name for name, _ in self._left_arm]

    def get_base_frame_name(self) -> str:
        return self._base_name

    def get_left_tcp_frame_name(self) -> str:
        return self._left_tcp_name

    def get_right_tcp_frame_name(self) -> str:
        return self._right_tcp_name

    def get_tcp_pose(self, q: np.ndarray) -> Pose:
        self._update_left_arm(q)
        return self._fk_pose(self._base_fid, self._left_tcp_fid)

    def _fk_pose(self, base_frame_id: int, tcp_frame_id: int) -> Pose:
        pin.forwardKinematics(self.pin_model, self.data, self.q)
        pin.updateFramePlacements(self.pin_model, self.data)
        T_base = self.data.oMf[base_frame_id]
        T_tcp = self.data.oMf[tcp_frame_id]
        T_rel = T_base.inverse() * T_tcp
        pos = T_rel.translation.copy()
        quat = pin.Quaternion(T_rel.rotation).coeffs()  # xyzw
        return Pose(position=pos, quat_xyzw=quat)

    def _update_left_arm(self, q_left: np.ndarray) -> None:
        q_left = np.asarray(q_left).ravel()
        for i, (_, q_idx) in enumerate(self._left_arm):
            if i < len(q_left):
                self.q[q_idx] = q_left[i]

    def _update_right_arm(self, q_right: np.ndarray) -> None:
        q_right = np.asarray(q_right).ravel()
        for i, (_, q_idx) in enumerate(self._right_arm):
            if i < len(q_right):
                self.q[q_idx] = q_right[i]

    def get_left_tcp_pose(self, q_left: np.ndarray) -> Pose:
        self._update_left_arm(q_left)
        return self._fk_pose(self._base_fid, self._left_tcp_fid)

    def get_right_tcp_pose(self, q_right: np.ndarray) -> Pose:
        self._update_right_arm(q_right)
        return self._fk_pose(self._base_fid, self._right_tcp_fid)


def get_kinematics(model_name: str) -> JointToEePose:
    folder = _MODEL_FOLDER_ALIAS.get(model_name, model_name)
    cfg = _MODEL_FRAME_CONFIG.get(folder, _MODEL_FRAME_CONFIG.get(model_name, {}))
    preferred = cfg.get("urdf_filename")
    urdf_path, pkg_root = _find_urdf_for_model(folder, preferred_urdf_filename=preferred)
    return JointToEePose(urdf_path, model_name=folder)
