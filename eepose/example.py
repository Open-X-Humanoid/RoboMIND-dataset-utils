import numpy as np
from joint2pose import get_kinematics


MODEL_NAME = "tianyi2"

# ---------------------------------------------------------------------------
# 双臂机型:agilex, arx, tiangong_station, tiangong2dex, tiangong2pro, tiansuo1, tianyi2
# ---------------------------------------------------------------------------
kin = get_kinematics(MODEL_NAME)

left_joints = kin.get_left_arm_joint_names()
right_joints = kin.get_right_arm_joint_names()
print(f"左臂关节 ({len(left_joints)}): {left_joints}")
print(f"右臂关节 ({len(right_joints)}): {right_joints}")

q_left = np.zeros(7)     # 替换为joints
q_right = np.zeros(7)
# q_left = [0.04, -0.26, 0.04, -0.62, 0.04, -0.40, -0.40]


pose_left = kin.get_left_tcp_pose(q_left)
print("q_left:", q_left)

pose_right = kin.get_right_tcp_pose(q_right)
print("q_right:", q_right)

print("左 TCP 位姿:", pose_left.position, pose_left.quat_xyzw)
print("右 TCP 位姿:", pose_right.position, pose_right.quat_xyzw)

# ---------------------------------------------------------------------------
# 单臂机型：franka，ur5e
# ---------------------------------------------------------------------------
kin = get_kinematics(MODEL_NAME)
arm_joints = kin.get_arm_joint_names()
q = np.zeros(len(arm_joints))   # 替换为joints
pose = kin.get_tcp_pose(q)
print("TCP 位姿:", pose.position, pose.quat_xyzw)
