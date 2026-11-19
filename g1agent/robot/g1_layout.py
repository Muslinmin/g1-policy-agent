"""Unitree G1 (29-DoF) upper-body layout: joint names, DDS indices, limits, default gains.

Sources:
  * motor indices: unitree_sdk2_python example/g1/high_level/g1_arm7_sdk_dds_example.py (G1JointIndex);
    index 29 (kNotUsedJoint) of rt/arm_sdk carries the arm_sdk blend weight (1 = arm_sdk owns the arms).
  * limits: g1_29dof_with_hand_only_kinematics.urdf (revolute <limit lower/upper>).
  * Dex3 slot order on rt/dex3/{left,right}/{cmd,state}: thumb_0, thumb_1, thumb_2, index_0, index_1,
    middle_0, middle_1 (verified one finger at a time on a real G1).
  * gains: a deployed G1 upper-body controller (arm kp/kd per joint, hand kp/kd).

VERIFY ON YOUR ROBOT before the first run: 23-DoF G1s have no wrist pitch/yaw and a locked waist, and
hand variants differ. Change ARM_JOINTS / limits here, nowhere else.
"""

from __future__ import annotations

import numpy as np

SIDES = ("left", "right")
ARM_SUFFIX = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch",
              "wrist_yaw")
ARM_JOINTS = [f"{s}_{j}_joint" for s in SIDES for j in ARM_SUFFIX]           # 14
ARM_MOTOR_INDEX = list(range(15, 22)) + list(range(22, 29))                   # same order as ARM_JOINTS
WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
WAIST_MOTOR_INDEX = [12, 13, 14]
ARM_SDK_WEIGHT_INDEX = 29

HAND_DDS_ORDER = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
HAND_JOINTS = {s: [f"{s}_hand_{j}_joint" for j in HAND_DDS_ORDER] for s in SIDES}

LIMITS = {  # rad, from the URDF
    "left_shoulder_pitch_joint": (-3.0892, 2.6704), "right_shoulder_pitch_joint": (-3.0892, 2.6704),
    "left_shoulder_roll_joint": (-1.5882, 2.2515), "right_shoulder_roll_joint": (-2.2515, 1.5882),
    "left_shoulder_yaw_joint": (-2.6180, 2.6180), "right_shoulder_yaw_joint": (-2.6180, 2.6180),
    "left_elbow_joint": (-1.0472, 2.0944), "right_elbow_joint": (-1.0472, 2.0944),
    "left_wrist_roll_joint": (-1.9722, 1.9722), "right_wrist_roll_joint": (-1.9722, 1.9722),
    "left_wrist_pitch_joint": (-1.6144, 1.6144), "right_wrist_pitch_joint": (-1.6144, 1.6144),
    "left_wrist_yaw_joint": (-1.6144, 1.6144), "right_wrist_yaw_joint": (-1.6144, 1.6144),
    "waist_yaw_joint": (-0.5236, 0.5236), "waist_roll_joint": (-0.2618, 0.2618),
    "waist_pitch_joint": (-0.2618, 0.2618),
    "left_hand_thumb_0_joint": (-1.0472, 1.0472), "right_hand_thumb_0_joint": (-1.0472, 1.0472),
    "left_hand_thumb_1_joint": (-0.7243, 1.0472), "right_hand_thumb_1_joint": (-1.0472, 0.7243),
    "left_hand_thumb_2_joint": (0.0, 1.7453), "right_hand_thumb_2_joint": (-1.7453, 0.0),
    "left_hand_index_0_joint": (-1.5708, 0.0), "right_hand_index_0_joint": (0.0, 1.5708),
    "left_hand_index_1_joint": (-1.7453, 0.0), "right_hand_index_1_joint": (0.0, 1.7453),
    "left_hand_middle_0_joint": (-1.5708, 0.0), "right_hand_middle_0_joint": (0.0, 1.5708),
    "left_hand_middle_1_joint": (-1.7453, 0.0), "right_hand_middle_1_joint": (0.0, 1.7453),
}

ARM_KP = (100.0, 100.0, 40.0, 40.0, 20.0, 20.0, 20.0) * 2
ARM_KD = (5.0, 5.0, 2.0, 2.0, 2.0, 2.0, 2.0) * 2
WAIST_KP, WAIST_KD = (200.0,) * 3, (5.0,) * 3
HAND_KP = (2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)   # DDS order (thumb_0 stiffer); real fingers have static
HAND_KD = (0.1,) * 7                            # friction: x2-x3 kp may be needed, see docs/G1_SETUP.md


def joint_names(with_waist: bool = False, with_hands: bool = True) -> list[str]:
    names = list(ARM_JOINTS)
    if with_waist:
        names += WAIST_JOINTS
    if with_hands:
        names += HAND_JOINTS["left"] + HAND_JOINTS["right"]
    return names


def limits_for(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([LIMITS[n][0] for n in names], np.float32)
    hi = np.array([LIMITS[n][1] for n in names], np.float32)
    return lo, hi
