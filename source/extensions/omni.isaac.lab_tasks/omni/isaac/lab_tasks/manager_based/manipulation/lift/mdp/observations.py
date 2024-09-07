# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import Camera, FrameTransformer
from omni.isaac.lab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


def robot_eef_pos(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """The position of the ee pose in the world's root frame."""
    # End-effector position: (num_envs, 3)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    return ee_pos


def robot_eef_quat(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """The quaternion of the ee pose in the world's root frame."""
    # End-effector quaternion: (num_envs, 4)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_quat = ee_frame.data.target_quat_w[..., 0, :].clone()
    return ee_quat


def top_view_camera(
    env: ManagerBasedRLEnv,
    top_view_camera_cfg: SceneEntityCfg = SceneEntityCfg("top_view_camera"),
) -> torch.Tensor:
    tv_camera: Camera = env.scene.sensors[top_view_camera_cfg.name]
    tv_image = tv_camera.data.output[0]['rgb'][:, :, :3]
    return torch.unsqueeze(tv_image, 0)


def eye_in_hand_camera(
    env: ManagerBasedRLEnv,
    eye_in_hand_cfg: SceneEntityCfg = SceneEntityCfg("eye_in_hand_camera"),
) -> torch.Tensor:
    eih_camera: Camera = env.scene.sensors[eye_in_hand_cfg.name]
    eih_image = eih_camera.data.output[0]['rgb'][:, :, :3]
    return torch.unsqueeze(eih_image, 0)