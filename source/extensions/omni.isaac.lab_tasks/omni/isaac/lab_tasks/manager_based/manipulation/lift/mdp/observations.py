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
from omni.isaac.lab.utils.math import quat_inv, matrix_from_quat, subtract_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def target_position_in_robot_root_frame(
        env: ManagerBasedRLEnv,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        target_cfg: SceneEntityCfg = None
) -> torch.Tensor:
    """
    Generalized function to get the position of a target object in the robot's root frame.

    Args:
        env (ManagerBasedRLEnv): The environment containing the scene.
        robot_cfg (SceneEntityCfg): Configuration for the robot.
        target_cfg (SceneEntityCfg): Configuration for the target object.

    Returns:
        torch.Tensor: The position of the target object in the robot's root frame.
    """
    if target_cfg is None:
        raise ValueError("A valid target configuration is required.")

    robot: RigidObject = env.scene[robot_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    target_pos_w = target.data.root_state_w[:, :3]
    target_quat_w = target.data.root_state_w[:, 3:7]
    target_pos_b, target_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], target_pos_w, target_quat_w
    )
    return torch.concat([target_pos_b, target_quat_b], dim=-1)


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    object_pos_b = target_position_in_robot_root_frame(env, target_cfg=object_cfg)
    return object_pos_b


def bin_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    bin_cfg: SceneEntityCfg = SceneEntityCfg("bin"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    bin_pos_b = target_position_in_robot_root_frame(env, target_cfg=bin_cfg)
    return bin_pos_b


def cabinet_position_in_robot_root_frame(
        env: ManagerBasedRLEnv,
        cabinet_cfg: SceneEntityCfg = SceneEntityCfg("cabinet"),
) -> torch.Tensor:
    """The position of the cabinet in the robot's root frame."""
    cabinet_pos_b = target_position_in_robot_root_frame(env, target_cfg=cabinet_cfg)
    return cabinet_pos_b


def bowl_position_in_robot_root_frame(
        env: ManagerBasedRLEnv,
        target_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
) -> torch.Tensor:
    """The position of the bowl in the robot's root frame."""
    target_pos_b = target_position_in_robot_root_frame(env, target_cfg=target_cfg)
    return target_pos_b


def plate_position_in_robot_root_frame(
        env: ManagerBasedRLEnv,
        target_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
) -> torch.Tensor:
    """The position of the plate in the robot's root frame."""
    target_pos_b = target_position_in_robot_root_frame(env, target_cfg=target_cfg)
    return target_pos_b


def tray_position_in_robot_root_frame(
        env: ManagerBasedRLEnv,
        target_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
) -> torch.Tensor:
    """The position of the tray in the robot's root frame."""
    target_pos_b = target_position_in_robot_root_frame(env, target_cfg=target_cfg)
    return target_pos_b


def chopsticks_position_in_robot_root_frame(
        env: ManagerBasedRLEnv,
        target_cfg: SceneEntityCfg = SceneEntityCfg("chopsticks"),
) -> torch.Tensor:
    """The position of the tray in the robot's root frame."""
    target_pos_b = target_position_in_robot_root_frame(env, target_cfg=target_cfg)
    return target_pos_b


def broom_position_in_robot_root_frame(
        env: ManagerBasedRLEnv,
        target_cfg: SceneEntityCfg = SceneEntityCfg("broom"),
) -> torch.Tensor:
    """The position of the tray in the robot's root frame."""
    target_pos_b = target_position_in_robot_root_frame(env, target_cfg=target_cfg)
    return target_pos_b


def spoon_position_in_robot_root_frame(
        env: ManagerBasedRLEnv,
        target_cfg: SceneEntityCfg = SceneEntityCfg("spoon"),
) -> torch.Tensor:
    """The position of the tray in the robot's root frame."""
    target_pos_b = target_position_in_robot_root_frame(env, target_cfg=target_cfg)
    return target_pos_b


def mug_position_in_robot_root_frame(
        env: ManagerBasedRLEnv,
        target_cfg: SceneEntityCfg = SceneEntityCfg("mug"),
) -> torch.Tensor:
    """The position of the mug in the robot's root frame."""
    target_pos_b = target_position_in_robot_root_frame(env, target_cfg=target_cfg)
    return target_pos_b


#################################### ee pose ##################################
def right_robot_eef_pose(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ee_frame"),
) -> torch.Tensor:
    """The position of the ee pose in the world's root frame."""
    # End-effector position: (num_envs, 3)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :].clone()
    ee_quat = ee_frame.data.target_quat_w[..., 0, :].clone()
    return torch.concat([ee_pos, ee_quat], dim=-1)


def right_robot_eef_width(
        env: ManagerBasedRLEnv,
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ee_frame"),
        left_finger_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_robot_left_finger_frame"),
) -> torch.Tensor:
    """The position of the ee pose in the world's root frame."""
    # End-effector width: (num_envs, 1)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :].clone()
    left_finger_frame: FrameTransformer = env.scene[left_finger_frame_cfg.name]
    left_finger_pos = left_finger_frame.data.target_pos_w[..., 0, :].clone()
    finger_width = 2 * torch.norm((ee_pos - left_finger_pos), dim=-1)
    return finger_width.unsqueeze(0)

def left_robot_eef_pose(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("left_ee_frame"),
) -> torch.Tensor:
    """The position of the ee pose in the world's root frame."""
    # End-effector position: (num_envs, 3)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :].clone()
    ee_quat = ee_frame.data.target_quat_w[..., 0, :].clone()
    return torch.concat([ee_pos, ee_quat], dim=-1)


def left_robot_eef_width(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("left_ee_frame"),
    left_finger_frame_cfg: SceneEntityCfg = SceneEntityCfg("left_robot_left_finger_frame"),
) -> torch.Tensor:
    """The position of the ee finger in the world's root frame."""
    # End-effector position: (num_envs, 3)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :].clone()
    left_finger_frame: FrameTransformer = env.scene[left_finger_frame_cfg.name]
    left_finger_pos = left_finger_frame.data.target_pos_w[..., 0, :].clone()
    finger_width = 2 * torch.norm((ee_pos - left_finger_pos), dim=-1)
    return finger_width.unsqueeze(0)


def left_robot_base_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot_left"),
) -> torch.Tensor:
    """The position of the left robot base position in the world's root frame."""
    # End-effector position: (num_envs, 3)
    robot: RigidObject = env.scene[robot_cfg.name]
    return robot.data.root_state_w


############################ Camera ###########################################
def right_eye_camera(
    env: ManagerBasedRLEnv,
    right_eye_camera_cfg: SceneEntityCfg = SceneEntityCfg("right_eye_camera"),
) -> torch.Tensor:
    tv_camera: Camera = env.scene.sensors[right_eye_camera_cfg.name]
    tv_image = tv_camera.data.output[0]['rgb'][:, :, :3]
    return torch.unsqueeze(tv_image, 0)


def left_eye_camera(
    env: ManagerBasedRLEnv,
    left_eye_camera_cfg: SceneEntityCfg = SceneEntityCfg("left_eye_camera"),
) -> torch.Tensor:
    le_camera: Camera = env.scene.sensors[left_eye_camera_cfg.name]
    le_image = le_camera.data.output[0]['rgb'][:, :, :3]
    return torch.unsqueeze(le_image, 0)


def eye_in_hand_camera(
    env: ManagerBasedRLEnv,
    eye_in_hand_cfg: SceneEntityCfg = SceneEntityCfg("eye_in_hand_camera"),
) -> torch.Tensor:
    eih_camera: Camera = env.scene.sensors[eye_in_hand_cfg.name]
    eih_image = eih_camera.data.output[0]['rgb'][:, :, :3]
    return torch.unsqueeze(eih_image, 0)