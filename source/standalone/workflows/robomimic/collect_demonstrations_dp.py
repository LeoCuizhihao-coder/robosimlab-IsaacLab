# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Collect demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Abs-dp", help="Name of the task.")
parser.add_argument("--teleop_device", type=str, default="keyboard_dual_arm", help="Device for interacting with environment")
parser.add_argument("--num_demos", type=int, default=1, help="Number of episodes to store in the dataset.")
parser.add_argument("--fps", type=int, default=30, help="diffusion video fps.")
parser.add_argument("--skip_frame", type=int, default=1, help="save every N frame")
parser.add_argument("--replicator", type=bool, default=False, help="enable table replicator")
parser.add_argument("--debug", type=bool, default=False, help="draw ee pose and action on video.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import os
import torch
import cv2
import numpy as np

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.devices import Se3Keyboard, Se3SpaceMouse, Se3Keyboard_Dual
# from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
from omni.isaac.lab.utils.math import (project_points,
                                       matrix_from_quat,
                                       matrix_from_euler,
                                       compute_pose_error,
                                       subtract_frame_transforms)
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
import omni.isaac.lab_tasks  # noqa: F401
# from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg

import omni.replicator.core as rep

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecorder
from action_atom import (GripperOpen,
                         GripperClose,
                         TranslateToObject,
                         RotateToObject,
                         TranslateToFixed,
                         RotateToFixed,
                         Wait,
                         MoveTo,
                         HumanControl)
from action import Action
from manipulation import Manipulation
from task import Task
from episode import Episode

from dp_utils import (quat_from_euler_xyz,
                      euler_xyz_from_quat,
                      inverse_transformation,
                      pre_process_actions,
                      get_object_pose_in_camera_frame)


def run_env_until_converge(get_obs_dict_callback, actions,
                           pos_error_thresh=0.01, axis_angle_error_thresh=0.01, max_attempts=100, verbose=False):
    """
    Runs the environment and exits the loop if the position error and axis-angle error are below the thresholds.

    Args:
        get_obs_dict_callback: Callback function to fetch `obs_dict`. It should return the observation dictionary.
        actions: Actions, a 2D array where the first dimension is the timestep, and the second dimension includes position and quaternion commands.
        pos_error_thresh: Position error threshold, below which the loop exits.
        axis_angle_error_thresh: Axis-angle error threshold, below which the loop exits.
        max_attempts: Maximum number of attempts (iterations) to try before exiting.
        verbose: If True, prints debug information during execution.

    Returns:
        (converged: bool, steps: int):
            converged: True if the loop exits due to error thresholds being met, False if it exits after max_attempts or due to termination.
            steps: The number of steps actually executed.
    """

    for i in range(max_attempts):
        # print("actions ", actions[:, :3])
        # 对环境执行动作
        obs_dict = get_obs_dict_callback(actions)

        # 获取机器人末端执行器的位置和四元数
        next_right_robot_eef_pos = obs_dict["policy"]["right_robot_eef_pose"][:, :3]
        next_right_robot_eef_quat = obs_dict["policy"]["right_robot_eef_pose"][:, 3:7]
        next_left_robot_eef_pos = obs_dict["policy"]["left_robot_eef_pose"][:, :3]
        next_left_robot_eef_quat = obs_dict["policy"]["left_robot_eef_pose"][:, 3:7]

        ############ left arm frame to self frame, action manage ik solver needs self frame instead world frame ####
        next_left_robot_eef_pos, _ = subtract_frame_transforms(obs_dict["policy"]["left_robot_position_w"][:, :3],
                                                               obs_dict["policy"]["left_robot_position_w"][:, 3:7],
                                                               next_left_robot_eef_pos)

        # 计算位置误差和轴角误差
        right_pos_error, right_axis_angle_error = compute_pose_error(next_right_robot_eef_pos,
                                                                     next_right_robot_eef_quat,
                                                                     actions[:, :3],
                                                                     actions[:, 3:7])
        left_pos_error, left_axis_angle_error = compute_pose_error(next_left_robot_eef_pos,
                                                                   next_left_robot_eef_quat,
                                                                   actions[:, 8:11],
                                                                   actions[:, 11:15])

        right_arm_ok = torch.norm(right_pos_error) < pos_error_thresh and torch.norm(right_axis_angle_error) < axis_angle_error_thresh
        left_arm_ok = torch.norm(left_pos_error) < pos_error_thresh and torch.norm(left_axis_angle_error) < axis_angle_error_thresh

        # 如果误差小于阈值，退出循环
        if right_arm_ok and left_arm_ok:
            if verbose:
                print(f"right arm : {right_arm_ok}")
                print(f"left arm : {left_arm_ok}")
                print(f"Converged at step {i}")
            return obs_dict, True, i

    # 如果达到最大尝试次数后还未退出，返回未收敛
    if verbose:
        print(f"Did not converge after {max_attempts} attempts")
    return obs_dict, False, max_attempts


def save_video(image_list, video_dir, file_name, fps, crf=22):
    video_path = os.path.join(video_dir, file_name)
    video_writer = VideoRecorder.create_h264(
        fps=fps,
        codec='h264',
        input_pix_fmt='rgb24',
        crf=crf,
        thread_type='FRAME',
        thread_count=1
    )
    video_writer.start(video_path)

    for image in image_list:
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()[0]  # shape is (1, H, W, 3)
        assert image.dtype == np.uint8
        video_writer.write_frame(image)

    video_writer.stop()

    return video_writer


def project_pose_to_2d(image,
                       cam_matrix_inv_w,
                       cam_pos_inv_w,
                       pose_w,
                       cam_intrinsic,
                       image_w,
                       color,
                       markerType=0,
                       markerSize=10):
    """
    Projects a 3D pose to 2D image coordinates and draws a marker on the image.

    Parameters:
        image (np.ndarray): The image on which to draw.
        cam_matrix_inv_w (torch.Tensor): Inverse camera matrix in world frame.
        cam_pos_inv_w (torch.Tensor): Inverse camera position in world frame.
        pose_w (torch.Tensor): 3D pose in world frame.
        cam_intrinsic (torch.Tensor): Camera intrinsic matrix.
        image_w (int): Image width.
        color (tuple): Marker color in BGR format.
        markerType (int): Type of marker to draw (default: 0).
        markerSize (int): Size of the marker (default: 10).

    Returns:
        np.ndarray: Updated image with the marker.
    """
    # Decompose pose into rotation and position
    matrix_w = matrix_from_euler(pose_w[:, 3:6], "XYZ").squeeze()  # Pose rotation in world frame
    pos_w = pose_w[:, :3].squeeze()  # Pose position in world frame

    # Project the 3D pose to the 2D camera view
    _, pos_in_cam = get_object_pose_in_camera_frame(cam_matrix_inv_w, cam_pos_inv_w, matrix_w, pos_w)
    pos_in_cam_2d = project_points(pos_in_cam.unsqueeze(0), cam_intrinsic).squeeze().cpu().numpy()
    pos_2d = np.round(pos_in_cam_2d[:2]).astype(int)

    # Draw the marker on the image
    cv2.drawMarker(image, (image_w - pos_2d[0], pos_2d[1]), color, markerSize=markerSize, markerType=markerType)

    return image


def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.01, 0.01, 0.01),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


def main():
    """Collect demonstrations from the environment using teleop interfaces."""
    assert (args_cli.task == "Isaac-Lift-Cube-Franka-IK-Abs-dp" or args_cli.task == "Isaac-Lift-Cube-Franka-IK-Rel-dp"), \
        "Only 'Isaac-Lift-Cube-Franka-IK-Abs-v0' or 'Isaac-Lift-Cube-Franka-IK-Rel-v0' is supported currently."

    global obs_dict

    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    # modify configuration such that the environment runs indefinitely
    # set the resampling time range to large number to avoid resampling
    env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
    # we want to have the terms in the observations returned as a dictionary
    # rather than a concatenated tensor
    env_cfg.observations.policy.concatenate_terms = False


    # create markers
    if args_cli.debug:
        traj_visualizer = define_markers()
        pick_pose_visualizer = define_markers()

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs_dict, _ = env.reset()  # reset environment


    # create controller
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(pos_sensitivity=0.004, rot_sensitivity=1)
    elif  args_cli.teleop_device.lower() == "keyboard_dual_arm":
        teleop_interface = Se3Keyboard_Dual(pos_sensitivity=0.004, rot_sensitivity=1)
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(pos_sensitivity=0.05, rot_sensitivity=0.005)
    else:
        raise ValueError(f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse'.")
    # print helper
    print(teleop_interface)

    def f2_func():
        global action_res
        # task_1.scene_reader(obs_dict)
        episode.scene_reader(obs_dict)
        robot_info = {
            "robot_id": 0,  # must
            "safe_pos": [0.42, 0.16, 0.50], # config here
        }
        move_to_waypoint = TranslateToFixed(to_pos="safe_pos", step=0.004, docs="safe position")
        move_to_waypoint.set_robot_info(robot_info)
        # action_res = task_1.execute_atom(move_to_waypoint)
        action_res = episode.execute_atom(move_to_waypoint)
        print("[Info] Pressed F2 defined function, planning interrupted")

    def f3_func():
        global action_res
        episode.scene_reader(obs_dict)
        robot_info = {
            "robot_id": 0,  # must
            "safe_rot": [180, 0, 90], # config here
        }
        rotate_to_waypoint = RotateToFixed(to_rot="safe_rot", step=1, docs="safe rotation")
        rotate_to_waypoint.set_robot_info(robot_info)
        action_res = episode.execute_atom(rotate_to_waypoint)
        print("[Info] Pressed F3 defined function, planning interrupted")

    def f4_func():
        global action_res
        episode.scene_reader(obs_dict)
        hm = HumanControl()
        action_res = episode.execute_atom(hm)
        print("[Info] Pressed F4 defined function, planning interrupted")

    def episode_success():
        global is_success
        global save_episode
        is_success = True
        save_episode = True

    def episode_fail():
        global is_success
        global save_episode
        is_success = False
        save_episode = True

    def execute_action():
        global action_res
        # task_1.scene_reader(obs_dict)
        # action_res = task_1.execute()
        episode.scene_reader(obs_dict)
        action_res = episode.execute()

    # add teleportation key for env reset
    teleop_interface.add_callback("NUMPAD_1", episode_success)
    teleop_interface.add_callback("NUMPAD_2", execute_action)
    teleop_interface.add_callback("NUMPAD_3", episode_fail)
    teleop_interface.add_callback("F2", f2_func) # example move to safe
    teleop_interface.add_callback("F3", f3_func) # example rotate to safe
    teleop_interface.add_callback("F4", f4_func) # example rotate to safe

    # specify directory for logging experiments, dump the configuration into log-directory
    log_dir = os.path.join("./logs/diffusion_policy", args_cli.task)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    zarr_path = os.path.join(log_dir, "replay_buffer.zarr")

    # # create data-collector
    replay_buffer = ReplayBuffer.create_empty_numpy()

    def get_obs_dict(actions):
        # Simulate interacting with the environment and getting obs_dict
        obs_dict, _, _, _, _ = env.step(actions)
        return obs_dict

    def prepare_environment(init_actions):
        """
        Prepares the environment by performing initialization actions for a number of iterations.
        """
        obs_dict, _ = env.reset()  # reset environment
        teleop_interface.reset()  # reset interfaces
        print("[Info] Prepare env, please don't move .... ")
        for _ in range(200):
            obs_dict = get_obs_dict(init_actions)
        print("[Info] You can move now")
        return obs_dict

    # get camera info
    re_cam_name = "right_eye_camera"
    le_camera = "left_eye_camera"
    eih_cam_name = "eye_in_hand_camera"
    tv_cam = env.scene.sensors[re_cam_name]
    image_h, image_w = tv_cam.image_shape
    # eih_cam = env.scene.sensors[eih_cam_name]
    # print(tv_cam.cfg.prim_path)
    # print(tv_cam.render_product_paths)

    # camera params
    tv_cam_intrinsic = tv_cam.data.intrinsic_matrices[0] # (3, 3)
    tv_cam_pos_w = tv_cam.data.pos_w[0]  # (3,)
    tv_cam_quat_w = tv_cam.data.quat_w_opengl  # camera in usd format, cannot use tv_cam.data.quat_w_world
    tv_cam_matrix_w = matrix_from_quat(tv_cam_quat_w)[0]  # (3, 3)
    # default world frame is identical to the robot frame (inverse is the camera extrinsic
    tv_cam_matrix_inv_w, tv_cam_pos_inv_w = inverse_transformation(tv_cam_matrix_w, tv_cam_pos_w)
    tv_camera_extrinsic = torch.eye(4)
    tv_camera_extrinsic[:3, :3] = tv_cam_matrix_inv_w
    tv_camera_extrinsic[:3, 3] = tv_cam_pos_inv_w

    print("[Info] image_h ", image_h)
    print("[Info] image_w ", image_w)
    print("[Info] tv_camera_intrinsic ", tv_cam_intrinsic)
    print("[Info] tv_camera_extrinsic ", tv_camera_extrinsic)

    ############################ replicator #############################
    if args_cli.replicator:
        table_name = "table"
        table = env.scene[table_name]
        table_path = table.prim_paths[0]
        table_prim = rep.get.prims(path_pattern=table_path)
        def base_table_random():
            with table_prim:
                rep.randomizer.color(colors=rep.distribution.uniform((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
            return table_prim.node
        rep.randomizer.register(base_table_random, override=True)
        with rep.trigger.on_custom_event(event_name="Randomize!"):
            rep.randomizer.base_table_random()

    #######################################################################

    def define_arm_init_pose(init_arm_pos, init_arm_rpy):
        init_pos = torch.tensor([init_arm_pos], dtype=torch.float, device=env.device).repeat(env.num_envs, 1)
        init_rpy = torch.tensor([init_arm_rpy], dtype=torch.float, device=env.device).repeat(env.num_envs, 1)
        init_g = torch.tensor([[1]], dtype=torch.float, device=env.device).repeat(env.num_envs, 1)
        init_quat = quat_from_euler_xyz(init_rpy)
        init_actions = torch.concat([init_pos, init_quat, init_g], dim=-1)
        return init_pos, init_rpy, init_g, init_actions

    # define frank default position (xyz,wxyz,g)
    INIT_RIGHT_ARM_POS = [0.22, 0.16, 0.50]
    INIT_RIGHT_ARM_RPY = [180, 0, 90]  # [180, 0, 90]
    init_right_pos, init_right_rpy, init_right_g, init_right_actions = define_arm_init_pose(INIT_RIGHT_ARM_POS, INIT_RIGHT_ARM_RPY)

    # INIT_LEFT_ARM_POS = [0.22, -0.16, 0.50] # in robot left frame
    INIT_LEFT_ARM_POS = [0.22, 0.46, 0.50] # in world frame which is robot right
    INIT_LEFT_ARM_RPY = [180, 0, 90]
    init_left_pos, init_left_rpy, init_left_g, init_left_actions = define_arm_init_pose(INIT_LEFT_ARM_POS, INIT_LEFT_ARM_RPY)

    ############ left arm frame to self frame, action manage ik solver needs self frame instead world frame ############
    left_robot_position_w = obs_dict["policy"]["left_robot_position_w"]
    left_robot_pos_in_w, _ = subtract_frame_transforms(left_robot_position_w[:, :3],
                                                       left_robot_position_w[:, 3:7],
                                                       init_left_actions[:, :3])
    init_left_actions[:, :3] = left_robot_pos_in_w
    ####################################################################################################################

    # right + left action
    init_actions = torch.concat([init_right_actions, init_left_actions], dim=-1)
    # init_actions = torch.concat([init_right_actions], dim=-1)
    print("[Info] Init action : ", init_actions)
    print("[Info] Init action shape : ", init_actions.shape)

    # init environment
    obs_dict = prepare_environment(init_actions)
    right_abs_action_pos = init_right_pos.clone()
    right_abs_action_rpy = init_right_rpy.clone()
    left_abs_action_pos = init_left_pos.clone()
    left_abs_action_rpy = init_left_rpy.clone()

    global is_success, save_episode
    global action_res

    # construct robot
    robot_right = {
        "robot_id": 0, # must
        "init_pos": [0.22, 0.16, 0.35],
        "init_rot": INIT_RIGHT_ARM_RPY,
        "bin_pos": [0.48, -0.08, 0.42],  # obs_dict["policy"]["bin_pose"][:, :3].squeeze()
    }

    robot_left = {
        "robot_id": 1, # must
        "init_pos": INIT_LEFT_ARM_POS,
        "init_rot": INIT_LEFT_ARM_RPY,
        "bin_pos": [0.48, -0.08, 0.42], # obs_dict["policy"]["bin_pose"][:, :3].squeeze()
    }
    plate = {
        "category" :"plate",
        "pick_points": [
            {
                "name": "wider_grasp",
                "offset": {"pos": [0, 0, 0.002], "quat": [1, 0, 0, 0]},
                "score": 1,
                "waypoints": [
                    {"name": "waypoint_1", "offset": {"pos": [0, 0, 0.2], "rot": [0, 0, 0]}, "score": 1},
                ]
            },
            # {
            #     "name": "front_grasp",
            #     "offset": {"pos": [0.07, 0, -0.005], "quat": [0.70711, 0, 0, 0.70711]},
            #     "score": 1,
            #     "waypoints": [
            #         {"name": "waypoint_1", "offset": {"pos": [0, 0, 0.05], "rot": [0, 0, 0]}, "score": 1},
            #     ]
            # },
            # {
            #     "name": "end_grasp",
            #     "offset": {"pos": [-0.07, 0, -0.005], "quat": [0.70711, 0, 0, 0.70711]},
            #     "score": 1,
            #     "waypoints": [
            #         {"name": "waypoint_1", "offset": {"pos": [0, 0, 0.05], "rot": [0, 0, 0]}, "score": 1},
            #     ]
            # },
        ],
    }
    mug = {
        "category": "mug",
        "pick_points": [
            {
                "name": "mug_up_handle",
                "offset": {"pos": [0.06, 0, 0.005], "quat": [0.70711, 0, 0, 0.70711]},
                "score": 0.8,
                "waypoints": [
                    {"name": "waypoint_1", "offset": {"pos": [0, 0, 0.15], "rot": [0, 0, 0]}, "score": 1},
                    {"name": "waypoint_2", "offset": {"pos": [0, 0, 0.1], "rot": [0, 0, 0]}, "score": 1},
                ]
            },
            {
                "name": "mug_handle_opposite",
                "offset": {"pos": [-0.05, 0, 0.005], "quat": [0.70711, 0, 0, 0.70711]},
                "score": 1,
                "waypoints": [
                    {"name": "waypoint_1", "offset": {"pos": [0, 0, 0.15], "rot": [0, 0, 0]}, "score": 1},
                    {"name": "waypoint_2", "offset": {"pos": [0, 0, 0.1], "rot": [0, 0, 0]}, "score": 1},
                ]
            },
            {
                "name": "mug_up_1",
                "offset": {"pos": [0, -0.05, 0.005], "quat": [1, 0, 0 ,0]},
                "score": 0.9,
                "waypoints": [
                    {"name": "waypoint_1", "offset": {"pos": [0, 0, 0.15], "rot": [0, 0, 0]}, "score": 1},
                    {"name": "waypoint_2", "offset": {"pos": [0, 0, 0.1], "rot": [0, 0, 0]}, "score": 1},
                ]
            },
            {
                "name": "mug_up_2",
                "offset": {"pos": [0, 0.05, 0.005], "quat": [1, 0, 0, 0]},
                "score": 0.9,
                "waypoints": [
                    {"name": "waypoint_1", "offset": {"pos": [0, 0, 0.15], "rot": [0, 0, 0]}, "score": 1},
                    {"name": "waypoint_2", "offset": {"pos": [0, 0, 0.1], "rot": [0, 0, 0]}, "score": 1},
                ]
            },
            {
                "name": "mug_bottom_0",
                "offset": {"pos": [0, 0, -0.025], "quat": [0, 1, 0, 0]},
                "score": 0.8,
                "waypoints": [
                    {"name": "waypoint_1", "offset": {"pos": [0, 0, 0.15], "rot": [0, 0, 0]}, "score": 1},
                    {"name": "waypoint_2", "offset": {"pos": [0, 0, 0.1], "rot": [0, 0, 0]}, "score": 1},
                ]
            },
            {
                "name": "mug_bottom_1",
                "offset": {"pos": [0, 0, -0.025], "quat": [0, 0, -1, 0]},
                "score": 0.8,
                "waypoints": [
                    {"name": "waypoint_1", "offset": {"pos": [0, 0, 0.15], "rot": [0, 0, 0]}, "score": 1},
                    {"name": "waypoint_2", "offset": {"pos": [0, 0, 0.1], "rot": [0, 0, 0]}, "score": 1},
                ]
            },
            # {
            #     "name": "mug_side_0",
            #     "offset": {"pos": [0, -0.015, 0], "quat": [0.5, 0.5, -0.5, 0.5]},
            #     "score": 0.9,
            #     "waypoints": [
            #         {"name": "waypoint_1", "offset": {"pos": [0, 0, 0.15], "rot": [0, 0, 0]}, "score": 1},
            #         {"name": "waypoint_2", "offset": {"pos": [0, 0, 0.1], "rot": [0, 0, 0]}, "score": 1},
            #     ]
            # },
            # {
            #     "name": "mug_side_1",
            #     "offset": {"pos": [0.01, -0.015, 0], "quat": [0.35355, 0.61237, -0.35355, 0.61237]},
            #     "score": 0.9,
            #     "waypoints": [
            #         {"name": "waypoint_1", "offset": {"pos": [0, 0, 0.15], "rot": [0, 0, 0]}, "score": 1},
            #         {"name": "waypoint_2", "offset": {"pos": [0, 0, 0.1], "rot": [0, 0, 0]}, "score": 1},
            #     ]
            # },
            # {
            #     "name": "mug_side_2",
            #     "offset": {"pos": [0, 0.015, 0], "quat": [0.5, -0.5, 0.5, 0.5]},
            #     "score": 0.9,
            #     "waypoints": [
            #         {"name": "waypoint_1", "offset": {"pos": [0, 0, 0.15], "rot": [0, 0, 0]}, "score": 1},
            #         {"name": "waypoint_2", "offset": {"pos": [0, 0, 0.1], "rot": [0, 0, 0]}, "score": 1},
            #     ]
            # },
            # {
            #     "name": "mug_side_3",
            #     "offset": {"pos": [0.005, 0.01, 0], "quat": [0.65328, -0.2706, 0.65328, 0.2706]},
            #     "score": 0.9,
            #     "waypoints": [
            #         {"name": "waypoint_1", "offset": {"pos": [0, 0, 0.15], "rot": [0, 0, 0]}, "score": 1},
            #         {"name": "waypoint_2", "offset": {"pos": [0, 0, 0.1], "rot": [0, 0, 0]}, "score": 1},
            #     ]
            # },
            {
                "name": "mug_side_4_handle",
                "offset": {"pos": [0.045, 0.0, 0], "quat": [0.70711, 0, 0.70711, 0]},
                "score": 0.9,
                "waypoints": [
                    {"name": "waypoint_1", "offset": {"pos": [0, 0, 0.15], "rot": [0, 0, 0]}, "score": 1},
                    {"name": "waypoint_2", "offset": {"pos": [0, 0, 0.1], "rot": [0, 0, 0]}, "score": 1},
                ]
            },
        ],
    }
    spoon = {
        "category" :"spoon",
        "pick_points": [
            {
                "name": "upward",
                "offset": {"pos": [0, 0, 0.002], "quat": [1, 0, 0, 0]},
                "score": 1,
                "waypoints": [
                    {"name": "waypoint_1", "offset": {"pos": [0, 0, 0.1], "rot": [0, 0, 0]}, "score": 1},
                    {"name": "waypoint_2", "offset": {"pos": [0, 0, 0.05], "rot": [0, 0, 0]}, "score": 1},
                ]
            },
        ],
    }
    bowl = {
        "category" :"bowl",
        "pick_points": [
            {
                "name": "bowl_right",
                "offset": {"pos": [0, 0.07, 0.02], "quat": [1, 0, 0, 0]},
                "score": 1,
                "waypoints": [
                    {"name": "waypoint_2", "offset": {"pos": [0, 0, 0.05], "rot": [0, 0, 0]}, "score": 1},
                ]
            },
            {
                "name": "bow_left",
                "offset": {"pos": [0, -0.07, 0.02], "quat": [1, 0, 0, 0]},
                "score": 1,
                "waypoints": [
                    {"name": "waypoint_2", "offset": {"pos": [0, 0, 0.05], "rot": [0, 0, 0]}, "score": 1},
                ]
            },
        ],

    }
    broom = {
        "category" :"broom",
        "pick_points": [
            {
                "name": "broom_upward",
                "offset": {"pos": [0, 0, 0], "quat": [1, 0, 0, 0]},
                "score": 1,
                "waypoints": [
                    {"name": "waypoint_1", "offset": {"pos": [0, 0, 0.1], "rot": [0, 0, 0]}, "score": 1},
                ]
            },
        ],

    }

    episode = Episode("house keeper")

    # construct Task
    task_1 = Task('Order the table')
    task_2 = Task('Sweep the tabel')

    manipulation_1 = Manipulation(robot=robot_right, target_object=plate)
    manipulation_2 = Manipulation(robot=robot_right, target_object=spoon)
    manipulation_3 = Manipulation(robot=robot_right, target_object=mug)
    manipulation_4 = Manipulation(robot=robot_right, target_object=bowl)
    manipulation_6 = Manipulation(robot=robot_right, target_object=broom)

    # construct Action
    pick_action = Action(action_name="pick")
    pick_action.set_auto_next(True)
    place_action = Action(action_name="place")
    place_action.set_auto_next(True)

    # construct ActionAtom
    gripper_open = GripperOpen()
    move_to_object = MoveTo()
    trans_to_object = TranslateToObject(docs="")
    rotate_to_object = RotateToObject(docs="")
    move_to_safe = TranslateToFixed(to_pos="init_pos", step=0.004, docs="init position")
    rotate_to_safe = RotateToFixed(to_rot="init_rot", step=1, docs="init rotation")
    move_to_bin = TranslateToFixed(to_pos="bin_pos", step=0.004, docs="bin top")
    gripper_close = GripperClose()
    human_control = HumanControl()  # human control
    wait = Wait()

    # 添加原子动作，比如打开夹爪、移动到目标、关闭夹爪
    pick_action.add_atom(gripper_open)
    pick_action.add_atom(move_to_object)
    pick_action.add_atom(wait)
    # pick_action.add_atom(human_control)
    pick_action.add_atom(gripper_close)
    pick_action.add_atom(move_to_safe)
    pick_action.add_atom(rotate_to_safe)

    place_action.add_atom(move_to_bin)
    place_action.add_atom(gripper_open)
    place_action.add_atom(move_to_safe)

    # add action into manipulation
    manipulation_1.add_action(pick_action)
    manipulation_1.add_action(place_action)
    manipulation_2.add_action(pick_action)
    manipulation_2.add_action(place_action)
    manipulation_3.add_action(pick_action)
    manipulation_3.add_action(place_action)
    manipulation_4.add_action(pick_action)
    manipulation_4.add_action(place_action)
    manipulation_6.add_action(pick_action)

    # add manipulation to task viewer
    # pick and place
    task_1.add_manipulation(manipulation_1)
    task_1.add_manipulation(manipulation_2)
    task_1.add_manipulation(manipulation_3)
    task_1.add_manipulation(manipulation_4)
    # sweep the table
    task_2.add_manipulation(manipulation_6)

    episode.add_task(task_1)
    episode.add_task(task_2)

    frame_cnt = 0
    is_success, save_episode = False, False
    tv_images_list, eih_images_list, le_images_list = [], [], []
    robot_eef_list, action_list = [], []
    if args_cli.debug:
        right_abs_action_pos_debug, right_abs_action_quat_debug = [], []

    # auto start
    execute_action()

    episode_cnt = 0
    # simulate environment -- run everything in inference mode
    with ((contextlib.suppress(KeyboardInterrupt) and torch.inference_mode())):
        while episode_cnt < args_cli.num_demos:
            # get keyboard command
            right_arm_delta_pose, right_arm_gripper_command, left_arm_delta_pose, left_arm_gripper_command = teleop_interface.advance()
            # convert to torch
            right_arm_delta_pose = torch.tensor(right_arm_delta_pose, dtype=torch.float, device=env.device).repeat(env.num_envs, 1)
            left_arm_delta_pose = torch.tensor(left_arm_delta_pose, dtype=torch.float, device=env.device).repeat(env.num_envs, 1)
            # compute actions based on environment
            right_actions = pre_process_actions(right_arm_delta_pose, right_arm_gripper_command)
            left_actions = pre_process_actions(left_arm_delta_pose, left_arm_gripper_command)

            right_action_xyz = right_actions[:, :3]
            right_action_rpy = right_actions[:, 3:6]
            right_action_g = right_actions[:, 6:7]

            left_action_xyz = left_actions[:, :3]
            left_action_rpy = left_actions[:, 3:6]
            left_action_g = left_actions[:, 6:7]

            # -- obs:
            tv_image = obs_dict["policy"][re_cam_name]
            eih_image = obs_dict["policy"][eih_cam_name]
            le_image = obs_dict["policy"][le_camera]
            right_robot_eef_pos = obs_dict["policy"]["right_robot_eef_pose"][:, :3]
            right_robot_eef_quat = obs_dict["policy"]["right_robot_eef_pose"][:, 3:7]
            right_robot_eef_rpy = euler_xyz_from_quat(right_robot_eef_quat)  # [:, 3] [-180, 180]
            right_robot_eef_width = obs_dict["policy"]["right_robot_eef_width"]
            right_robot_eef = torch.concat([right_robot_eef_pos, right_robot_eef_rpy, right_robot_eef_width], dim=-1) # [1, 8 = xyz, quat, g_width]
            # gripper_actions = obs_dict["policy"]["gripper_actions"] # [-1, 1]

            # if frame_cnt % 10 == 0:
            #     print("[Todo] Invoke detector 10Hz, maybe for safety check")

            # 0 is right robot arm, 1 is left robot arm
            arm_id = action_res["robot_id"]
            primitive_act = action_res["primitive_act"]
            auto_next = action_res["auto_next"]
            episode_end = action_res["episode_end"]
            pick_poses = action_res["pick_poses"]
            act_type, act = next(primitive_act)
            if args_cli.debug:
                if pick_poses:
                    pick_point_pos = [pp['offset']['pos'] for pp in pick_poses]
                    pick_point_quat = [pp['offset']['quat'] for pp in pick_poses]
                    pick_point_pos = torch.stack(pick_point_pos)
                    pick_point_quat = torch.stack(pick_point_quat)
                    pick_pose_visualizer.visualize(pick_point_pos, pick_point_quat)
                    pick_pose_visualizer.set_visibility(True)
                else:
                    pick_pose_visualizer.set_visibility(False)

            if "Translate" in act_type:
                if arm_id == 0:
                    right_abs_action_pos = act.unsqueeze(0)
                elif arm_id == 1:
                    left_abs_action_pos = act.unsqueeze(0)
                else:
                    raise TypeError(f"Unknown action position: {arm_id}")
            elif "Rotate" in act_type:
                if arm_id == 0:
                    right_abs_action_rpy = act.unsqueeze(0)
                elif arm_id == 1:
                    left_abs_action_rpy = act.unsqueeze(0)
                else:
                    raise TypeError(f"Unknown action position: {arm_id}")
            elif act_type in ["GripperOpen", "GripperClose"]:
                teleop_interface.reset()
                if arm_id == 0:
                    right_action_g = act.unsqueeze(0)
                    teleop_interface._close_right_arm_gripper = False if act == 1 else True
                elif arm_id == 1:
                    left_action_g = act.unsqueeze(0)
                    teleop_interface._close_left_arm_gripper = False if act == 1 else True
                else:
                    raise TypeError(f"Unknown action position: {arm_id}")
            elif act_type == "Wait":
                pass
            elif act_type == "HumanControl":
                pass
            else:
                raise TypeError(f"Unknown action type: {act_type}")

            if primitive_act.has_ended():
                if episode_end: episode_success()
                if auto_next: execute_action()
                # right arm
                right_abs_action_pos += right_action_xyz
                right_abs_action_rpy += right_action_rpy
                # left arm
                left_abs_action_pos += left_action_xyz
                left_abs_action_rpy += left_action_rpy

            # -- actions
            right_abs_action_label = torch.concat([right_abs_action_pos, right_abs_action_rpy, right_action_g], dim=-1)
            left_abs_action_label = torch.concat([left_abs_action_pos, left_abs_action_rpy, left_action_g], dim=-1)
            # save dual arm
            # action_abs_label = torch.concat([right_abs_action_label, left_abs_action_label], dim=-1)  # 14-dims
            # save single arm
            action_abs_label = right_abs_action_label  # 8-dims

            if frame_cnt % args_cli.skip_frame == 0:
                tv_images_list.append(tv_image)
                eih_images_list.append(eih_image)
                le_images_list.append(le_image)
                robot_eef_list.append(right_robot_eef)
                action_list.append(action_abs_label)

            # perform action on environment
            # right & left arm
            right_abs_action_quat = quat_from_euler_xyz(right_abs_action_rpy)
            left_abs_action_quat = quat_from_euler_xyz(left_abs_action_rpy)
            right_abs_actions = torch.concat([right_abs_action_pos, right_abs_action_quat, right_action_g], dim=-1)
            left_abs_actions = torch.concat([left_abs_action_pos, left_abs_action_quat, left_action_g], dim=-1)
            # visualizer debug
            if args_cli.debug:
                right_abs_action_pos_debug.append(right_abs_action_pos.clone())
                right_abs_action_quat_debug.append(right_abs_action_quat.clone())
                traj_visualizer.visualize(torch.concat(right_abs_action_pos_debug, dim=0)[-50:],
                                        torch.concat(right_abs_action_quat_debug, dim=0)[-50:])
            ############ left arm frame to self frame, action manage ik solver needs self frame instead world frame ####
            left_robot_pos_in_w, _ = subtract_frame_transforms(left_robot_position_w[:, :3],
                                                               left_robot_position_w[:, 3:7],
                                                               left_abs_actions[:, :3])
            left_abs_actions[:, :3] = left_robot_pos_in_w
            ############################################################################################################
            actions = torch.concat([right_abs_actions, left_abs_actions], dim=-1)
            obs_dict, converged, steps = run_env_until_converge(get_obs_dict,
                                                                actions,
                                                                pos_error_thresh=0.006,
                                                                axis_angle_error_thresh=0.5,
                                                                max_attempts=4,
                                                                verbose=False)
            frame_cnt += 1

            # check that simulation is stopped or not
            if env.unwrapped.sim.is_stopped():
                break

            if save_episode:
                print("[Info] episode status : ", is_success)
                video_dir = os.path.join(log_dir, "videos", str(episode_cnt))
                os.makedirs(video_dir, exist_ok=True)

                if is_success:
                    ############################## save zarr format ####################################
                    print("[Info] Action length ", len(action_list))
                    __robot_eef_list = torch.concat(robot_eef_list, dim=0) # [Action, 7]
                    episode_data = {
                        'action': torch.concat(action_list, dim=0).cpu().numpy(), # [Action, 7]
                        'robot_eef_pose': __robot_eef_list[:, :6].cpu().numpy(),  # [Action, 6]
                        'gripper_position': __robot_eef_list[:, 6:7].cpu().numpy()  # [Action, 1]
                    }
                    replay_buffer.add_episode(episode_data)

                    ############################## save demo video #####################################
                    save_video(eih_images_list, video_dir, '1.mp4', args_cli.fps)
                    save_video(le_images_list, video_dir, '2.mp4', args_cli.fps)
                    save_video(tv_images_list, video_dir, '3.mp4', args_cli.fps)

                    ############################## save debug video ####################################
                    if args_cli.debug:
                        tv_images_debug_list = []
                        for image, eef_pose_w, action_w in zip(tv_images_list, robot_eef_list, action_list):
                            if isinstance(image, torch.Tensor): image = image.cpu().numpy()[0]
                            assert image.dtype == np.uint8
                            # Use the function to project end-effector and action poses to 2D and draw markers
                            image = project_pose_to_2d(image, tv_cam_matrix_inv_w, tv_cam_pos_inv_w, eef_pose_w,
                                                       tv_cam_intrinsic, image_w, (0, 0, 255))  # Blue for end-effector
                            image = project_pose_to_2d(image, tv_cam_matrix_inv_w, tv_cam_pos_inv_w, action_w,
                                                       tv_cam_intrinsic, image_w, (255, 0, 0))  # Red for action
                            tv_images_debug_list.append(image)
                        save_video(tv_images_debug_list, video_dir, '3_debug.mp4', args_cli.fps)
                        tv_images_debug_list.clear()

                    episode_cnt += 1

                frame_cnt = 0
                is_success, save_episode = False, False
                tv_images_list, eih_images_list, le_images_list = [], [], []
                robot_eef_list, action_list = [], []
                if args_cli.debug:
                    right_abs_action_pos_debug, right_abs_action_quat_debug = [], []

                print("video saved {} / {} ".format(episode_cnt, args_cli.num_demos))
                print("env reset ")

                # exit
                if episode_cnt < args_cli.num_demos:
                    obs_dict = prepare_environment(init_actions)
                    right_abs_action_pos = init_right_pos.clone()
                    right_abs_action_rpy = init_right_rpy.clone()
                    left_abs_action_pos = init_left_pos.clone()
                    left_abs_action_rpy = init_left_rpy.clone()

                    rep.utils.send_og_event("Randomize!")  # new rep
                    task_1.reset_controller()
                    execute_action()

    print("finish data collection, exit simulator")
    # close the simulator
    replay_buffer.save_to_path(zarr_path=zarr_path, chunk_length=-1)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
