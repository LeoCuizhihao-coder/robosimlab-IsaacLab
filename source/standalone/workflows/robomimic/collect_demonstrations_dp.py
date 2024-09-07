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
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Rel-dp", help="Name of the task.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--num_demos", type=int, default=1, help="Number of episodes to store in the dataset.")
parser.add_argument("--fps", type=int, default=30, help="diffusion video fps.")
parser.add_argument("--skip_frame", type=int, default=2, help="save every N frame")
parser.add_argument("--replicator", type=bool, default=False, help="enable table replicator")
parser.add_argument("--gripper", type=bool, default=True, help="collect gripper status, open 1, close -1")
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

from omni.isaac.lab.devices import Se3Keyboard, Se3SpaceMouse
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg
from omni.isaac.lab.utils.math import project_points, euler_xyz_from_quat, matrix_from_quat

import omni.replicator.core as rep

from diffusion_policy.common.replay_buffer import ReplayBuffer


def pre_process_actions(delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    if "Reach" in args_cli.task:
        # note: reach is the only one that uses a different action space
        # compute actions
        return delta_pose
    else:
        # resolve gripper command
        gripper_vel = torch.zeros((delta_pose.shape[0], 1), dtype=torch.float, device=delta_pose.device)
        gripper_vel[:] = -1 if gripper_command else 1
        # compute actions
        return torch.concat([delta_pose, gripper_vel], dim=1)


def inverse_transformation(R, t):
    """计算位姿矩阵的逆"""
    R_inv = R.t()  # 旋转矩阵的逆是它的转置
    t_inv = -torch.matmul(R_inv, t)
    return R_inv, t_inv


def get_object_pose_in_camera_frame(R_cam_inv, t_cam_inv, R_obj, t_obj):
    """计算物体在相机坐标系下的位姿"""
    # 物体相对于相机 = 相机逆位姿 * 物体位姿
    R_obj_to_cam = torch.matmul(R_cam_inv, R_obj)
    t_obj_to_cam = torch.matmul(R_cam_inv, t_obj) + t_cam_inv

    return R_obj_to_cam, t_obj_to_cam


def convert_quat_to_euler(quat):
    roll, pitch, yaw = euler_xyz_from_quat(quat)
    # quat to euler in xyz-format degree unit
    euler_rad = torch.concat([roll.unsqueeze(1), pitch.unsqueeze(1), yaw.unsqueeze(1)],
                             dim=-1)  # [Action, 3]
    euler_deg = torch.rad2deg(euler_rad)  # [Action, 3]
    return euler_deg


def main():
    """Collect demonstrations from the environment using teleop interfaces."""
    assert (args_cli.task == "Isaac-Lift-Cube-Franka-IK-Abs-dp" or args_cli.task == "Isaac-Lift-Cube-Franka-IK-Rel-dp"), \
        "Only 'Isaac-Lift-Cube-Franka-IK-Abs-v0' is supported currently."

    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    # modify configuration such that the environment runs indefinitely
    # until goal is reached
    env_cfg.terminations.time_out = None
    # set the resampling time range to large number to avoid resampling
    env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
    # we want to have the terms in the observations returned as a dictionary
    # rather than a concatenated tensor
    env_cfg.observations.policy.concatenate_terms = False

    # add termination condition for reaching the goal otherwise the environment won't reset
    env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # create controller
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(pos_sensitivity=0.04, rot_sensitivity=0.08)
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(pos_sensitivity=0.05, rot_sensitivity=0.005)
    else:
        raise ValueError(f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse'.")
    # print helper
    print(teleop_interface)

    # specify directory for logging experiments
    log_dir = os.path.join("./logs/robomimic", args_cli.task)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    zarr_path = os.path.join(log_dir, "replay_buffer.zarr")

    # # create data-collector
    replay_buffer = ReplayBuffer.create_empty_numpy()

    # reset environment
    obs_dict, _ = env.reset()

    # reset interfaces
    teleop_interface.reset()

    print("Prepare env, plz dont move")
    for i in range(50):
        # T|R, gripper state
        actions = torch.tensor([[0., 0., 0., 0., 0., 0., 1.]], dtype=torch.float, device=env.device)
        actions = actions.repeat(env.num_envs, 1)
        obs_dict, rewards, terminated, truncated, info = env.step(actions)
    init_robot_eef_pos = obs_dict["policy"]["robot_eef_pos"]
    init_robot_eef_quat = obs_dict["policy"]["robot_eef_quat"]
    init_robot_eef_position = torch.concat([init_robot_eef_pos, init_robot_eef_quat], dim=-1)
    print("You can move now")
    print("Robot init robot eef pos : ", init_robot_eef_pos)
    print("Robot init robot eef quat : ", init_robot_eef_quat)

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

    # add teleoperation key for env reset
    teleop_interface.add_callback("L", episode_fail)
    teleop_interface.add_callback("J", episode_success)

    # get camera info
    tv_cam_name = "top_view_camera"
    eih_cam_name = "eye_in_hand_camera"
    tv_cam = env.scene.sensors[tv_cam_name]
    image_h, image_w = tv_cam.image_shape
    # eih_cam = env.scene.sensors[eih_cam_name]
    # print(tv_cam)
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
        # # get background info
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

        # # get plane info
        # plane_name = "plane"
        # plane = env.scene[plane_name]
        # plane_path = plane.prim_paths[0]
        # def base_plane_colors():
        #     plane_prim = rep.get.prims(path_pattern=plane_path)
        #     with plane_prim:
        #         rep.randomizer.color(colors=rep.distribution.uniform((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
        #     return plane_prim.node
        # rep.randomizer.register(base_plane_colors, override=True)
        # # with rep.trigger.on_frame():
        # with rep.trigger.on_custom_event(event_name="Randomize!"):
        #     rep.randomizer.base_plane_colors()
    #######################################################################

    # save mp4 video for diffusion policy training
    global is_success
    global save_episode
    global tv_images_list
    global eih_images_list
    # global increase_robot_eef_position
    frame_cnt = 0
    episode_cnt = 0
    is_success = False
    save_episode = False
    tv_images_list = []
    eih_images_list = []
    action_list = []
    robot_eef_pose_list = []
    # increase_robot_eef_position = init_robot_eef_position.clone()
    # simulate environment -- run everything in inference mode
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while episode_cnt < args_cli.num_demos:
            # get keyboard command
            delta_pose, gripper_command = teleop_interface.advance()
            # convert to torch
            delta_pose = torch.tensor(delta_pose, dtype=torch.float, device=env.device).repeat(env.num_envs, 1)
            # compute actions based on environment
            actions = pre_process_actions(delta_pose, gripper_command)

            if frame_cnt % args_cli.skip_frame == 0:
                # The observations need to be recollected.
                # store signals before stepping
                # -- obs
                # -- camera image
                tv_images_list.append(obs_dict["policy"][tv_cam_name])
                eih_images_list.append(obs_dict["policy"][eih_cam_name])

                # -- ee pose
                robot_eef_pos = obs_dict["policy"]["robot_eef_pos"]
                robot_eef_quat = obs_dict["policy"]["robot_eef_quat"]
                if args_cli.gripper:
                    robot_eef_position = torch.concat([robot_eef_pos, robot_eef_quat, actions[:, -1:]], dim=-1)  # [1, 8=xyz+quat+gripper]
                else:
                    robot_eef_position = torch.concat([robot_eef_pos, robot_eef_quat], dim=-1)  # [1, 7=xyz+quat]
                robot_eef_pose_list.append(robot_eef_position)

                # -- actions (relative action)
                abs_action_pos = robot_eef_pos + actions[:, :3]
                if args_cli.gripper:
                    abs_action_position = torch.concat([abs_action_pos, robot_eef_quat, actions[:, -1:]], dim=-1)  # [1, 8=xyz+quat+gripper]
                else:
                    abs_action_position = torch.concat([abs_action_pos, robot_eef_quat], dim=-1)  # [1, 7=xyz+quat]
                action_list.append(abs_action_position)
                # print("\n")

            # perform action on environment
            # print("actions ", actions)
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            frame_cnt += 1

            # check that simulation is stopped or not
            if env.unwrapped.sim.is_stopped():
                break

            # robomimic only cares about policy observations
            # -- next ee pose
            # next_robot_eef_position = torch.concat([obs_dict["policy"]["robot_eef_pos"],
            #                                    obs_dict["policy"]["robot_eef_quat"]], dim=-1)

            if save_episode:
                print("[Info] episode status : ", is_success)
                # create video folder
                video_dir = os.path.join(log_dir, "videos", str(episode_cnt))
                os.makedirs(video_dir, exist_ok=True)

                if is_success:
                    print("[Info] Action length ", len(action_list))
                    # eep pose (euler format)
                    robot_eef_poses = torch.concat(robot_eef_pose_list, dim=0) # [Action, 7 or 8]
                    robot_eef_euler_deg = convert_quat_to_euler(robot_eef_poses[:, 3:7]) # [3]
                    if args_cli.gripper:
                        robot_eef_poses = torch.concat([robot_eef_poses[:, :3],
                                                        robot_eef_euler_deg,
                                                        robot_eef_poses[:, 7:8]], dim=-1)  # [Action=7]
                    else:
                        robot_eef_poses = torch.concat([robot_eef_poses[:, :3],
                                                        robot_eef_euler_deg], dim=-1)  # [Action=6]

                    # action (euler format)
                    action_poses = torch.concat(action_list, dim=0) # [Action, 7]
                    action_euler_deg = convert_quat_to_euler(action_poses[:, 3:7]) # [3]
                    if args_cli.gripper:
                        action_poses = torch.concat([action_poses[:, :3],
                                                    action_euler_deg,
                                                    action_poses[:, 7:8]], dim=-1)  # [Action=7]
                    else:
                        action_poses = torch.concat([action_poses[:, :3],
                                                    action_euler_deg], dim=-1)  # [Action=6]

                    episode = {
                        'action': action_poses.cpu().numpy(), # [Action, 6 or 7]
                        'robot_eef_pose': robot_eef_poses.cpu().numpy()  # [Action, 6 or 7]
                    }
                    replay_buffer.add_episode(episode)
                    episode_cnt += 1

                    # top view
                    tv_video_path = os.path.join(video_dir, '3.mp4')
                    tv_video_writer = cv2.VideoWriter(tv_video_path, cv2.VideoWriter_fourcc(*'mp4v'), args_cli.fps, (image_w, image_h))
                    # eye in hand view
                    eih_video_path = os.path.join(video_dir, '1.mp4')
                    eih_video_writer = cv2.VideoWriter(eih_video_path, cv2.VideoWriter_fourcc(*'mp4v'), args_cli.fps, (image_w, image_h))

                    # top view video writer
                    # warning isaac lab saved image in BGR format !!
                    for image, eef_pose_w, action_w in zip(tv_images_list, robot_eef_pose_list, action_list):
                        image = image.cpu().numpy()[0]

                        if args_cli.debug:
                            # -- project ee frame pose to 2d image (for debug)
                            ee_matrix_w = matrix_from_quat(eef_pose_w[:, 3:7]).squeeze() # ee pose quat in world frame
                            ee_pos_w = eef_pose_w[:, :3].squeeze() # ee pose position in world frame
                            _, ee_pos_in_cam = get_object_pose_in_camera_frame(tv_cam_matrix_inv_w,
                                                                               tv_cam_pos_inv_w,
                                                                               ee_matrix_w,
                                                                               ee_pos_w)
                            ee_pos_in_cam_2d = project_points(ee_pos_in_cam.unsqueeze(0), tv_cam_intrinsic)
                            ee_pos_in_cam_2d = ee_pos_in_cam_2d.squeeze().cpu().numpy()
                            ee_pos = np.round(ee_pos_in_cam_2d[:2]).astype(int)
                            cv2.circle(image, (image_w - ee_pos[0], ee_pos[1]), 1, (0, 0, 255), -1)  # red, in BGR format

                            # -- project abs action pose to 2d image (for debug)
                            action_matrix_w = matrix_from_quat(action_w[:, 3:7]).squeeze()  # ee pose quat in world frame
                            action_pos_w = action_w[:, :3].squeeze()  # ee pose position in world frame
                            _, action_pos_in_cam = get_object_pose_in_camera_frame(tv_cam_matrix_inv_w,
                                                                                   tv_cam_pos_inv_w,
                                                                                   action_matrix_w,
                                                                                   action_pos_w)
                            action_pos_in_cam_2d = project_points(action_pos_in_cam.unsqueeze(0), tv_cam_intrinsic)
                            action_pos_in_cam_2d = action_pos_in_cam_2d.squeeze().cpu().numpy()
                            action_pos = np.round(action_pos_in_cam_2d[:2]).astype(int)
                            cv2.circle(image, (image_w - action_pos[0], action_pos[1]), 1, (255, 0, 0), -1) # blue, in BGR format

                        tv_video_writer.write(image)

                    for image in eih_images_list:
                        image = image.cpu().numpy()[0]
                        eih_video_writer.write(image)

                    cv2.destroyAllWindows()
                    tv_video_writer.release()
                    eih_video_writer.release()

                is_success = False
                save_episode = False
                env.reset()
                teleop_interface.reset() # reset gripper state
                rep.utils.send_og_event("Randomize!") # new rep

                # warning reset
                tv_images_list.clear()
                eih_images_list.clear()
                action_list.clear()
                robot_eef_pose_list.clear()
                # increase_robot_eef_position = init_robot_eef_position
                print("video saved {} / {} ".format(episode_cnt, args_cli.num_demos))
                print("env reset ")

    print("finish data collection, exit simulator")
    # close the simulator
    replay_buffer.save_to_path(zarr_path=zarr_path, chunk_length=-1)
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
