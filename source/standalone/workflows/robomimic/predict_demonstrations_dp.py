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
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Abs-dp", help="Name of the task.")
parser.add_argument("--ckpt_path", type=str, default=r"D:\git_project\IsaacLab\logs\diffusion_policy\traj15\latest.ckpt", help="Name of the task.")
parser.add_argument("--replicator", type=bool, default=False, help="enable table replicator")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
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
import torch
import hydra
import dill

from omni.isaac.lab.devices import Se3Keyboard, Se3SpaceMouse, Se3Keyboard_Dual
# from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm

import omni.isaac.lab_tasks  # noqa: F401
# from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg
from omni.isaac.lab.utils.math import matrix_from_quat, compute_pose_error, subtract_frame_transforms

import omni.replicator.core as rep

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.real_world.real_inference_util import get_real_obs_dict

from dp_utils import (quat_from_euler_xyz,
                      euler_xyz_from_quat,
                      inverse_transformation,
                      pre_process_actions)


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


def load_dp(ckpt_path):
    # load checkpoint
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # hacks for method-specific setup.
    if 'diffusion' in cfg.name:
        # diffusion model
        policy: BaseImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device('cuda')
        policy.eval().to(device)

        # set inference params
        policy.num_inference_steps = 16  # DDIM inference iterations
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

        return policy, cfg, device


def main():
    """Collect demonstrations from the environment using teleop interfaces."""
    assert (args_cli.task == "Isaac-Lift-Cube-Franka-IK-Abs-dp"), \
        "Only 'Isaac-Lift-Cube-Franka-IK-Abs-dp' is supported currently."

    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    # modify configuration such that the environment runs indefinitely
    # set the resampling time range to large number to avoid resampling
    env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
    # we want to have the terms in the observations returned as a dictionary
    # rather than a concatenated tensor
    env_cfg.observations.policy.concatenate_terms = False

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
        for _ in range(100):
            obs_dict = get_obs_dict(init_actions)
        print("[Info] You can move now")
        return obs_dict

    # add teleoperation key for env reset
    teleop_interface.add_callback("L", episode_fail)
    teleop_interface.add_callback("J", episode_success)

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


    ############################ load dp policy #############################
    policy, cfg, device = load_dp(args_cli.ckpt_path)
    print("[Info] Finish load dp model ")
    #########################################################################

    global is_success, save_episode

    tv_images_list, eih_images_list, le_images_list = [], [], []
    is_success, save_episode = False, False
    robot_eef_list = []

    # simulate environment -- run everything in inference mode
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while True:
            # get keyboard command
            delta_pose, gripper_command = teleop_interface.advance()
            # convert to torch
            delta_pose = torch.tensor(delta_pose, dtype=torch.float, device=env.device).repeat(env.num_envs, 1)
            # compute actions based on environment
            human_actions = pre_process_actions(delta_pose, gripper_command)

            ################### convert isaac BGR to RGB ###################
            # tv_cam_image = cv2.cvtColor(tv_cam_image, cv2.COLOR_BGR2RGB)
            # eih_cam_image = cv2.cvtColor(eih_cam_image, cv2.COLOR_BGR2RGB)
            # cv2.imshow('save', tv_cam_image)
            # cv2.waitKey(1)

            ################################################################
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

            tv_images_list.append(tv_image)
            eih_images_list.append(eih_image)
            le_images_list.append(le_image)
            robot_eef_list.append(right_robot_eef)

            if len(tv_images_list) == 2 and len(eih_images_list) == 2 and len(robot_eef_list) == 2:
                tv_images_arr = torch.concat(tv_images_list, dim=0).cpu().numpy()
                eih_images_arr = torch.concat(eih_images_list, dim=0).cpu().numpy()
                robot_eef_arr = torch.concat(robot_eef_list, dim=0).cpu().numpy()

                obs = {
                    "camera_1": eih_images_arr,  # [2, H, W, 3]
                    "camera_3": tv_images_arr,  # [2, H, W, 3]
                    "robot_eef_pose": robot_eef_arr[:, :6],  # [2, 6]
                    'gripper_position': robot_eef_arr[:, 6:7],  # [2, 1]
                }

                with torch.no_grad():
                    policy.reset()
                    obs_dict_np = get_real_obs_dict(
                        env_obs=obs, shape_meta=cfg.task.shape_meta)
                    obs_dict = dict_apply(obs_dict_np,
                                          lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                    result = policy.predict_action(obs_dict)
                    # Receding Horizon Control
                    actions_rhc = result['action']  # [1, N=15, 7] N frames
                    assert actions_rhc.shape[-1] == 7
                    # print("actions_ddpm ", actions_ddpm[:, :, :3])

                    # must clear !
                    tv_images_list.clear()
                    eih_images_list.clear()
                    robot_eef_list.clear()

                    num_action = 2
                    for idx in range(num_action):
                        actions = actions_rhc[:, idx, :]
                        abs_action_pos = actions[:, :3]
                        abs_action_rpy = actions[:, 3:6]
                        action_gripper = actions[:, 6:7]  # gripper predict by policy
                        action_gripper = torch.where(action_gripper > 0, torch.tensor(1.0), torch.tensor(-1.0))
                        # action_gripper = human_actions[:, 6:7]  # [1, 1]
                        right_abs_actions = torch.concat([abs_action_pos,  # XYZ
                                                          quat_from_euler_xyz(abs_action_rpy),  # WXYZ
                                                          action_gripper],
                                                          dim=-1)  # [1, 8] XYZ, WXYZ, gripper
                        actions = torch.concat([right_abs_actions, init_left_actions], dim=-1)
                        obs_dict, converged, steps = run_env_until_converge(get_obs_dict,
                                                                            actions,
                                                                            pos_error_thresh=0.02,
                                                                            axis_angle_error_thresh=0.5,
                                                                            max_attempts=3,
                                                                            verbose=True)

                # check that simulation is stopped or not
                if env.unwrapped.sim.is_stopped():
                    break

            if save_episode:
                rep.utils.send_og_event("Randomize!") # new rep

                # must clear !
                is_success, save_episode = False, False
                tv_images_list, eih_images_list, le_images_list = [], [], []
                robot_eef_list = []

                obs_dict = prepare_environment(init_actions)
                # right_abs_action_pos = init_right_pos.clone()
                # right_abs_action_rpy = init_right_rpy.clone()
                # left_abs_action_pos = init_left_pos.clone()
                # left_abs_action_rpy = init_left_rpy.clone()

    print("exit simulator")
    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
