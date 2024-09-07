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
parser.add_argument("--ckpt_path", type=str, default=r"D:\git_project\IsaacLab\logs\robomimic\Isaac-Lift-Cube-Franka-IK-Rel-dp_20240907_6d_frame2\latest_50.ckpt", help="Name of the task.")
parser.add_argument("--replicator", type=bool, default=False, help="enable table replicator")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--out_dim", type=int, default=6, help="low-dim or high dim task")
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

from omni.isaac.lab.devices import Se3Keyboard, Se3SpaceMouse
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg
from omni.isaac.lab.utils.math import project_points, euler_xyz_from_quat, matrix_from_quat

import omni.replicator.core as rep

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


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


def convert_quat_to_euler(poses):
    roll, pitch, yaw = euler_xyz_from_quat(poses[:, 3:])
    # quat to euler in xyz-format degree unit
    euler_rad = torch.concat([roll.unsqueeze(1), pitch.unsqueeze(1), yaw.unsqueeze(1)],
                             dim=-1)  # [Action, 3]
    euler_deg = torch.rad2deg(euler_rad)  # [Action, 3]
    poses = torch.concat([poses[:, :3], euler_deg], dim=-1)  # [Action, 6]
    return poses


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
    # log_dir = os.path.join("./logs/robomimic", args_cli.task)
    # dump the configuration into log-directory
    # dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    # dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)

    # reset environment
    obs_dict, _ = env.reset()

    # reset interfaces
    teleop_interface.reset()

    # define frank default position (xyz,wxyz,gripper)
    init_actions = torch.tensor([[0.4567, 0.0005, 0.3838, 0.0080, 0.9226, 0.0326, 0.3842, 1]],
                                  dtype=torch.float,
                                  device=env.device)
    init_actions = init_actions.repeat(env.num_envs, 1)

    print("Prepare env, plz dont move")
    for i in range(50):
        # T|R, gripper state
        obs_dict, rewards, terminated, truncated, info = env.step(init_actions)
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

    ############################ load dp policy #############################
    policy, cfg, device = load_dp(args_cli.ckpt_path)
    print("[Info] Finish load dp model ")
    #########################################################################

    # save mp4 video for diffusion policy training
    global is_success
    global save_episode
    global tv_images_list
    global eih_images_list
    is_success = False
    save_episode = False
    tv_images_list = []
    eih_images_list = []
    robot_eef_pose_list = []
    # simulate environment -- run everything in inference mode
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while True:
            # store signals before stepping
            # -- obs
            # -- camera image
            tv_cam_image = obs_dict["policy"][tv_cam_name]  # [1, H, W, C]
            eih_cam_image = obs_dict["policy"][eih_cam_name]

            # -- preprocess image, refer to diffusion_policy\real_world\real_inference_util.py (get_real_obs_dict)
            # Step1: WARNING BGR to RGB (isaac is BGR, training DP in RGB)
            tv_cam_image = tv_cam_image[..., [2, 1, 0]]
            eih_cam_image = eih_cam_image[..., [2, 1, 0]]
            # Step2: Normalize / 255
            tv_cam_image = tv_cam_image.float() / 255.0
            eih_cam_image = eih_cam_image.float() / 255.0

            tv_images_list.append(tv_cam_image)
            eih_images_list.append(eih_cam_image)

            # -- ee pose
            robot_eef_pos = obs_dict["policy"]["robot_eef_pos"]
            robot_eef_quat = obs_dict["policy"]["robot_eef_quat"]
            robot_eef_position = torch.concat([robot_eef_pos, robot_eef_quat], dim=-1)
            robot_eef_pose_list.append(robot_eef_position)

            # predict every two frames [(1, H, W, 3) ... (1, H, W, 3)]
            if len(tv_images_list) == 2:
                camera_1 = torch.concat(eih_images_list, dim=0) # [frame=2, H, W, C]
                camera_1 = camera_1.permute(0, 3, 1, 2).contiguous() # [2, C, H, W]
                camera_1 = camera_1.unsqueeze(0) # [1, 2, C, H, W]
                # print("camera_1 ", camera_1.shape)

                camera_3 = torch.concat(tv_images_list, dim=0) # [2, H, W, C]
                camera_3 = camera_3.permute(0, 3, 1, 2).contiguous()  # [2, C, H, W]
                camera_3 = camera_3.unsqueeze(0) # [1, 2, C, H, W]
                # print("camera_3 ", camera_3.shape)

                robot_eef_pose = torch.concat(robot_eef_pose_list, dim=0) # [2, 7]
                robot_eef_pose = convert_quat_to_euler(robot_eef_pose) # [2, 6] quat to euler

                # for XY (low-dim) prediction
                if args_cli.out_dim == 2:
                    robot_eef_pose = robot_eef_pose[:, :2].unsqueeze(0) # [1, 2, 2]
                # for XYZRPY (high-dim) prediction
                if args_cli.out_dim == 6:
                    robot_eef_pose = robot_eef_pose.unsqueeze(0)  # [1, 2, 6]

                # print("robot_eef_pose ", robot_eef_pose.shape)

                obs = {
                    "camera_1": camera_1,
                    "camera_3": camera_3,
                    "robot_eef_pose": robot_eef_pose,
                }

                with torch.no_grad():
                    policy.reset()
                    result = policy.predict_action(obs)

                actions = result['action']  # [1, N=15, XY-Plane=2] N frames
                # take first frame
                xy_action = actions[:, 0, :]  # take first frame

                # 因为DP源代码没有XYZ的config, 因此这里手动取前3个action
                act_dim = 3 # 2 or 3 or 6
                actions = torch.concat([xy_action[:, :act_dim], init_actions[:, act_dim:]], dim=-1)  # [1, 8] xyz,wxyz,gripper
                # print("actions ", actions.shape)

                # perform action on environment
                obs_dict, rewards, terminated, truncated, info = env.step(actions)

                tv_images_list.clear()
                eih_images_list.clear()
                robot_eef_pose_list.clear()

                # check that simulation is stopped or not
                if env.unwrapped.sim.is_stopped():
                    break

                # robomimic only cares about policy observations
                # -- next ee pose
                # next_robot_eef_position = torch.concat([obs_dict["policy"]["robot_eef_pos"],
                #                                    obs_dict["policy"]["robot_eef_quat"]], dim=-1)

            if save_episode:
                env.reset()
                save_episode = False
                teleop_interface.reset() # reset gripper state
                rep.utils.send_og_event("Randomize!") # new rep

    print("exit simulator")
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
