import os
import pathlib

import hydra
import dill
import av
import cv2
import numpy as np
import torch

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.real_world.real_inference_util import get_real_obs_dict
from diffusion_policy.real_world.video_recorder import read_video
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer

from omni.isaac.lab.utils.math import project_points, matrix_from_quat

fps = 30

def get_object_pose_in_camera_frame(R_cam_inv, t_cam_inv, R_obj, t_obj):
    """计算物体在相机坐标系下的位姿"""
    # 物体相对于相机 = 相机逆位姿 * 物体位姿
    R_obj_to_cam = torch.matmul(R_cam_inv, R_obj)
    t_obj_to_cam = torch.matmul(R_cam_inv, t_obj) + t_cam_inv
    return R_obj_to_cam, t_obj_to_cam


def project_world_frame_point_to_camera_2d_plane(eef_pose_w, intrinsic, extrinsic):
    # -- project ee frame pose to 2d image (for debug)
    ee_matrix_w = matrix_from_quat(eef_pose_w[:, 3:]).squeeze()  # ee pose quat in world frame
    ee_pos_w = eef_pose_w[:, :3].squeeze()  # ee pose position in world frame
    cam_matrix_w = extrinsic[:3, :3]
    cam_pos_w = extrinsic[:3, 3]
    _, ee_pos_in_cam = get_object_pose_in_camera_frame(cam_matrix_w,
                                                       cam_pos_w,
                                                       ee_matrix_w,
                                                       ee_pos_w)
    ee_pos_in_cam_2d = project_points(ee_pos_in_cam.unsqueeze(0), intrinsic)
    ee_pos_in_cam_2d = ee_pos_in_cam_2d.squeeze().cpu().numpy()
    ee_pos = np.round(ee_pos_in_cam_2d[:2]).astype(int)
    return ee_pos


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
    dataset_path = r'D:\git_project\IsaacLab\logs\robomimic\Isaac-Lift-Cube-Franka-IK-Rel-dp_exp2'
    ckpt_path = r"D:\git_project\IsaacLab\logs\robomimic\Isaac-Lift-Cube-Franka-IK-Rel-dp_exp2\latest.ckpt"

    input = pathlib.Path(os.path.expanduser(dataset_path))
    in_zarr_path = input.joinpath('replay_buffer.zarr')
    in_video_dir = input.joinpath('videos')
    assert in_zarr_path.is_dir(), "File {} not found ".format(in_zarr_path)
    assert in_video_dir.is_dir(), "File {} not found ".format(in_video_dir)

    in_replay_buffer = ReplayBuffer.create_from_path(in_zarr_path, mode='r')
    actions = in_replay_buffer.data['action'][:]
    robot_eef_poses = in_replay_buffer.data['robot_eef_pose'][:]
    # timestamps = in_replay_buffer['timestamp'][:]

    n_steps = in_replay_buffer.n_steps
    episode_starts = in_replay_buffer.episode_ends[:] - in_replay_buffer.episode_lengths[:]
    episode_lengths = in_replay_buffer.episode_lengths
    # dt = timestamps[1] - timestamps[0]
    dt = 1 / fps

    # load dp policy
    policy, cfg, device = load_dp(ckpt_path)

    camera_1_arr = []
    camera_3_arr = []
    eef_arr = []

    frame_id = 0
    episode_id = 0
    for episode_idx, episode_length in enumerate(episode_lengths):
        episode_video_dir = in_video_dir.joinpath(str(episode_idx))
        episode_start = episode_starts[episode_idx]
        episode_video_paths = sorted(episode_video_dir.glob('*.mp4'), key=lambda x: int(x.stem))
        view_1 = str(episode_video_paths[0])
        view_3 = str(episode_video_paths[1])

        # read resolution
        with av.open(str(episode_video_paths[0].absolute())) as container:
            video = container.streams.video[0]
            vcc = video.codec_context
            this_res = (vcc.width, vcc.height)
        in_img_res = this_res
        image_w, image_h = in_img_res

        print("[Info] episode_idx : ", episode_idx)
        print("[Info] FPS : ", fps)
        print("[Info] video input image size ", in_img_res)
        print("[Info] video output image size ", in_img_res)
        tv_intrinsic = torch.tensor([[366.4996, 0.0000, 160.0000],
                                     [0.0000, 366.4996, 120.0000],
                                     [0.0000, 0.0000, 1.0000]]).to(device)
        tv_extrinsic = torch.tensor([[-8.2254e-06, 1.0000e+00, 2.2039e-06, 6.8441e-06],
                                     [-8.6603e-01, -8.2254e-06, 5.0000e-01, 4.5263e-01],
                                     [5.0000e-01, 2.1687e-06, 8.6603e-01, -1.4160e+00],
                                     [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]]).to(device)


        image_tf = get_image_transform(input_res=in_img_res, output_res=in_img_res)
        for step_idx, (camera_1_image, camera_3_image) in enumerate(
                zip(read_video(video_path=view_1, dt=dt, thread_type='FRAME', img_transform=image_tf),
                    read_video(video_path=view_3, dt=dt, thread_type='FRAME', img_transform=image_tf))):
            gt_action = actions[frame_id]
            robot_eef_pose = robot_eef_poses[frame_id]

            camera_1_arr.append(camera_1_image)
            camera_3_arr.append(camera_3_image)
            eef_arr.append(robot_eef_pose)

            if step_idx == (episode_length - 1):
                break

            # dp predict every 2 frames, which means take two images at a time.
            if len(camera_1_arr) == 2 and len(camera_3_arr) == 2 and len(eef_arr) == 2:
                obs = {
                    "camera_1": np.array(camera_1_arr),
                    "camera_3": np.array(camera_3_arr),
                    "robot_eef_pose": np.array(eef_arr),
                }

                with torch.no_grad():
                    policy.reset()
                    print("obs camera_3 ", obs["camera_3"])
                    obs_dict_np = get_real_obs_dict(
                        env_obs=obs, shape_meta=cfg.task.shape_meta)
                    print("obs_dict_np camera_3 ", obs_dict_np["camera_3"])
                    obs_dict = dict_apply(obs_dict_np,
                                          lambda x: torch.from_numpy(x).unsqueeze(0).to(device))

                    # print("obs_dict camera_1 ", obs_dict["camera_1"].shape)
                    print("obs_dict camera_3 ", obs_dict["camera_3"])
                    # print("obs_dict robot_eef_pose ", obs_dict["robot_eef_pose"].shape)

                    result = policy.predict_action(obs_dict)
                    action = result['action'][0]  # N frames
                    action = action.detach().to('cpu').numpy()
                    assert action.shape[-1] == 2
                    first_action = action[0] # dp has 15 frames, use the first to predict.

                    camera_1_arr = []
                    camera_3_arr = []
                    eef_arr = []

                    # ground truth ee pose in torch format
                    robot_eef_pose = torch.from_numpy(robot_eef_pose).unsqueeze(0).to(device)
                    ee_pos = project_world_frame_point_to_camera_2d_plane(robot_eef_pose,
                                                                          intrinsic=tv_intrinsic,
                                                                          extrinsic=tv_extrinsic)
                    # print("ee_pos ", ee_pos)

                    # predict action pose in torch format
                    first_action = torch.from_numpy(first_action).unsqueeze(0).to(device)
                    num_action = first_action.shape[1]
                    # use gt as the rest (for xy or xyz task)
                    first_action = torch.concat([first_action, robot_eef_pose[:, num_action:]], dim=-1) # for xy task
                    pred_action_pos = project_world_frame_point_to_camera_2d_plane(first_action,
                                                                                   intrinsic=tv_intrinsic,
                                                                                   extrinsic=tv_extrinsic)

                    cv2.circle(camera_3_image, (image_w - ee_pos[0], ee_pos[1]), 1, (0, 0, 255), -1)  # red, in BGR format
                    cv2.circle(camera_3_image, (image_w - pred_action_pos[0], pred_action_pos[1]), 1, (255, 0, 0), -1)  # blue, in BGR format
                    cv2.imshow('save', camera_3_image)
                    cv2.waitKey(100)

            frame_id += 1
        episode_id += 1


if __name__ == "__main__":
    # run the main function
    main()
