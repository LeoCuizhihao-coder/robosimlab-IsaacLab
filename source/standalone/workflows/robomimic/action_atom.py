import random

import torch

from omni.isaac.lab.utils.math import matrix_from_quat, quat_from_matrix

from action_handlers import ActionIterator
from atom.atom_meta import Atom
from dp_utils import euler_xyz_from_quat, quat_from_euler_xyz, cubic_interpolation, interpolate_rpy


def check_xyz_rpy_shape(vec, expected_len=3):
    assert len(vec) == expected_len, f"Shape must be {expected_len}, got {len(vec)}"


def rotation_matrix_180(axis, device):
    axis = axis.lower()
    if axis == 'x':
        return torch.tensor([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ], dtype=torch.float, device=device)
    elif axis == 'y':
        return torch.tensor([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ], dtype=torch.float, device=device)
    elif axis == 'z':
        return torch.tensor([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ], dtype=torch.float, device=device)
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")


def parse_pick_points_in_world(t_object_world, r_object_world, target_object):
    """
    解析抓取点，将相对物体的抓取点偏移和旋转转换为世界坐标系下的位姿
    :param t_object_world: 'translation': (3),
    :param r_object_world: 'quat':  (4),
    :param target_object:  'category', 'tcp_offset', 'pick_points'
    :return:
    """

    device = t_object_world.device
    r_object_world = matrix_from_quat(r_object_world)  # geometry matrix

    world_pick_points = []
    if 'pick_points' in target_object:
        pick_points = target_object["pick_points"]  # relative to target pose
        for pick_point in pick_points:
            # return data structure
            pick_point_data = {
                "name": pick_point["name"],
                "offset": {"pos": None, "quat": None},
                "score": pick_point.get("score", 0),
                "waypoints": []
            }

            if 'offset' in pick_point:
                t_offset_object = pick_point['offset']['pos']
                r_offset_object = pick_point['offset']['quat']
            else:
                t_offset_object = [0, 0, 0]
                r_offset_object = [1, 0, 0, 0]
                print("[Info] Did not find [pick point] offset")

            t_offset_object = torch.tensor(t_offset_object, dtype=torch.float, device=device)
            r_offset_object = torch.tensor(r_offset_object, dtype=torch.float, device=device)
            r_offset_object = matrix_from_quat(r_offset_object)

            # target offset to world frame
            t_offset_world = torch.matmul(r_object_world, t_offset_object).squeeze() + t_object_world
            r_offset_world = r_object_world @ r_offset_object
            # print("pick point name ", pick_point['name'])

            pick_point_data["offset"]["pos"] = t_offset_world
            pick_point_data["offset"]["quat"] = quat_from_matrix(r_offset_world)

            # parse waypoints, default sort in order
            if 'waypoints' in pick_point:
                target_waypoints = pick_point['waypoints']
                for waypoint in target_waypoints:
                    # return data structure
                    waypoint_data = {
                        "name": waypoint["name"],
                        "offset": {"pos": None, "rot": None},
                        "score": waypoint.get("score", 0)
                    }

                    if 'offset' in waypoint:
                        t_waypoint_offset = waypoint['offset']['pos']
                        r_waypoint_offset = waypoint['offset']['rot']
                    else:
                        t_waypoint_offset = [0, 0, 0]
                        r_waypoint_offset = [0, 0, 0]
                        print("[Info] Did not find [waypoint] offset")

                    t_waypoint_offset = torch.tensor(t_waypoint_offset, dtype=torch.float, device=device)
                    r_waypoint_offset = torch.tensor(r_waypoint_offset, dtype=torch.float, device=device)
                    r_waypoint_offset = matrix_from_quat(quat_from_euler_xyz(r_waypoint_offset))

                    # 计算 waypoint 的位置和旋转
                    t_waypoint_world = torch.matmul(r_offset_world, t_waypoint_offset).squeeze() + t_offset_world
                    r_waypoint_world = r_offset_world @ r_waypoint_offset

                    waypoint_data["offset"]["pos"] = t_waypoint_world
                    waypoint_data["offset"]["rot"] = quat_from_matrix(r_waypoint_world)

                    pick_point_data["waypoints"].append(waypoint_data)

            world_pick_points.append(pick_point_data)

    return world_pick_points


class GripperOpen(Atom):
    def __init__(self):
        super().__init__()

    def execute(self, scene_obs=None):
        robot_id = self.robot_info["robot_id"]
        ee_key = self.robot_ee_pose[robot_id]
        ee_pose = scene_obs[ee_key]
        status = torch.tensor([1], dtype=torch.float, device=ee_pose.device)
        return ActionIterator([status], self.__class__.__name__)


class GripperClose(Atom):
    def __init__(self):
        super().__init__()

    def execute(self, scene_obs=None):
        robot_id = self.robot_info["robot_id"]
        ee_key = self.robot_ee_pose[robot_id]
        ee_pose = scene_obs[ee_key]
        status = torch.tensor([-1], dtype=torch.float, device=ee_pose.device)
        return ActionIterator([status], self.__class__.__name__)


class IsGripperClosed(Atom):
    def __init__(self):
        super().__init__()

    def execute(self, scene_obs=None):
        robot_id = self.robot_info["robot_id"]
        ee_key = self.robot_ee_width[robot_id]
        ee_width = scene_obs[ee_key]
        return ActionIterator([ee_width], self.__class__.__name__)


class MoveTo(Atom):
    def __init__(self, t_step=0.004, r_step=1, docs=""):
        super().__init__()
        self.trans_step = t_step # translate
        self.rot_step = r_step # rotate
        self.docs = docs
        self.pick_pose_name = None

    def get_trans_traj(self, ee_pos, to_pos):
        trajectory = cubic_interpolation(ee_pos, to_pos, max_step=self.trans_step)
        return trajectory

    def get_rot_traj(self, ee_rpy, to_rpy):
        trajectory = interpolate_rpy(ee_rpy, to_rpy, step_degree=self.rot_step)
        return trajectory

    @staticmethod
    def get_obs_pose(scene_obs, target_key):
        _pos = scene_obs[target_key][:, :3].squeeze()  # [3,]
        _quat = scene_obs[target_key][:, 3:7].squeeze()  # [4,]
        return _pos, _quat

    def get_ee_pose(self, scene_obs):
        robot_id = self.robot_info["robot_id"]
        ee_key = self.robot_ee_pose[robot_id]
        ee_pos, ee_quat = self.get_obs_pose(scene_obs, ee_key) # get ee pose
        device = ee_pos.device
        return ee_pos, ee_quat, device

    def get_pick_poses(self, scene_obs=None):
        assert isinstance(self.target_object, dict)
        target_key = self.target_object["category"] + "_pose"
        target_pos, target_quat = self.get_obs_pose(scene_obs, target_key)
        target_pick_poses_w = parse_pick_points_in_world(target_pos, target_quat, self.target_object) # get object pose
        return target_pick_poses_w

    def calc_trans_waypoints(self, ee_pos, target_pick_point):
        trajectory = []
        pre_waypoint = ee_pos
        pick_point_pos = target_pick_point['offset']['pos']
        for waypoint in target_pick_point['waypoints']:
            waypoint_pos =  waypoint['offset']['pos']
            trajectory.append(self.get_trans_traj(pre_waypoint, waypoint_pos))
            pre_waypoint = waypoint_pos.clone()
        wp_2_pp = self.get_trans_traj(pre_waypoint, pick_point_pos)
        trajectory.append(wp_2_pp)
        return trajectory

    def calc_rot_waypoints(self, ee_rot, target_pick_point):
        trajectory = []
        pre_waypoint = ee_rot
        pick_point_rot = target_pick_point['offset']['quat']
        pick_point_rpy = euler_xyz_from_quat(pick_point_rot)
        # print(">>>> pick_point_rpy ", pick_point_rpy)
        for waypoint in target_pick_point['waypoints']:
            waypoint_quat = waypoint['offset']['rot']
            waypoint_rpy = euler_xyz_from_quat(waypoint_quat)  # [-180, 180]
            # print(">>>> waypoint_rpy ", waypoint_rpy)
            waypoint_rpy = self.match_obj_pose_and_ee(waypoint_rpy, ee_rot)
            waypoint_rpy = (waypoint_rpy + 180) % 360 - 180
            trajectory.append(self.get_rot_traj(pre_waypoint, waypoint_rpy))
            pre_waypoint = waypoint_rpy.clone()
        pick_point_rpy = self.match_obj_pose_and_ee(pick_point_rpy, ee_rot)
        pick_point_rpy = (pick_point_rpy + 180) % 360 - 180
        wp_2_pp = self.get_rot_traj(pre_waypoint, pick_point_rpy)
        trajectory.append(wp_2_pp)
        return trajectory

    def sort_pick_pose(self, target_pick_poses, ee_pos, ee_quat):
        ################### choose the best pose, sort the pick-point pose #########################
        weight1 = 0.6
        weight2 = 0.4
        def normalize(tensor):
            max_val = torch.max(tensor)
            min_val = torch.min(tensor)
            return (tensor - min_val) / (max_val - min_val)
        """
        sorting strategy, 1. filter z-axis. 2. sort by z height and x-axis matching (weighted sum)
        """
        target_pick_poses = self.filter_invalid_z_axis_pose(target_pick_poses, threshold=0.05)
        if len(target_pick_poses) > 1:
            height = self.get_objects_height(target_pick_poses)
            x_axis_matching = self.get_objects_x_axis_matching(target_pick_poses, ee_pos, ee_quat)
            height = normalize(height)
            x_axis_matching = normalize(x_axis_matching)
            weighted_sum = height * weight1 + x_axis_matching * weight2
            normalized_weighted_sum = normalize(weighted_sum)
            sorted_indices = torch.argsort(normalized_weighted_sum, descending=True)
            target_pick_poses = [target_pick_poses[i] for i in sorted_indices][0]  # pick the first sort
            # target_pick_poses = random.choice(target_pick_poses)
            return target_pick_poses
        elif len(target_pick_poses) == 1:
            return target_pick_poses[0]
        else:
            return target_pick_poses


    def execute(self, scene_obs=None):
        ee_pos, ee_quat, device = self.get_ee_pose(scene_obs)
        ee_rpy = euler_xyz_from_quat(ee_quat)  # [-180, 180]
        self.pick_poses = self.get_pick_poses(scene_obs)
        pick_poses = self.sort_pick_pose(self.pick_poses, ee_pos, ee_quat)
        if len(pick_poses) == 0:
            self.set_auto_next(False)
            print("[Warning] No pickable pose, human takeover")
            return HumanControl().execute()
        else:
            self.set_auto_next(True)
            self.pick_pose_name = pick_poses["name"]
            waypoint_traj_trans = self.calc_trans_waypoints(ee_pos, pick_poses)
            waypoint_traj_rot = self.calc_rot_waypoints(ee_rpy, pick_poses)
            # key check in collection_demo.py (bad design the last dim must equal to 3)
            full_trajectory, full_act_type = [], []
            for _t, _r in zip(waypoint_traj_trans, waypoint_traj_rot):
                k1 = ['Translate' for _ in range(len(_t))]
                k2 = ['Rotate' for _ in range(len(_r))]
                # translate then rotate
                trajectory = torch.concat([_t, _r], dim=0)
                act_type = k1 + k2
                # rotate then translate
                # trajectory = torch.concat([_r, _t], dim=0)
                # act_type = k2 + k1
                full_trajectory.extend(trajectory)
                full_act_type.extend(act_type)
            assert len(full_trajectory) == len(full_act_type)
            return ActionIterator(full_trajectory, full_act_type)

    @staticmethod
    def match_obj_pose_and_ee(target_rpy, ee_rpy):
        # 对物体Z轴进行操作，这个需要根据ee的位姿来决定，当前ee的z向下，所以遇到物体z向上的需要按照X轴旋转180进行尝试
        # 同理，ee的x-axis默认是forward，所以当物体的x-axis指向后，理论上是一个不好的抓取点，最好是再将Z转180，让X向前
        device = ee_rpy.device
        target_matrix = matrix_from_quat(quat_from_euler_xyz(target_rpy))
        ee_matrix = matrix_from_quat(quat_from_euler_xyz(ee_rpy))
        target_z_axis = target_matrix[2, :3]  # (3, )
        # target_x_axis = target_matrix[0, :3]  # (3, )
        ee_z_axis = ee_matrix[2, :3]  # (3, )
        # ee_x_axis = ee_matrix[0, :3]  # (3, )
        is_z_align = torch.matmul(target_z_axis, ee_z_axis).squeeze()
        # is_x_align = torch.matmul(target_x_axis, ee_x_axis).squeeze()
        # 先对X进行旋转(z-downward)
        if is_z_align < 0:
            target_matrix = target_matrix @ rotation_matrix_180("X", device=device)
        # # 再对Z进行旋转(x-forward)
        # if is_x_align < 0:
        #     target_matrix = target_matrix @ rotation_matrix_180("Z", device=device)
        target_rpy = euler_xyz_from_quat(quat_from_matrix(target_matrix))
        return target_rpy

    @staticmethod
    def filter_invalid_z_axis_pose(objects, threshold=0.8):
        valid_pick_point = []
        # print("[Info] All pick point ", len(objects))
        for obj in objects:
            name = obj['name']
            object_matrix_w = matrix_from_quat(obj['offset']['quat'])
            z_axis = object_matrix_w[:, 2]
            z_value = z_axis[2]
            if z_value > threshold:
                # print("valid pick point ", name)
                valid_pick_point.append(obj)
        print("[Info] Valid pick point ", len(valid_pick_point))
        return valid_pick_point

    @staticmethod
    def get_objects_height(objects):
        height_list = []
        for obj in objects:
            height = obj['offset']['pos'][2]
            height_list.append(height)
        return torch.stack(height_list)

    @staticmethod
    def get_objects_x_axis_matching(objects, ee_pos, ee_quat):
        x_axis_list = []
        for obj in objects:
            obj_mat = matrix_from_quat(obj['offset']['quat'])
            ee_mat = matrix_from_quat(ee_quat)
            obj_x_axis = obj_mat[:, 0]
            ee_x_axis = ee_mat[:, 0]
            x_match_rate = torch.matmul(obj_x_axis, ee_x_axis).squeeze()
            x_axis_list.append(x_match_rate)
        return torch.stack(x_axis_list)


class TranslateToObject(MoveTo):
    def __init__(self, step=0.004, docs=""):
        super().__init__(t_step=step, docs="")
        self.docs = docs

    def execute(self, scene_obs=None):
        ee_pos, ee_quat, device = super().get_ee_pose(scene_obs)
        self.pick_poses = super().get_pick_poses(scene_obs)
        pick_poses = self.sort_pick_pose(self.pick_poses, ee_pos, ee_quat)
        trajectory = super().calc_trans_waypoints(ee_pos, pick_poses)
        return ActionIterator(trajectory, self.__class__.__name__)


class RotateToObject(MoveTo):
    def __init__(self, step=1, docs=""):
        super().__init__(r_step=step, docs=docs)
        self.docs = docs

    def execute(self, scene_obs=None):
        ee_pos, ee_quat, device = super().get_ee_pose(scene_obs)
        ee_rpy = euler_xyz_from_quat(ee_quat)  # [-180, 180]
        self.pick_poses = super().get_pick_poses(scene_obs)
        pick_poses = self.sort_pick_pose(self.pick_poses, ee_pos, ee_quat)
        trajectory = super().calc_rot_waypoints(ee_rpy, pick_poses)
        return ActionIterator(trajectory, self.__class__.__name__)


class TranslateToFixed(MoveTo):
    def __init__(self, to_pos, step=0.004, docs=""):
        super().__init__(t_step=step, docs=docs)
        self.to_pos = to_pos
        self.docs = docs

    def execute(self, scene_obs=None):
        ee_pos, ee_quat, device = super().get_ee_pose(scene_obs)
        assert isinstance(self.to_pos, str) and (self.to_pos in self.robot_info.keys()), \
                "Check if the robot config key and Atom to_pos or to_rot key are matched"
        to_pos = self.robot_info[self.to_pos]
        if isinstance(to_pos, (list, tuple, torch.Tensor)):
            if not isinstance(to_pos, torch.Tensor):
                to_pos = torch.tensor(to_pos, dtype=torch.float, device=device)
        else:
            raise TypeError(r"Only support List, Tuple, or Tensor")
        trajectory = self.get_trans_traj(ee_pos, to_pos)
        return ActionIterator(trajectory, self.__class__.__name__)


class RotateToFixed(MoveTo):
    def __init__(self, to_rot, step=1, docs=""):
        super().__init__(r_step=step, docs=docs)
        self.to_rot = to_rot
        self.docs = docs

    def execute(self, scene_obs=None):
        ee_pos, ee_quat, device = super().get_ee_pose(scene_obs)
        ee_rpy = euler_xyz_from_quat(ee_quat)  # [-180, 180]
        assert isinstance(self.to_rot, str) and (self.to_rot in self.robot_info.keys()), \
                    "Check if the robot config key and Atom to_pos or to_rot key are matched"
        to_rot = self.robot_info[self.to_rot]
        if isinstance(to_rot, (list, tuple, torch.Tensor)):
            if not isinstance(to_rot, torch.Tensor):
                to_rot = torch.tensor(to_rot, dtype=torch.float, device=device)
        else:
            raise TypeError(r"Only support List, Tuple, or Tensor")
        to_rot = (to_rot + 180) % 360 - 180  # constraint to [-180, 180]
        trajectory = self.get_rot_traj(ee_rpy, to_rot)
        return ActionIterator(trajectory, self.__class__.__name__)


class Wait(Atom):
    def __init__(self, duration=0.3):
        super().__init__()
        self.duration = duration

    def execute(self, scene_obs=None):
        return ActionIterator([i for i in range(int(self.duration * 100) + 1)], self.__class__.__name__)


class HumanControl(Atom):
    def __init__(self, docs="human takeover"):
        super().__init__()
        self.auto_next_atom = False
        self.docs = docs

    def execute(self, scene_obs=None):
        return ActionIterator([0], self.__class__.__name__)

