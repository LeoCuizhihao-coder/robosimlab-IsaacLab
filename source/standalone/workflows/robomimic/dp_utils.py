import torch


def pre_process_actions(delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # resolve gripper command
    gripper_vel = torch.zeros((delta_pose.shape[0], 1), dtype=torch.float, device=delta_pose.device)
    gripper_vel[:] = -1 if gripper_command else 1
    # compute actions
    return torch.concat([delta_pose, gripper_vel], dim=1)


def linear_interpolation(A, B, num_points=None, max_step=None):
    """
    计算两点A和B之间的多个线性插值点 (使用PyTorch)。

    参数:
    A, B - 3D坐标 (x, y, z) 形式的元组或列表
    num_points - 生成的插值点数量

    返回:
    points - 多个3D 插值点 (PyTorch张量)
    """
    assert A.shape == (3,) and B.shape == (3,)
    device = A.device
    if num_points:
        t_values = torch.linspace(0, 1, steps=num_points)
        points = [(1 - t) * A + t * B for t in t_values]
        points = torch.stack(points)
    elif max_step:
        # 计算两点之间的距离
        distance = torch.norm(B - A)
        # 计算所需步数(暂时不要+1)
        num_steps = int(distance / max_step)
        # 生成插值点
        x_vals = torch.linspace(A[0], B[0], num_steps)
        y_vals = torch.linspace(A[1], B[1], num_steps)
        z_vals = torch.linspace(A[2], B[2], num_steps)
        points = torch.stack((x_vals, y_vals, z_vals), dim=1)
        points = points.to(device)
    if points.shape[0] == 0:
        return B.unsqueeze(0) # (1, 3)
    return points # (N, 3)


def interpolate_rpy(RPY1, RPY2, step_degree=1):
    device = RPY1.device

    # 计算每个角度的差值
    diff_r = angle_diff(RPY1[0], RPY2[0])
    diff_p = angle_diff(RPY1[1], RPY2[1])
    diff_y = angle_diff(RPY1[2], RPY2[2])

    # 找到最大的旋转差来决定步数
    max_diff = max(abs(diff_r), abs(diff_p), abs(diff_y))
    num_steps = int(max_diff / step_degree) # + 1  # (暂时不要+1)

    # 生成插值的角度
    r_vals = torch.linspace(RPY1[0], RPY1[0] + diff_r, num_steps)
    p_vals = torch.linspace(RPY1[1], RPY1[1] + diff_p, num_steps)
    y_vals = torch.linspace(RPY1[2], RPY1[2] + diff_y, num_steps)
    rpy_points = torch.stack((r_vals, p_vals, y_vals), dim=1)
    # print("rpy_points ", rpy_points)

    # 确保每个角度在 [-180, 180] 范围内
    # rpy_points = (rpy_points + 180) % 360 - 180

    rpy_points = torch.round(rpy_points).to(device)
    if rpy_points.shape[0] == 0:
        return RPY2.unsqueeze(0) # (1, 3)

    return rpy_points


# 定义三次多项式的系数计算函数
def cubic_coefficients(P0, P1, V0=0, V1=0):
    """
    计算三次多项式的系数
    P0: 起点坐标 (X1, Y1, Z1)
    P1: 终点坐标 (X2, Y2, Z2)
    V0: 起点速度（默认为0）
    V1: 终点速度（默认为0）
    """
    # P(t) = a0 + a1 * t + a2 * t^2 + a3 * t^3
    a0 = P0
    a1 = V0
    a2 = 3 * (P1 - P0) - 2 * V0 - V1
    a3 = -2 * (P1 - P0) + V0 + V1
    return a0, a1, a2, a3


# 定义多项式插值函数
def cubic_interpolation(A, B, num_points=None, max_step=None):
    """
    在A和B之间做三次多项式插值
    A: 起点 (X1, Y1, Z1)
    B: 终点 (X2, Y2, Z2)
    steps: 插值步数，默认为100
    """
    device = A.device

    # 拆解出各个坐标分量
    X1, Y1, Z1 = A
    X2, Y2, Z2 = B

    if max_step:
        # 计算两点之间的欧几里得距离
        total_distance = torch.sqrt((X2 - X1)**2 + (Y2 - Y1)**2 + (Z2 - Z1)**2)
        # 计算所需的插值点数量，至少2个点
        steps = max(int(torch.ceil(total_distance / max_step)), 2)
    elif num_points:
        steps = num_points
    else:
        raise TypeError

    # 分别计算每个坐标方向的多项式系数
    a0_x, a1_x, a2_x, a3_x = cubic_coefficients(X1, X2)
    a0_y, a1_y, a2_y, a3_y = cubic_coefficients(Y1, Y2)
    a0_z, a1_z, a2_z, a3_z = cubic_coefficients(Z1, Z2)

    # 时间参数 t 从 0 到 1 均匀分布
    t_values = torch.linspace(0, 1, steps).to(device)

    # 计算每个坐标方向的插值值
    X_t = a0_x + a1_x * t_values + a2_x * t_values ** 2 + a3_x * t_values ** 3
    Y_t = a0_y + a1_y * t_values + a2_y * t_values ** 2 + a3_y * t_values ** 3
    Z_t = a0_z + a1_z * t_values + a2_z * t_values ** 2 + a3_z * t_values ** 3

    # 合并成轨迹点
    trajectory = torch.stack([X_t, Y_t, Z_t], dim=-1)
    trajectory = trajectory.to(device)
    # print("trajectory ", trajectory.shape)

    if trajectory.shape[0] == 0:
        return B.unsqueeze(0) # (1, 3)

    return trajectory


def angle_diff(a, b):
    """ 计算角度差，确保在 [-180, 180] 范围内 """
    diff = b - a
    return (diff + 180) % 360 - 180  # 确保结果在 [-180, 180] 之间


def euler_xyz_from_quat(quaternion: torch.Tensor) -> torch.Tensor:
    """
    将四元数转换为欧拉角（单位：角度）
    输入四元数格式为 (w, x, y, z)
    """
    w, x, y, z = quaternion.unbind(-1)

    # 计算roll (x轴的旋转)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)

    # 计算pitch (y轴的旋转)
    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)  # 防止出现数值误差导致的异常值
    pitch_y = torch.asin(t2)

    # 计算yaw (z轴的旋转)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)

    # 将弧度转换为角度
    euler_angles = torch.stack([roll_x, pitch_y, yaw_z], dim=-1)
    euler_angles_deg = torch.rad2deg(euler_angles)  # 转换为角度

    return euler_angles_deg


def quat_from_euler_xyz(euler_angles: torch.Tensor) -> torch.Tensor:
    """
    将欧拉角（角度）转换为四元数 (w, x, y, z)
    输入: euler_angles (torch.Tensor) - 欧拉角张量，格式为 [roll, pitch, yaw]，单位为角度
    输出: 四元数张量，格式为 [w, x, y, z]
    """
    # 将角度转换为弧度
    euler_angles_rad = torch.deg2rad(euler_angles)

    roll, pitch, yaw = euler_angles_rad.unbind(-1)

    # 计算四元数的各分量
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    # 返回四元数 (w, x, y, z)
    return torch.stack([w, x, y, z], dim=-1)


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