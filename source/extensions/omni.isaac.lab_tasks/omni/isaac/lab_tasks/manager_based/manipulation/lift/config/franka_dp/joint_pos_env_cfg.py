# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.assets import ArticulationCfg, RigidObjectCfg
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
# from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.manager_based.manipulation.lift.lift_env_with_camera_cfg import (
    LiftEnvWithCameraCfg,
    ISAAC_LOCAL_DIR,
    LEFT_ROBOT_OFFSET,
    TABLE_OFFSET)

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab_assets.franka import FRANKA_PANDA_CFG  # isort: skip


def create_rigid_object_cfg(prim_name, usd_path, pos, rot, scale=(1, 1, 1)):
    # general object Usd config
    object_usd_cfg = UsdFileCfg(
        usd_path=usd_path,
        scale=scale,
        rigid_props=RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
    )
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + str(prim_name),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
        spawn=object_usd_cfg,
    )

@configclass
class FrankaCubeLiftEnvCfg(LiftEnvWithCameraCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot as RIGHT arm
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"

        # Set Franka as robot as LEFT arm
        self.scene.robot_left = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_Left")

        # # Set actions for the specific robot type (franka)
        self.actions.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot_left", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.left_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_left",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # # Set the body name for the end effector
        self.commands.left_object_pose.body_name = "panda_hand"

        # ---------------------------------------------ee (right) ----------------------------------------------
        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.right_ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1074],
                    ),
                ),
            ],
        )
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.right_robot_left_finger_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="left_finger",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0475],
                    ),
                ),
            ],
        )

        # ---------------------------------------------ee (left) ----------------------------------------------
        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.left_ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_Left/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_Left/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1074],  # add 0.004 which is the delta of control
                    ),
                ),
            ],
        )
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.left_robot_left_finger_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_Left/panda_hand",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot_Left/panda_leftfinger",
                    name="left_finger",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0475],
                    ),
                ),
            ],
        )

        # --------------------------------------------Object-------------------------------------------
        tray_x, tray_y, tray_z = 0.48, TABLE_OFFSET, 0.165
        usd_path = f"{ISAAC_LOCAL_DIR}/tableware/tray.usd"
        self.scene.tray = create_rigid_object_cfg("Tray", usd_path=usd_path,
                                                  pos=[tray_x, tray_y, tray_z],
                                                  rot=[0.5, 0.5, 0.5, 0.5])
        usd_path = f"{ISAAC_LOCAL_DIR}/tableware/plate.usd"
        self.scene.plate = create_rigid_object_cfg("Plate", usd_path=usd_path,
                                                   pos=[tray_x, tray_y - 0.15, tray_z + 0.01],
                                                   rot=[1, 0, 0, 0])
        usd_path = f"{ISAAC_LOCAL_DIR}/tableware/spoon.usd"
        self.scene.spoon = create_rigid_object_cfg("Spoon", usd_path=usd_path,
                                                   pos=[tray_x + 0.1, tray_y - 0.065, tray_z + 0.02],
                                                   rot=[1, 0, 0, 0],
                                                   scale=(0.001, 0.001, 0.001))
        usd_path = f"{ISAAC_LOCAL_DIR}/tableware/mug.usd"
        self.scene.mug = create_rigid_object_cfg("Mug", usd_path=usd_path,
                                                 pos=[tray_x, tray_y, tray_z + 0.045],
                                                 rot=[1, 0, 0, 0],
                                                 scale=(0.8, 0.8, 0.8))
        usd_path = f"{ISAAC_LOCAL_DIR}/tableware/chopsticks.usd"
        self.scene.chopsticks = create_rigid_object_cfg("Chopsticks", usd_path=usd_path,
                                                        pos=[tray_x-0.1, tray_y-0.04, tray_z+0.022],
                                                        rot=[0, 0, 0, 1])
        usd_path = f"{ISAAC_LOCAL_DIR}/tableware/chopsticks_01.usd"
        self.scene.chopsticks_01 = create_rigid_object_cfg("Chopsticks_01", usd_path=usd_path,
                                                        pos=[tray_x-0.1, tray_y-0.045, tray_z+0.022],
                                                        rot=[0, 0, 0, 1])
        usd_path = f"{ISAAC_LOCAL_DIR}/tableware/bowl.usd"
        self.scene.bowl = create_rigid_object_cfg("Bowl", usd_path=usd_path,
                                                  pos=[tray_x, tray_y+0.15, tray_z+0.035],
                                                  rot=[1, 0, 0, 0])
        usd_path = f"{ISAAC_LOCAL_DIR}/tableware/broom.usd"
        self.scene.broom = create_rigid_object_cfg("Broom", usd_path=usd_path,
                                                  pos=[0.24, tray_y+0.15, 0.175],
                                                  rot=[-0.70711, 0, 0, 0.70711])


@configclass
class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
