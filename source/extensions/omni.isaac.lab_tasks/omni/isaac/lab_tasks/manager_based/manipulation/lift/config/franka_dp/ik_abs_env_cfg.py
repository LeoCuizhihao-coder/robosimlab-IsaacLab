# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.manager_based.manipulation.lift.lift_env_with_camera_cfg import LEFT_ROBOT_OFFSET
from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from omni.isaac.lab_assets.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class FrankaCubeLiftEnvCfg(joint_pos_env_cfg.FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_hand"].friction=20.0
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        # Set Franka as robot as LEFT arm !
        self.scene.robot_left = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_Left",
                                                                 init_state=ArticulationCfg.InitialStateCfg(
                                                                     joint_pos={
                                                                         "panda_joint1": 0.0,
                                                                         "panda_joint2": -0.569,
                                                                         "panda_joint3": 0.0,
                                                                         "panda_joint4": -2.810,
                                                                         "panda_joint5": 0.0,
                                                                         "panda_joint6": 3.037,
                                                                         "panda_joint7": 0.741,
                                                                         "panda_finger_joint.*": 0.04,
                                                                     },
                                                                     pos=(0.0, LEFT_ROBOT_OFFSET, 0.0),
                                                                     )
                                                                     )

        # # warning!! careful with the body offset
        self.actions.left_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot_left",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )


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
