# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os
import sys
from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
# from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
# from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
# from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
# from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from omni.isaac.lab.sensors.camera import Camera, CameraCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR


from . import mdp

dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
ISAAC_LOCAL_DIR = os.path.join(dirname, "assets")
LEFT_ROBOT_OFFSET = 0.52
TABLE_OFFSET = 0.28 # right arm base is 0,0,0
print("[Info] ISAAC_LOCAL_DIR : ", ISAAC_LOCAL_DIR)


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

##
# Scene definition
##

@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and an object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    robot_left: ArticulationCfg = MISSING

    # end-effector sensor: will be populated by agent env cfg
    right_ee_frame: FrameTransformerCfg = MISSING
    left_ee_frame: FrameTransformerCfg = MISSING

    #################################### Tableware #########################################
    # target : will be populated by agent env cfg
    bin = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Bin",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.48, -0.08, 0.22], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT.usd", scale=(0.7, 0.9, 0.65)),
        collision_group=-1,
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, TABLE_OFFSET, -0.6], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_LOCAL_DIR}/table/table.usd"),
        collision_group=-1,
    )
    stand_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Stand_Right",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_LOCAL_DIR}/Stand/stand_instanceable.usd"),
        collision_group=-1, # -1 means no collision in the scene
    )
    stand_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Stand_Left",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, LEFT_ROBOT_OFFSET, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_LOCAL_DIR}/Stand/stand_instanceable.usd"),
        collision_group=-1,  # -1 means no collision in the scene
    )

    # avoid collision!
    # g1_head = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/G1",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[-0.22, 0.26, 0.28], rot=[1, 0, 0, 0]),
    #     spawn=UsdFileCfg(usd_path=f"{ISAAC_LOCAL_DIR}/robot/g1_torso.usd"),
    #     collision_group = -1 # -1 means no collision in the scene
    # )

    plate: RigidObjectCfg = MISSING
    tray: RigidObjectCfg = MISSING
    spoon: RigidObjectCfg = MISSING
    mug: RigidObjectCfg = MISSING
    bowl: RigidObjectCfg = MISSING
    chopsticks: RigidObjectCfg = MISSING
    chopsticks_01: RigidObjectCfg = MISSING
    broom: RigidObjectCfg = MISSING

    #################################### Camera #########################################
    # camera (right eye)
    right_eye_camera = CameraCfg(
        prim_path="/World/envs/env_.*/Camera_Right",
        update_period=0.1,
        height=240,
        width=320,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.22, 0.20, 0.72), rot=(0.64086, 0.29884, -0.29884, -0.64086), convention="usd"),
    )

    # camera (left eye)
    left_eye_camera = CameraCfg(
        prim_path="/World/envs/env_.*/Camera_Left",
        update_period=0.1,
        height=240,
        width=320,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.22, 0.32, 0.72), rot=(0.64086, 0.29884, -0.29884, -0.64086), convention="usd"),
    )

    # camera (right in hand eye)
    eye_in_hand_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/hand_camera",
        update_period=0.1,
        height=240,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        # spawn=sim_utils.FisheyeCameraCfg(
        #         projection_type="fisheye_equidistant",
        #         focal_length=2.0,
        #         f_stop=10.0,
        #         clipping_range=(0.1, 1000.0),
        #         horizontal_aperture=10.0,
        #     ),
        # very hard to tune, this like UMI gripper
        offset=CameraCfg.OffsetCfg(pos=(0.065, 0.00, -0.08), rot=(-0.03084, -0.70643, -0.70579, -0.04317), convention="usd"),
    )

    # camera (left in hand eye)
    # eye_in_left_hand_camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/panda_hand/hand_left_camera",
    #     update_period=0.1,
    #     height=240,
    #     width=320,
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
    #     ),
    #     # spawn=sim_utils.FisheyeCameraCfg(
    #     #         projection_type="fisheye_equidistant",
    #     #         focal_length=2.0,
    #     #         f_stop=10.0,
    #     #         clipping_range=(0.1, 1000.0),
    #     #         horizontal_aperture=10.0,
    #     #     ),
    #     # offset=CameraCfg.OffsetCfg(pos=(0.00, 0.00, -0.02), rot=(0.0, 0.70711, 0.70711, 0.0), convention="usd"),
    #     # very hard to tune, this like UMI gripper
    #     offset=CameraCfg.OffsetCfg(pos=(0.065, 0.00, -0.08), rot=(-0.03084, -0.70643, -0.70579, -0.04317), convention="usd"),
    # )

    #################################### Environment #########################################
    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -0.6]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # object_pose = mdp.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     body_name=MISSING,  # will be set by agent env cfg
    #     resampling_time_range=(5.0, 5.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x = (0, 0), pos_y = (-0, 0), pos_z = (0, 0), roll = (0.0, 0.0), pitch = (0, 0), yaw = (0,0)
    #     ),
    # )

    # left_object_pose = mdp.UniformPoseCommandCfg(
    #     asset_name="robot_left",
    #     body_name=MISSING,  # will be set by agent env cfg
    #     resampling_time_range=(5.0, 5.0),
    #     debug_vis=False,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         # pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
    #         pos_x=(0, 0.7), pos_y=(-0.2, 0.5), pos_z=(0, 0.9), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
    #
    #     ),
    # )

    object_pose = mdp.NullCommandCfg(resampling_time_range=(5.0, 5.0), debug_vis=False)
    left_object_pose = mdp.NullCommandCfg(resampling_time_range=(5.0, 5.0), debug_vis=False)


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING
    left_arm_action: mdp.JointPositionActionCfg = MISSING
    left_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # camera
        right_eye_camera = ObsTerm(func=mdp.right_eye_camera)
        left_eye_camera = ObsTerm(func=mdp.left_eye_camera)
        eye_in_hand_camera = ObsTerm(func=mdp.eye_in_hand_camera)

        # robot right & left
        # joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)
        left_robot_position_w = ObsTerm(func=mdp.left_robot_base_pose_in_robot_root_frame)

        # ee right & left
        right_robot_eef_pose = ObsTerm(func=mdp.right_robot_eef_pose)
        right_robot_eef_width = ObsTerm(func=mdp.right_robot_eef_width)
        # gripper_actions = ObsTerm(func=mdp.last_action, params={"action_name": "gripper_action"})

        left_robot_eef_pose = ObsTerm(func=mdp.left_robot_eef_pose)
        left_robot_eef_width = ObsTerm(func=mdp.left_robot_eef_width)
        # left_gripper_actions = ObsTerm(func=mdp.last_action, params={"action_name": "gripper_action"})

        # object position
        plate_pose = ObsTerm(func=mdp.plate_position_in_robot_root_frame)
        tray_pose = ObsTerm(func=mdp.tray_position_in_robot_root_frame)
        spoon_pose = ObsTerm(func=mdp.spoon_position_in_robot_root_frame)
        mug_pose = ObsTerm(func=mdp.mug_position_in_robot_root_frame)
        bowl_pose = ObsTerm(func=mdp.bowl_position_in_robot_root_frame)
        chopsticks_pose = ObsTerm(func=mdp.chopsticks_position_in_robot_root_frame)
        broom_pose = ObsTerm(func=mdp.broom_position_in_robot_root_frame)
        # bin_pose = ObsTerm(func=mdp.bin_position_in_robot_root_frame)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


def create_event_term(x_range, y_range, z_range, roll_range=(0, 0), yaw_range=(0, 0), pitch_range=(0, 0), var_name=None):
    return EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": x_range, "y": y_range, "z": z_range,
                           "roll": roll_range, "yaw": yaw_range, "pitch": pitch_range
                           },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg(var_name),
        },
    )


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


    reset_bowl_position = create_event_term(x_range=(-0.0, 0.0),
                                           y_range=(-0.0, 0.0),
                                           z_range=(0.0, 0.0),
                                           var_name="bowl",
                                           )

    reset_plate_position = create_event_term(x_range=(-0.02, 0.02),
                                           y_range=(-0.02, 0.02),
                                           z_range=(0.0, 0.0),
                                           pitch_range=(-0, 0),
                                           var_name="plate",
                                           )

    reset_tray_position = create_event_term(x_range=(0.0, 0.0),
                                           y_range=(0.0, 0.0),
                                           z_range=(0.0, 0.0),
                                           var_name="tray",
                                           )

    reset_chopsticks_position = create_event_term(x_range=(-0.0, 0.0),
                                           y_range=(-0.0, 0.0),
                                           z_range=(0.0, 0.0),
                                           var_name="chopsticks"
                                           )
    # reset_chopsticks_01_position = create_event_term(x_range=(-0.0, 0.0),
    #                                        y_range=(-0.0, 0.0),
    #                                        z_range=(0.0, 0.0),
    #                                        var_name="chopsticks_01"
    #                                        )

    reset_spoon_position = create_event_term(x_range=(-0.02, 0.02),
                                           y_range=(-0.02, 0.02),
                                           z_range=(0.0, 0.0),
                                           var_name="spoon",
                                           )

    reset_mug_position = create_event_term(x_range=(-0.03, 0.03),
                                           y_range=(-0.02, 0.02),
                                           z_range=(0.0, 0.02),
                                           roll_range=(-0, 0),
                                           yaw_range=(-10, 10),
                                           pitch_range=(-0, 0),
                                           var_name="mug",
                                           )

    create_event_term(x_range=(-0.0, 0.0),
                       y_range=(-0.0, 0.0),
                       z_range=(0.0, 0.0),
                       var_name="broom"
                       )

    # cabinet_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("cabinet", body_names="drawer_handle_top"),
    #         "static_friction_range": (1.0, 1.25),
    #         "dynamic_friction_range": (1.25, 1.5),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 16,
    #     },
    # )



##
# Environment configuration
##


@configclass
class LiftEnvWithCameraCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment with camera."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
