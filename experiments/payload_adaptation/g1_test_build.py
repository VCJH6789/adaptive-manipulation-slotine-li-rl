from pathlib import Path; import sys
REPO_ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO_ROOT))

# =========================================================
# 2. Launch Isaac Sim
# =========================================================

import torch

import isaaclab.sim as sim_utils

from isaaclab.assets import Articulation, RigidObjectCfg, RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.configclass import configclass
from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab_assets import G1_CFG
from isaaclab.sensors import ContactSensorCfg

import math
import warp as wp
import numpy as np

import os
from controller.my_pin_model import PinModel
from controller.slotine_li_controller import SlotineLiController

@configclass
class G1ReachEnvCfg(DirectRLEnvCfg):

    # =====================================================
    # RL environment
    # =====================================================

    decimation = 2

    episode_length_s = 300.0

    action_space = 1

    observation_space = 34

    state_space = 0

    payload_mass = 0.5 # kg, default payload mass to be added to the palm

    adapt = True # enable adaptation to unknown payload

    # =====================================================
    # Simulation
    # =====================================================

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
    )

    # =====================================================
    # Scene
    # =====================================================

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=3.0,
        replicate_physics=True,
    )

    # =====================================================
    # Robot 
    # =====================================================

    robot_cfg = G1_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    robot_cfg.spawn.articulation_props.fix_root_link = True

    # =====================================================
    # Task
    # =====================================================

    action_scale = 1.0

#=================================================

class G1ReachEnv(DirectRLEnv):

    cfg: G1ReachEnvCfg

    def __init__(
        self,
        cfg: G1ReachEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):

        super().__init__(
            cfg,
            render_mode,
            **kwargs,
        )

        # =====================================================
        # Inspection of G1 Joints & Bodies
        # =====================================================

        # inspecting G1 joints
        print("Robot Joint Names")
        for idx, joint_name in enumerate(self.robot.joint_names):
            print(f"DoF Index {idx}: {joint_name}")
        print("-----------------------------------------------")
        
        # inspecting G1 bodies
        print("Robot Body Names")
        for idx, body_name in enumerate(self.robot.body_names):
            print(f"DoF Index {idx}: {body_name}")
        print("-----------------------------------------------")

        # =====================================================
        # Getting pelvis pose of robot for platform pose    
        # =====================================================
        
        # getting the pelvis pose of robot
        pelvis_id, _ = self.robot.find_bodies("pelvis")
        pelvis_pose = self.robot.data.body_pos_w[
            :, pelvis_id[0], :
        ]
        print("Pelvis World Positions", pelvis_pose[0])

        # =====================================================
        # Getting palm body
        # =====================================================
        self.left_palm_id, _ = (
            self.robot.find_bodies("left_palm_link")
        )

        # =====================================================
        # Getting Arm and Hand joints
        # =====================================================

        # find G1 arm joints
        self.left_arm_ids, self.left_arm_names = (
            self.robot.find_joints(
                [
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_pitch_joint",
                    "left_elbow_roll_joint",
                ],
                preserve_order=True,
            )
        )
        # find G1 hand joints
        self.left_hand_ids, self.left_hand_names = (
            self.robot.find_joints(
                [
                    "left_zero_joint",
                    "left_one_joint",
                    "left_two_joint",
                    "left_three_joint",
                    "left_four_joint",
                    "left_five_joint",
                    "left_six_joint",
                ],
                preserve_order=True,
            )
        )

        # =====================================================
        # Getting Arm and Hand soft joint limits
        # =====================================================

        print("Left_Hand_Soft_Pos_Limits: \n",
            self.robot.data.soft_joint_pos_limits[
                0, self.left_hand_ids
            ]
        )

        print("Left_Arm_Soft_Pos_Limits: \n",
            self.robot.data.soft_joint_pos_limits[
                0, self.left_arm_ids
            ]
        )

        # =====================================================
        # Getting Arm and Hand default poses
        # =====================================================

        print("Default left hand pose:",
              self.robot.data.default_joint_pos[
                  0, self.left_hand_ids
              ])

        print("Default left arm pose:",
            self.robot.data.default_joint_pos[
                0, self.left_arm_ids
            ])

        # =====================================================
        # Setting Arm and Hand config
        # =====================================================

        # cache the hand config
        self.default_left_hand_pos = self.robot.data.default_joint_pos[
                    :, self.left_hand_ids
                ].clone()
        
        self.hand_lower_limits = (
            self.robot.data.soft_joint_pos_limits[
                :, self.left_hand_ids, 0
            ]
        )

        self.hand_upper_limits = (
            self.robot.data.soft_joint_pos_limits[
                :, self.left_hand_ids, 1
            ]
        )

        self.hand_targets = (
            self.default_left_hand_pos.clone()
        )

        # cache the arm config----------------------------
        self.default_left_arm_pos = (
            self.robot.data.default_joint_pos[
                :, self.left_arm_ids
            ].clone()
        )

        self.arm_lower_limits = (
            self.robot.data.soft_joint_pos_limits[
                :, self.left_arm_ids, 0
            ]
        )

        self.arm_upper_limits = (
            self.robot.data.soft_joint_pos_limits[
                :, self.left_arm_ids, 1
            ]
        )

        self.arm_effort_limits = (
            self.robot.data.joint_effort_limits[
                :, self.left_arm_ids
            ]
        )

        self.arm_targets = (
            self.default_left_arm_pos.clone()
        )

        # =====================================================
        # Setting palm config
        # =====================================================

        self.left_palm_pose = self.robot.data.body_pos_w[
                            :, self.left_palm_id[0], :
                        ]

        # =====================================================
        # Specifically switch off PD controller for left arm only
        # =====================================================

        # Left arm only: switch off the built-in PD
        self.robot.write_joint_stiffness_to_sim(
            0.0, 
            joint_ids=self.left_arm_ids
        )
        self.robot.write_joint_damping_to_sim(
            0.0, 
            joint_ids=self.left_arm_ids
        )

        # =====================================================
        # Motor Armature for G1
        # =====================================================
        arm_armature = self.robot.data.joint_armature[
            0, self.left_arm_ids
        ].cpu().numpy()
        print("Armature for G1 left arm joints:", arm_armature)

        # =====================================================
        # Pinnochio Model for G1 dynamics computation
        # =====================================================
        # get the path to the URDF file
        urdf_path = str(REPO_ROOT / "assets" / "g1_37dof.urdf")
        self.pin_model = PinModel(
            urdf_path, 
            self.robot.data.joint_names, 
            arm_armature=arm_armature
        )

        # =====================================================
        # Slotine-Li Controller for G1
        # =====================================================
        self.adapt = cfg.adapt # enable adaptation

        self.ctrl = SlotineLiController(
            self.pin_model, 
            lam=10, K=[5, 5, 3, 3, 1], adapt=self.adapt,
            gamma=[1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0]
        )
        # create a timer
        self.phys_step = 0
        # create a log for arm joint positions
        self.log_track = {"q": [], "q_des": [], "s": [], "tau": [], "theta_hat": []}

        # =====================================================
        # Unknown Payload: adding extra mass on palm
        # =====================================================
        self.payload_mass = cfg.payload_mass # kg
        # getting the current mass values
        view = self.robot.root_physx_view
        masses = wp.to_torch(view.get_masses()).clone()
        print("masses device/shape:", masses.device, masses.shape)
        # getting the palm body index
        palm_body_index = self.left_palm_id[0]
        print("palm mass before:", masses[0, palm_body_index].item())
        # adding the payload mass to the palm body
        masses[0, palm_body_index] += self.payload_mass
        env_ids = torch.arange(self.num_envs, device=masses.device, dtype=torch.int32)
        view.set_masses(wp.from_torch(masses), wp.from_torch(env_ids))
        print("palm mass after:", masses[0, palm_body_index].item())

    # define scene setup
    def _setup_scene(self):

        # =================================================
        # Robot
        # =================================================

        self.robot = Articulation(
            self.cfg.robot_cfg
        )

        # =================================================
        # Ground
        # =================================================

        spawn_ground_plane(
            prim_path="/World/Ground",
            cfg=GroundPlaneCfg(),
        )

        # =================================================
        # Dynamic Cube Object
        # =================================================

        # setting the pose of the target in world frame
        TARGET_POS = (
            0.40, 0.12, 0.84
        )
        # setup the cube object config
        object_cfg = sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0,0.0,0.0)),
        )
        cube_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",
            spawn=object_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=TARGET_POS
            ),
        )

        # =================================================
        # Static Platform for Reaching Tasks
        # =================================================

        # setting the pose of the static platform
        platform_pos = (
            0.40, 0.00, 0.78
        )
        # setup the platform config
        platform_cfg = sim_utils.CuboidCfg(
            size=(0.5, 1.0, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0,1.0,1.0)
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True # keep it static
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        platform_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Platform",
            spawn=platform_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=platform_pos
            ),
        )

        # =================================================
        # Clone environments
        # =================================================

        self.scene.clone_environments(
            copy_from_source=False
        )

        if self.device == "cpu":
            self.scene.filter_collisions(
                global_prim_paths=[]
            )

        # =================================================
        # Register robot
        # =================================================

        self.scene.articulations[
            "robot"
        ] = self.robot

        # =================================================
        # Register object
        # =================================================
        self.object = RigidObject(cube_cfg)
        self.scene.rigid_objects["object"] = self.object

        # =================================================
        # Register static platform
        # =================================================
        self.platform = RigidObject(platform_cfg)
        self.scene.rigid_objects["platform"] = self.platform

        # =================================================
        # Lighting
        # =================================================

        light_cfg = sim_utils.DomeLightCfg(
            intensity=3000.0,
        )

        light_cfg.func(
            "/World/Light",
            light_cfg,
        )

    # define physics setup
    def _pre_physics_step(
        self,
        actions: torch.Tensor,
    ):
        # actions --> timer = step_count * cfg.sim.dt * cfg.decimation

        # ---------------------------------------------
        # Hand policy action
        # ---------------------------------------------
        hand_amplitude = 0.5
        hand_frequency = 0.25 # Hz

        hand_action = hand_amplitude * torch.sin(2.0*math.pi*hand_frequency*actions)
        hand_action += 0.5

        # hand joints for open
        hand_open = torch.cat([self.hand_lower_limits[:, :3],
                                self.hand_upper_limits[:, 3:]],
                                dim=1) # stack horizontally
        hand_close = torch.cat([self.hand_upper_limits[:, :3],
                                self.hand_lower_limits[:, 3:]],
                                dim=1)
        # synergy grasping equation
        hand_state = (hand_open + hand_action *
                        (hand_close - hand_open))

        # ---------------------------------------------
        # [-1,1] -> hand position target
        # ---------------------------------------------
        self.hand_targets = hand_state

        # ---------------------------------------------
        # Physical safety limits
        # ---------------------------------------------
        # Hand Joints
        self.hand_targets = torch.clamp(
            self.hand_targets,
            min=self.hand_lower_limits,
            max=self.hand_upper_limits,
        )

        # single float type
        self.phys_step = float(actions[0, 0]) # store the timer

    def arm_trajectory(self, t):
        # compute the desired trajectory for the 5 arm joints at time t
        q0 = self.default_left_arm_pos[0].cpu().numpy()
        A = np.array([0.3, 0.2, 0.2, 0.3, 0.3])   # rad, per joint
        w = 2 * np.pi * 0.25                      # 0.25 Hz: one cycle every 4 s

        q_des = q0 + A * np.sin(w * t)
        v_des = A * w * np.cos(w * t)
        a_des = -A * w**2 * np.sin(w * t)
        return q_des, v_des, a_des

    def _apply_action(self):
        # ---------------------------------------------
        # Apply to each individual joint at a time
        # ---------------------------------------------

        # Arm: compute the desired trajectory for the 5 arm joints at time t
        q_des, v_des, a_des = self.arm_trajectory(self.phys_step)
        self.phys_step += self.cfg.sim.dt # increment the timer for next step

        # compute the torque for the 5 arm joints using Slotine-Li controller
        tau, s = self.ctrl.compute_arm_torque(
            self.robot.data.joint_pos[0].cpu().numpy(),
            self.robot.data.joint_vel[0].cpu().numpy(),
            self.left_arm_ids,
            q_des, v_des, a_des,
            dt = self.cfg.sim.dt
        )

        # change tau to a torch tensor type
        tau = torch.tensor(tau, dtype=torch.float32, device=self.device).unsqueeze(0) # shape: (1, 5)

        tau = torch.clamp(
            tau,
            -self.arm_effort_limits,
            self.arm_effort_limits
        )

        self.robot.set_joint_effort_target(
            target=tau,
            joint_ids=self.left_arm_ids
        )

        # ---------------------------------------------

        # Hand: unchanged, PD position targets (synergy)
        self.robot.set_joint_position_target_index(
            target=self.hand_targets,
            joint_ids=self.left_hand_ids,
        )

        # Log arm joint positions (env 0)
        self.log_track["q"].append(
            self.robot.data.joint_pos[0, self.left_arm_ids].cpu().numpy()
        )
        self.log_track["q_des"].append(
            q_des
        )
        self.log_track["s"].append(
            s
        )
        self.log_track["tau"].append(
            tau[0].cpu().clone().numpy()
        )
        self.log_track["theta_hat"].append(
            self.ctrl.theta_hat.copy()
        )

    def _get_observations(self):

        # ---------------------------------------------
        # Actual arm positions & velocities
        # ---------------------------------------------

        arm_pos = self.robot.data.joint_pos[
            :, self.left_arm_ids
        ]

        arm_pos_relative = (
            arm_pos
            - self.default_left_arm_pos
        )

        arm_vel = self.robot.data.joint_vel[
            :, self.left_arm_ids
        ]

        # ---------------------------------------------
        # Actual hand positions & velocities & tracking
        # ---------------------------------------------

        hand_pos = self.robot.data.joint_pos[
            :, self.left_hand_ids
        ]

        hand_pos_relative = (
            hand_pos
            - self.default_left_hand_pos
        )

        hand_vel = self.robot.data.joint_vel[
            :, self.left_hand_ids
        ]

        hand_tracking_error = (
            self.hand_targets - hand_pos
        )

        # ---------------------------------------------
        # Relative pose between palm and cube object
        # ---------------------------------------------
        
        self.left_palm_pose = self.robot.data.body_pos_w[
            :, self.left_palm_id[0], :
        ]

        rel_cube = (self.object.data.root_pos_w - 
                    self.left_palm_pose)

        # ---------------------------------------------
        # 13-D observation
        # ---------------------------------------------

        obs = torch.cat(
            [
                arm_pos_relative,
                arm_vel,
                hand_pos_relative,
                hand_vel,
                hand_tracking_error,
                rel_cube,
            ],
            dim=-1,
        )

        return {
            "policy": obs,
            "rel_cube": rel_cube, 
        }

    def _get_rewards(self):

        # # ---------------------------------------------
        # # Relative distance metric between palm and cube
        # # ---------------------------------------------
        # rel_cube = (self.object.data.root_pos_w - 
        #                     self.left_palm_pose)
        # # non_linear reward function
        # reach_gain = 1.0
        # reach_reward = torch.exp(-reach_gain * 
        #         torch.linalg.norm(rel_cube, dim=-1))
        
        # # ---------------------------------------------
        # # Grasping metric based on distance
        # # ---------------------------------------------
        # # non-linear reward function
        # grasp_gain = 1.0
        # grasp_alpha = self.actions[:, -1]
        # grasp_reward = (grasp_alpha * 
        #                 torch.exp(-grasp_gain * rel_cube))

        # reward = reach_reward + grasp_reward

        reward = torch.zeros(self.num_envs, device=self.device)

        return reward

    def _get_dones(self):

        # ---------------------------------------------
        # Getting the cube object state
        # ---------------------------------------------
        cube_pos = self.object.data.root_pos_w

        # ---------------------------------------------
        # Failure:
        # Cube falls below the platform
        # ---------------------------------------------
        min_cube_height = self.platform.data.root_pos_w[0][2]
        cube_fallen = (
            cube_pos[:, 2] < min_cube_height
        )

        # ---------------------------------------------
        # Episode Timeout
        # ---------------------------------------------
        time_out = (
            self.episode_length_buf
            >= self.max_episode_length - 1
        )

        return cube_fallen, time_out

    def _reset_idx(self, env_ids):

        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # Let DirectRLEnv reset its own buffers first
        super()._reset_idx(env_ids)

        # ---------------------------------------------
        # Reset physical joint state
        # ---------------------------------------------

        joint_pos = (
            self.robot.data.default_joint_pos[
                env_ids
            ].clone()
        )

        joint_vel = (
            self.robot.data.default_joint_vel[
                env_ids
            ].clone()
        )

        self.robot.write_joint_state_to_sim(
            joint_pos,
            joint_vel,
            env_ids=env_ids,
        )

        # ---------------------------------------------
        # Reset actuator targets too
        # ---------------------------------------------

        self.robot.set_joint_position_target_index(
            target=joint_pos,
            env_ids=env_ids,
            full_data=True,
        )

        # ---------------------------------------------
        # Restore our arm target buffer
        # ---------------------------------------------

        self.arm_targets[env_ids] = (
            self.default_left_arm_pos[
                env_ids
            ]
        )

        self.robot.set_joint_effort_target(
            target=torch.zeros(len(env_ids), len(self.left_arm_ids), device=self.device),
            joint_ids=self.left_arm_ids,
            env_ids=env_ids,
        )

        # ---------------------------------------------
        # Restore our hand target buffer
        # ---------------------------------------------

        hand_open = torch.cat([self.hand_lower_limits[:, :3],
                                self.hand_upper_limits[:, 3:]],
                                dim=1) # stack horizontally
        
        self.hand_targets[env_ids] = (
            hand_open[
                env_ids
            ]
        )

        # ---------------------------------------------
        # Reset object cube
        # ---------------------------------------------
        cube_pos = self.object.data.default_root_pose[
            env_ids
        ].clone()

        cube_vel = self.object.data.default_root_vel[
            env_ids
        ].clone()

        # Add each env's world origin
        cube_pos[:, :3] += self.scene.env_origins[
            env_ids
        ]

        # ---------------------------------------------
        # Reset adaptation parameters
        # ---------------------------------------------
        self.ctrl.reset_adaptation()

        # resetting settings
        self.object.write_root_pose_to_sim_index(
            root_pose=cube_pos,
            env_ids=env_ids,
        )
        self.object.write_root_velocity_to_sim_index(
            root_velocity=cube_vel,
            env_ids=env_ids,
        )

        # reset internal buffer
        self.object.reset(env_ids)

