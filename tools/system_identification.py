import argparse
from isaaclab.app import AppLauncher

# =========================================================
# 1. Command-line arguments
# =========================================================

parser = argparse.ArgumentParser(
    description="G1 System Identification"
)

# Our own custom argument
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of parallel environments.",
)

# Add Isaac Lab launcher arguments
# such as --device, --viz, --headless, etc.
AppLauncher.add_app_launcher_args(parser)

# Parse CLI arguments
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =========================================================
# 2. Launch Isaac Sim
# =========================================================

import torch

import isaaclab.sim as sim_utils

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.configclass import configclass

from isaaclab_assets import G1_CFG

import math

@configclass
class G1ReachEnvCfg(DirectRLEnvCfg):

    # =====================================================
    # RL environment
    # =====================================================

    decimation = 2

    episode_length_s = 300.0

    action_space = 1

    observation_space = 12

    state_space = 0

    # =====================================================
    # Simulation
    # =====================================================

    sim: SimulationCfg = SimulationCfg(
        dt=0.01,
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

        # cache the arm config
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

        self.arm_targets = (
            self.default_left_arm_pos.clone()
        )

        # Increment update 
        self.step_count = 0

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

        # ---------------------------------------------
        # Policy action
        # ---------------------------------------------

        self.actions = torch.clamp(
            actions.clone(),
            min=-1.0,
            max=1.0,
        )

        # ---------------------------------------------
        # [-1,1] -> arm position target
        # ---------------------------------------------

        self.arm_targets = (
            self.default_left_arm_pos
            + self.cfg.action_scale
            * self.actions
        )

        # ---------------------------------------------
        # [-1,1] -> hand position target
        # ---------------------------------------------
        self.hand_targets = (
            self.default_left_hand_pos
            + self.cfg.action_scale
            * self.actions
        )

        # ---------------------------------------------
        # Physical safety limits
        # ---------------------------------------------
        # Arm Joints
        self.arm_targets = torch.clamp(
            self.arm_targets,
            min=self.arm_lower_limits,
            max=self.arm_upper_limits,
        )
        # Hand Joints
        self.hand_targets = torch.clamp(
            self.hand_targets,
            min=self.hand_lower_limits,
            max=self.hand_upper_limits,
        )

    def _apply_action(self):
        # ---------------------------------------------
        # Apply to each individual joint at a time
        # ---------------------------------------------
        frequency = 0.25
        dt_control = self.cfg.sim.dt * self.cfg.decimation
        period_s = 1.0 / frequency
        steps_per_cycle = int(period_s / dt_control)

        # combine joints
        total_len = len(self.left_arm_ids) + len(self.left_hand_ids)
        idx = (self.step_count // steps_per_cycle) % total_len
        target = torch.cat([self.default_left_arm_pos,
                            self.default_left_hand_pos], dim=1)
        ref = torch.cat([self.arm_targets, 
                         self.hand_targets], dim=1)
        target[:, idx] = ref[:, idx]

        num_arm = len(self.left_arm_ids)

        # Action to Arm Joints
        # ---------------------------------------------
        # Arm: first 5 columns
        # ---------------------------------------------
        self.robot.set_joint_position_target_index(
            target=target[:, :num_arm],
            joint_ids=self.left_arm_ids,
        )

        # Action to Hand Joints
        # ---------------------------------------------
        # Hand: remaining 7 columns
        # ---------------------------------------------
        self.robot.set_joint_position_target_index(
            target=target[:, num_arm:],
            joint_ids=self.left_hand_ids,
        )

        all_names = (
            self.left_arm_names
            + self.left_hand_names
        )

        relative = target - torch.cat(
            [
                self.default_left_arm_pos,
                self.default_left_hand_pos,
            ],
            dim=1,
        )

        print(
            f"Testing: {all_names[idx]} | "
            f"relative displacement = "
            f"{relative[0, idx].item():.4f} rad"
        )

        # increment update
        self.step_count += 1

    def _get_observations(self):

        # ---------------------------------------------
        # Actual arm positions
        # ---------------------------------------------

        arm_pos = self.robot.data.joint_pos[
            :, self.left_arm_ids
        ]

        arm_pos_relative = (
            arm_pos
            - self.default_left_arm_pos
        )

        # ---------------------------------------------
        # Actual hand positions
        # ---------------------------------------------

        hand_pos = self.robot.data.joint_pos[
            :, self.left_hand_ids
        ]

        hand_pos_relative = (
            hand_pos
            - self.default_left_hand_pos
        )

        # ---------------------------------------------
        # 13-D observation
        # ---------------------------------------------

        obs = torch.cat(
            [
                arm_pos_relative,
                hand_pos_relative,
            ],
            dim=-1,
        )

        return {
            "policy": obs, 
            "arm": [self.left_arm_names, arm_pos_relative], 
            "hand": [self.left_hand_names, hand_pos_relative]
        }

    def _get_rewards(self):

        return torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )

    def _get_dones(self):

        terminated = torch.zeros(
        self.num_envs,
        dtype=torch.bool,
        device=self.device,
        )

        time_out = (
            self.episode_length_buf
            >= self.max_episode_length - 1
        )

        return terminated, time_out

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

        # ---------------------------------------------
        # Restore our hand target buffer
        # ---------------------------------------------
        
        self.hand_targets[env_ids] = (
            self.default_left_hand_pos[
                env_ids
            ]
        )

# =========================================================
# Simulation loop
# =========================================================

cfg = G1ReachEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
cfg.sim.device = args_cli.device

env = G1ReachEnv(cfg)

# begin the physics and simulations
obs, info = env.reset()

# settings for simulation step
step_count = 0
frequency = 0.25 # hz
amplitude = 1.0
step_count = 0

while simulation_app.is_running():
    # ---------------------------------------------
    # Sine Wave Action for each joint
    # ---------------------------------------------

    time_s = step_count * cfg.sim.dt * cfg.decimation
    action_val = amplitude * math.sin(2.0*math.pi*frequency*time_s)

    actions = torch.full((env.num_envs, 1),
                         action_val,
                         device=env.device,
                         )

    # ---------------------------------------------
    # One RL Step
    # ---------------------------------------------

    obs, reward, terminated, truncated, info = (
        env.step(actions)
    )

    
    step_count += 1
    
    
env.close()

simulation_app.close()
