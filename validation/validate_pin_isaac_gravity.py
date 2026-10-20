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

import warp as wp

from pathlib import Path
from validation.my_pin_model import PinModel

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

        self.test_mode = "zero"
        self.log_arm_pos = []

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
        # Pinnochio Model for G1
        # =====================================================
        # get the path to the URDF file
        urdf_path = Path(__file__).resolve().parent / "g1_37dof.urdf"
        self.pin_model = PinModel(str(urdf_path), self.robot.data.joint_names)
        self.dbg_step = 0

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

        pass

    def _apply_action(self):
        # ---------------------------------------------
        # Apply to each individual joint at a time
        # ---------------------------------------------

        # Arm: our own torques
        if self.test_mode == "zero":
            tau = torch.zeros(self.num_envs, len(self.left_arm_ids), device=self.device)
        elif self.test_mode == "gravity":                     # Isaac's gravity
            g_all = wp.to_torch(self.robot.root_physx_view.get_gravity_compensation_forces())
            tau = g_all[:, self.left_arm_ids]
        elif self.test_mode == "pin_gravity":                 # your Pinocchio gravity (env 0)
            q_np = self.robot.data.joint_pos[0].cpu().numpy()
            g_pin = self.pin_model.gravity_arm(q_np)
            tau = torch.tensor(g_pin, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            raise ValueError(f"Unknown test_mode: {self.test_mode}")
          
        self.robot.set_joint_effort_target(
            target=tau,
            joint_ids=self.left_arm_ids
        )

        # Log arm joint positions (env 0)
        self.log_arm_pos.append(
            self.robot.data.joint_pos[0, self.left_arm_ids].cpu().clone()
        )

    def _get_observations(self):

        return {"policy": torch.zeros(
            self.num_envs, self.cfg.observation_space, device=self.device)}

    def _get_rewards(self):

        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(time_out), time_out

    def _reset_idx(self, env_ids):

        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # Let DirectRLEnv reset its own buffers first
        super()._reset_idx(env_ids)

        # Put every joint back to its default pose, at rest
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # PD-driven joints (body, hand) hold the default pose.
        # The arm's PD is off, so its position target has no effect.
        self.robot.set_joint_position_target_index(
            target=joint_pos, env_ids=env_ids, full_data=True,
        )

        # Clear the debug log and counter, so each run starts fresh
        self.log_arm_pos = []
        self.dbg_step = 0

