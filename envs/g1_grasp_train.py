import argparse
from isaaclab.app import AppLauncher

# =========================================================
# 1. Command-line arguments
# =========================================================

parser = argparse.ArgumentParser(
    description="G1 grasp training (work in progress)"
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
# 2. Register & Creating Gymnasium Environment
# =========================================================
import gymnasium as gym
from g1_grasp_build import G1ReachEnv, G1ReachEnvCfg

# registering env
gym.register(
    id="G1-Grasp-v0",
    entry_point=G1ReachEnv,
    disable_env_checker=True,
)
print("G1-Grasp-v0" in gym.registry)

# creating gym env
cfg = G1ReachEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
cfg.sim.device = args_cli.device

env = gym.make(
    "G1-Grasp-v0",
    cfg=cfg,
)

# =========================================================
# 3. Setting Neural Network
# =========================================================
import sys
from pathlib import Path

# repo root = parent of envs/, so `ppo` is importable as a package
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from ppo.actor import Actor, Critic
from ppo.rollout_buffer import RolloutBuffer
from ppo.rollout_collector import RolloutCollector
from ppo.PPO import PPO

# Generating random seed
torch.manual_seed(42)

# establishing object instances


# =========================================================
# 4. Begin Simulation
# =========================================================

obs, info = env.reset()
obs = obs["policy"]

for step in range(1000):

    # Policy network generate actions
    actions = torch.zeros(
        (args_cli.num_envs, 6),
        device=env.unwrapped.device,
    )

    # One env step
    obs, reward, terminated, truncated, info = (
        env.step(actions)
    )

    if step % 100 == 0:
        print("Reward:", reward)
        print("TERMINATED:", terminated)
        print("TRUNCATED:", truncated)
        print("NEXT OBS:", obs["policy"].shape)
