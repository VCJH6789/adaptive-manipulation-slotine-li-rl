from pathlib import Path; import sys
REPO_ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO_ROOT))

import argparse
from isaaclab.app import AppLauncher
import numpy as np

# =========================================================
# 1. Command-line arguments
# =========================================================

parser = argparse.ArgumentParser(
    description="G1 Slotine-Li Controller Test",
)

# Our own custom argument
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of parallel environments.",
)

parser.add_argument(
    "--payload",
    type=float,
    default=1.0,
    help="Payload mass to be added to the palm.",
)

parser.add_argument(
    "--adapt",
    action="store_true", # absent -> False, present -> True
    help="Enable Adaptation to Unknown Payload.",
)

# Add Isaac Lab launcher arguments
# such as --device, --viz, --headless, etc.
AppLauncher.add_app_launcher_args(parser)

# Parse CLI arguments
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =========================================================
# Simulation loop
# =========================================================

from g1_test_build import G1ReachEnv, G1ReachEnvCfg
import math
import torch

cfg = G1ReachEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
cfg.sim.device = args_cli.device
cfg.payload_mass = args_cli.payload
cfg.adapt = args_cli.adapt
env = G1ReachEnv(cfg)

# begin the physics and simulations
obs, info = env.reset()

# settings for simulation step
step_count = 0

policy_dt = cfg.sim.dt * cfg.decimation
max_steps = int(60.0 / policy_dt)        # 60 seconds

while simulation_app.is_running() and step_count < max_steps:

    # ---------------------------------------------
    # Sine Wave Action for each joint
    # ---------------------------------------------

    time_s = step_count * cfg.sim.dt * cfg.decimation

    actions = torch.full((env.num_envs, 1),
                         time_s,
                         device=env.device,
                         )

    # ---------------------------------------------
    # One RL Step
    # ---------------------------------------------

    obs, reward, terminated, truncated, info = (
        env.step(actions)
    )

    if step_count % 50 == 0:
        print("Step: ", step_count, "\n")

        # observing the distance between the palm and cube
        print("Relative pose between Cube and Palm: ",
            obs["rel_cube"][:], 
            "\n")
        print("Distance to Cube: ",
            torch.linalg.norm(obs["rel_cube"][:],
                                dim=-1),
                                "\n")
        print("Current Reward: ", reward[:], "\n\n")

    step_count += 1

# Save the arm joint log for analysis
tag = "adapt" if args_cli.adapt else "fixed"
LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
np.savez(LOG_DIR / f"track_log_{tag}_m{args_cli.payload}.npz",
         **{k: np.array(v) for k, v in env.log_track.items()})
    
    
env.close()

simulation_app.close()

