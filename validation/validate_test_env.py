import argparse
from isaaclab.app import AppLauncher

# =========================================================
# 1. Command-line arguments
# =========================================================

parser = argparse.ArgumentParser(
    description="Validate Pinocchio Gravity vs Isaac Sim Gravity" \
    " to check for mismatch in the gravity compensation torques for the arm joints"
)

# Our own custom argument
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of parallel environments.",
)
parser.add_argument(
    "--test_mode",
    type=str,
    default="zero",
    choices=["zero", "gravity", "pin_gravity"]
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

from validation.validate_pin_isaac_gravity import G1ReachEnv, G1ReachEnvCfg
import math
import torch
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"

cfg = G1ReachEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
cfg.sim.device = args_cli.device

env = G1ReachEnv(cfg)
env.test_mode = args_cli.test_mode

# begin the physics and simulations
obs, info = env.reset()

# settings for simulation step
step_count = 0
frequency = 0.25 # hz
amplitude = 0.5
step_count = 0

policy_dt = cfg.sim.dt * cfg.decimation
max_steps = int(10.0 / policy_dt)        # 10 seconds

while simulation_app.is_running() and step_count < max_steps:

    # ---------------------------------------------
    # Sine Wave Action for each joint
    # ---------------------------------------------

    time_s = step_count * cfg.sim.dt * cfg.decimation
    action_val = amplitude * math.sin(2.0*math.pi*frequency*time_s)
    action_val += 0.5

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

    if step_count % 50 == 0:
        print("Step: ", step_count, "\n")

    step_count += 1

# Save the arm joint log for analysis
arm_log = torch.stack(env.log_arm_pos)   # shape: (physics_steps, 5)
LOG_DIR.mkdir(exist_ok=True)
torch.save(arm_log, LOG_DIR / f"arm_log_{args_cli.test_mode}.pt")
    
    
env.close()

simulation_app.close()

