# Adaptive Manipulation with Slotine–Li Adaptive Control and Reinforcement Learning

> Work in progress (started September 2026). No training results yet.

## Goal
Enable a robot arm and hand to lift objects of unknown and varying mass, staying robust to
out-of-distribution payloads while learning from limited data. The approach combines a learned
policy (PPO) with classical adaptive control (Slotine–Li), whose Lyapunov-based analysis
guarantees bounded parameter estimates and convergence of the tracking error.

## Status
- [x] PPO re-implemented from scratch in PyTorch, following the rsl_rl architecture
- [x] Joint-motion and hand-synergy test scripts for the G1 left arm and hand
- [x] Unitree G1 arm–hand grasp environment defined in Isaac Lab (NVIDIA Isaac Sim 6.0)
- [ ] Add arm actions so the policy can lift, not only grasp (current env: fixed base, 1-D hand synergy action only)
- [ ] Randomize object mass during training (currently fixed at 0.2 kg)
- [ ] Connecting PPO to the G1 grasp environment for training (in progress; `g1_grasp_train.py` is a placeholder loop)
- [x] Pinocchio gravity torques for the G1 left arm checked against Isaac Sim (see [`validation/`](validation/README.md))
- [ ] Pinocchio-based online computation of the dynamics regressor (in progress)
- [ ] Slotine–Li adaptive control law (theory studied; implementation next)
- [ ] Combining the adaptive controller with the learned policy
- [ ] Evaluation on out-of-distribution payload masses

## Approach (planned)
- **Learned policy:** PPO learns basic manipulation behavior.
- **Adaptive control:** Slotine–Li adapts estimates of the payload's inertial parameters online.
  It relies on linearity of the dynamics in the parameters, skew-symmetry of Ṁ − 2C, and a
  positive-definite inertia matrix. A sliding variable based on a reference velocity avoids
  using noisy measured accelerations.
- **Dynamics:** Pinocchio computes the regressor quickly enough for online adaptation.

## Repository structure
```
ppo/                         PPO written from scratch in PyTorch
  actor.py                   Gaussian actor (learned log-std) and critic MLPs
  PPO.py                     Clipped-surrogate update, entropy bonus, grad clipping, KL / clip-fraction metrics
  rollout_buffer.py          Rollout storage, GAE, advantage normalisation, minibatching
  rollout_collector.py       Rollout collection with timeout bootstrapping
envs/
  g1_grasp_build.py          Isaac Lab DirectRLEnv: fixed-base G1, left arm + hand, cube on a platform,
                             34-D observation, 1-D hand-synergy action (open ↔ close).
                             First step: grasp only. Arm control for lifting comes next.
  g1_grasp_train.py          Registers the env as G1-Grasp-v0 (work in progress, see below)
tools/
  joint_motion_test.py       Moves one joint at a time with a sine wave to inspect each joint's motion
  open_close_test.py         Sweeps the hand synergy to check each finger's closing direction
  usd_to_urdf.py             Converts the Isaac Lab G1 USD into a dynamics-only URDF for Pinocchio
validation/                  Checks that Pinocchio's model of the G1 arm matches Isaac Sim (see below)
```

## Model validation
The adaptive controller will compute the arm's joint torques from a Pinocchio model of the robot,
so that model must match the robot simulated in Isaac Sim. Otherwise the stability analysis no
longer applies, and the adaptation law would absorb arm-model errors into its payload estimate.
[`validation/`](validation/README.md) holds a standalone check, separate from training, that
starts with gravity: the left arm is driven by Pinocchio's gravity torques and by Isaac's own, and
the two runs are compared. See [`validation/README.md`](validation/README.md) for the setup,
how to run it and the figures.

## Requirements
- NVIDIA Isaac Sim 6.0 and Isaac Lab (installed separately; developed with Isaac Sim 6.0.1 and Isaac Lab 6.1.14)
- Python 3.12
- Python packages: see `requirements.txt` (includes Pinocchio, needed for `validation/`)

## How to run
Scripts are launched through Isaac Lab's launcher from the Isaac Lab directory (Windows shown;
use `isaaclab.sh` on Linux). Isaac Lab launcher flags such as `--headless` and `--device` are supported.

```bash
# Sweep the G1 hand synergy (open -> close) to check finger directions
isaaclab.bat -p <path-to-repo>/tools/open_close_test.py --num_envs 1

# Move each left-arm and hand joint in turn with a sine wave
isaaclab.bat -p <path-to-repo>/tools/joint_motion_test.py --num_envs 1
```

`envs/g1_grasp_train.py` is not finished. It registers the G1 grasp env as a Gymnasium environment
(`G1-Grasp-v0`) and imports the PPO modules from `ppo/`, but PPO is not connected yet: the script
currently steps the environment with zero actions as a placeholder. No training happens yet.

## Acknowledgements
The PPO implementation follows the architecture of
[rsl_rl](https://github.com/leggedrobotics/rsl_rl) (BSD-3-Clause).
The environments are built on [Isaac Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause)
and use its Unitree G1 asset configuration. Rigid-body dynamics are computed with
[Pinocchio](https://github.com/stack-of-tasks/pinocchio).
