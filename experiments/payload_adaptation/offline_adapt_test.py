from pathlib import Path; import sys
REPO_ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pinocchio
from controller.my_pin_model import PinModel, LEFT_ARM
from controller.slotine_li_controller import SlotineLiController

# get the path to the URDF file
urdf_path = str(REPO_ROOT / "assets" / "g1_37dof.urdf")

# create a PinModel instance
names = list(pinocchio.buildModelFromUrdf(urdf_path).names)[1:] # 37 names, skipping 'universe'
rev = names[::-1]                                         # reversed, like a different simulator order
pm = PinModel(urdf_path, rev, arm_armature=np.zeros(5)) # arm armature is zero for now

# create a SlotineLiController instance
ctrl = SlotineLiController(pm, lam=10, K=[5, 5, 3, 3, 1], adapt=True, 
                                      gamma=[1,0.1,0.1,0.1,0,0,0,0,0,0]) # gamma is a 10-element vector for the adaptation law

# getting the left arm joint indices (5 joints)
arm_isaac_idx = [rev.index(n) for n in LEFT_ARM] # indices of the 5 arm joints in the Isaac order

# setting up a test case
rng = np.random.default_rng(seed=0)
q_isaac = rng.uniform(-0.5, 0.5, size=37) # current random joint positions
v_isaac = rng.uniform(-1, 1, size=37) # current random joint velocities

# TEST A: no movement, desired is current position, zero velocity and acceleration
q_des = q_isaac[arm_isaac_idx] # desired is current position
v_des = v_isaac[arm_isaac_idx] # keep current velocity
a_des = np.zeros(5) # zero desired acceleration

for _ in range(100): # dt --> sim physics step used in Isaac
    tau, s = ctrl.compute_arm_torque(q_isaac, v_isaac, arm_isaac_idx, q_des, v_des, a_des, dt=0.005)
print("TEST A:")
print("tau:", tau)
print("s:", s)
print("theta_hat:", ctrl.theta_hat) # adaptation parameters should change over time
print("\n")

# TEST B: moving to a new desired position
ctrl.reset_adaptation() # reset adaptation parameters
q_des = q_isaac[arm_isaac_idx] + 0.05 # slightly different desired position
v_des = v_isaac[arm_isaac_idx] # keep current velocity
a_des = np.zeros(5) # zero desired acceleration
tau, s = ctrl.compute_arm_torque(q_isaac, v_isaac, arm_isaac_idx, q_des, v_des, a_des, dt=0.005)
print("TEST B:")
print("tau:", tau)
print("s:", s)
print("theta_hat:", ctrl.theta_hat)

