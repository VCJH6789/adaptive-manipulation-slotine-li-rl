import torch
import matplotlib.pyplot as plt
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"

log = torch.load(LOG_DIR / "arm_log_gravity.pt")
names = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw",
         "elbow_pitch", "elbow_roll"]
for i, n in enumerate(names):
    plt.plot(log[:, i], label=n)
plt.xlabel("physics step")
plt.ylabel("joint position (rad)")
plt.legend()
plt.title("Left arm under gravity compensation torque")
plt.show()
