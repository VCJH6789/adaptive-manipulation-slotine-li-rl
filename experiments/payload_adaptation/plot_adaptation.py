from pathlib import Path; import sys
REPO_ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = REPO_ROOT / "logs"
FIG_DIR = Path(__file__).resolve().parent / "figures"

fixed = np.load(LOG_DIR / "track_log_fixed_m0.5.npz") # load the log file saved from g1_test_train.py
adapt = np.load(LOG_DIR / "track_log_adapt_m0.5.npz") # load the log file saved from g1_test_train.py
dt = 0.005            # physics dt

e_fixed = np.linalg.norm(fixed["q"] - fixed["q_des"], axis=1) # error norm for fixed
e_adapt = np.linalg.norm(adapt["q"] - adapt["q_des"], axis=1) # error norm for adapt

t = np.arange(len(e_fixed)) * dt

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

axes[0].plot(t, e_fixed[:], linestyle='--', label="No_Adaptation")
axes[0].plot(t, e_adapt[:], label="Adaptative Slotine-Li")
axes[0].set_ylabel("Error ||q - q_des|| (rad)")
axes[0].legend()

axes[1].plot(t, adapt["theta_hat"][:, 0], label="Estimated Payload") # take only the mass value (first element of theta_hat)
axes[1].axhline(0.5, color="r", linestyle="--", linewidth=1.8,
                label="True payload (0.5 kg)", zorder=3)
axes[1].set_ylim(0, 0.6)     # headroom so the line isn't on the frame
axes[1].set_ylabel("Estimated Payload Mass (kg)")
axes[1].set_xlabel("Time (s)")
axes[1].legend()

fig.suptitle("Unknow 0.5 kg payload on G1 left palm")
plt.tight_layout()
FIG_DIR.mkdir(exist_ok=True)
plt.savefig(FIG_DIR / "payload_adaptation.png")
plt.show()

print("mean error, last 10 s  fixed:", e_fixed[-2000:].mean(), " adaptive:", e_adapt[-2000:].mean())