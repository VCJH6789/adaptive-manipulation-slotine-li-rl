import numpy as np

class SlotineLiController:
    """Tracking controller for the 5 arm joints based on the 
    Slotine-Li adaptive control law for robotic manipulators."""

    def __init__(self, pin_model, lam, K, gamma, adapt=True):
        self.pin = pin_model
        n = len(self.pin.arm_v)  # number of arm joints
        self.lam = np.diag(np.broadcast_to(np.asarray(lam, dtype=float), (n,)))  # diagonal matrix
        self.K = np.diag(np.broadcast_to(np.asarray(K, dtype=float), (n,)))  # diagonal matrix

        # adaptive control parameters
        self.adapt = adapt
        self.gamma = np.diag(np.broadcast_to(np.asarray(gamma, dtype=float), (10,)))
        self.theta_hat = np.zeros(10) # (10 x 1) shape || assuming no object initially


    def compute_arm_torque(self, q_isaac, v_isaac, arm_in_isaac, q_des, v_des, a_des, dt):
        """q_isaac, v_isaac : all 37 joints, Isaac order (actual state)
        arm_in_isaac       : position index of the 5 arm joints in the Isaac vectors
        q_des, v_des, a_des: desired arm trajectory at this instant (5 each)
        Returns tau (5,) for the arm joints, and s (5,) for logging."""
        # check for numpy type
        q_isaac, v_isaac = np.asarray(q_isaac, float), np.asarray(v_isaac, float)
        q_des, v_des, a_des = (np.asarray(x, float) for x in (q_des, v_des, a_des))

        # actual arm position and velocity (pick the 5 arm entries)
        q_arm = q_isaac[arm_in_isaac]
        v_arm = v_isaac[arm_in_isaac]

        # errors, reference velocity / acceleration, sliding variable
        q_err = q_arm - q_des
        v_err = v_arm - v_des
        v_r   = v_des - self.lam @ q_err
        a_r   = a_des - self.lam @ v_err
        s     = v_arm - v_r

        # the control law 
        tau_arm = self.pin.arm_model_torque(q_isaac, v_isaac, v_r, a_r)  # your pin model's arm model torque, with v_r and a_r

        # object regressor Y parameter
        Y = self.pin.object_regressor_arm(q_isaac, v_isaac, v_r, a_r)  # (5 x 10)

        # compute the control law --> Tau = Y @ self.theta_hat - self.K @ s
        tau = tau_arm + Y @ self.theta_hat - self.K @ s

        # compute the adaptation law
        if self.adapt:
            self.theta_hat += dt * (-self.gamma @ Y.T @ s) # one time step of Euler integration

        return tau, s

    def reset_adaptation(self):
        """Reset the estimated object parameters to zero."""
        self.theta_hat = np.zeros(10)


# if __name__ == "__main__":
#     import pinocchio as pin
#     from controller.my_pin_model import PinModel, LEFT_ARM, REPO_ROOT

#     urdf_path = str(REPO_ROOT / "assets" / "g1_37dof.urdf")
#     names = list(pin.buildModelFromUrdf(urdf_path).names)[1:]
#     pm = PinModel(urdf_path, names)
#     arm_in_isaac = [names.index(n) for n in LEFT_ARM]
#     ctrl = SlotineLiController(pm, lam=10, K=[5, 5, 3, 3, 1])

#     q = np.zeros(37); v = np.zeros(37)
#     q[arm_in_isaac] = [0.35, 0.16, 0.0, 0.87, 0.0]
#     tau, s = ctrl.compute_arm_torque(q, v, arm_in_isaac,
#                                      q_des=q[arm_in_isaac], v_des=np.zeros(5), a_des=np.zeros(5))
#     print("s:", s)                  # expect [0 0 0 0 0]
#     print("tau:", tau.round(4))     # expect [0.6984 1.1115 0.2348 -0.7806 -0.0107]