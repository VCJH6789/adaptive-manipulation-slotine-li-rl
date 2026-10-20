import numpy as np
import pinocchio as pin

LEFT_ARM = ["left_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "left_elbow_pitch_joint",
            "left_elbow_roll_joint"]

class PinModel:
    def __init__(self, urdf_path, isaac_joint_names):
        # returns the G1 model from the urdf file
        # by default, the pelvis of G1 is fixed
        self.model = pin.buildModelFromUrdf(urdf_path)
        # storing model result data
        self.data = pin.Data(self.model)

        # For each Isaac joint name, 
        # find the corresponding Pinocchio joint index q

        # position-side indices (for q)
        self.isaac_to_q = np.array([self.model.joints[self.model.getJointId(n)].idx_q
                                    for n in isaac_joint_names])
        self.arm_q = np.array([self.model.joints[self.model.getJointId(n)].idx_q
                               for n in LEFT_ARM])

        # velocity-side indices (for q_dot, q_ddot, torques)
        self.isaac_to_v = np.array([self.model.joints[self.model.getJointId(n)].idx_v
                                    for n in isaac_joint_names])
        self.arm_v = np.array([self.model.joints[self.model.getJointId(n)].idx_v
                               for n in LEFT_ARM])
        
    # convert from Isaac joint order to Pinocchio joint order
    def to_pin_q(self, q_isaac):
        q_pin = pin.neutral(self.model)
        # make sure the input is a numpy array by converting with asarray
        q_pin[self.isaac_to_q] = np.asarray(q_isaac, dtype=float)
        return q_pin

    def to_pin_v(self, v_isaac):
        v_pin = np.zeros(self.model.nv)
        v_pin[self.isaac_to_v] = np.asarray(v_isaac, dtype=float)
        return v_pin

    def gravity_arm(self, q_isaac):
        # convert to Pinocchio joint order
        q_pin = self.to_pin_q(q_isaac)
        # compute gravity torque for the arm joints
        g_arm = pin.computeGeneralizedGravity(
            self.model, self.data, q_pin)[self.arm_v]
        return g_arm


# if __name__ == "__main__":
#     # testing the PinModel class
#     urdf_path = "g1_37dof.urdf"
#     names = list(pin.buildModelFromUrdf(urdf_path).names)[1:]   # 37 names, skipping 'universe'
#     rev = names[::-1]                                           # reversed, like a different simulator order
#     pm = PinModel(urdf_path, rev)
#     print("nq:", pm.model.nq)
#     print("isaac_to_q:", pm.isaac_to_q)    # expect 36, 35, ..., 0
#         print("arm_q:", pm.arm_q, " arm_v:", pm.arm_v)   # expect 5 indices each

#     q = np.zeros(37)
#     for n, v in zip(LEFT_ARM, [0.35, 0.16, 0.0, 0.87, 0.0]):
#         q[rev.index(n)] = v
#     print("g arm:", pm.gravity_arm(q).round(4))