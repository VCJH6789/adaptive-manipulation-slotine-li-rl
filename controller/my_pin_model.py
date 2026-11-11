from pathlib import Path
import numpy as np
import pinocchio as pin

REPO_ROOT = Path(__file__).resolve().parents[1]

LEFT_ARM = ["left_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "left_elbow_pitch_joint",
            "left_elbow_roll_joint"]

class PinModel:
    def __init__(self, urdf_path, isaac_joint_names, arm_armature=None):
        # returns the G1 model from the urdf file
        # by default, the pelvis of G1 is fixed
        self.model = pin.buildModelFromUrdf(urdf_path)
        # storing model result data
        self.data = pin.Data(self.model)
        # armature 
        self.arm_armature = np.zeros(len(LEFT_ARM)) if arm_armature is None \
              else np.asarray(arm_armature, float)

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

        # getting the palm frame index
        self.palm_frame_id = self.model.getFrameId("left_palm_link")
        
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
    
    def arm_model_torque(self, q_isaac, v_isaac, v_r_arm, a_r_arm):
        """M(q) a_r + C(q, v) v_r + g(q), for the 5 arm joints.
        q_isaac, v_isaac : all 37 joints, Isaac order
        v_r_arm, a_r_arm : reference velocity / acceleration of the 5 arm joints"""

        # TODO 1: convert q and v to Pinocchio vectors
        q_pin = self.to_pin_q(q_isaac)
        v_pin = self.to_pin_v(v_isaac)

        # TODO 2: build the full reference vectors (length nv)
        #   - velocity: start from the ACTUAL v for all joints, then overwrite the arm entries with v_r_arm
        v_r = v_pin.copy()
        v_r[self.arm_v] = v_r_arm

        #   - acceleration: start from zeros, then put a_r_arm in the arm entries
        a_r = np.zeros(self.model.nv)
        a_r[self.arm_v] = a_r_arm

        # TODO 3: compute M, C and g
        #   M = pin.crba(model, data, q)                     -> mass matrix
        #   C = pin.computeCoriolisMatrix(model, data, q, v) -> uses the ACTUAL velocity
        #   g = pin.computeGeneralizedGravity(model, data, q)
        M = pin.crba(self.model, self.data, q_pin) # only the upper traingle is filled
        M = np.triu(M) + np.triu(M, 1).T  # make symmetric
        M[self.arm_v, self.arm_v] += self.arm_armature  # add armature to the arm joints
        C = pin.computeCoriolisMatrix(self.model, self.data, q_pin, v_pin)
        g = pin.computeGeneralizedGravity(self.model, self.data, q_pin)

        # TODO 4: tau = M @ a_r + C @ v_r + g, and return the arm entries
        tau = M @ a_r + C @ v_r + g
        return tau[self.arm_v]

    def palm_regressor(self, q_pin, v_pin, a_pin, gravity=True):
        """Y (nv x 10) such that the payload torque = Y @ theta,
        for a payload rigidly attached to the palm frame.
        Standard inverse-dynamics form: uses the given v and a for everything."""

        # forward kinematics to get the palm frame placement
        pin.forwardKinematics(self.model, self.data, q_pin, v_pin, a_pin)
        pin.updateFramePlacements(self.model, self.data)

        # getting the palm frame velocity and acceleration expressed in local coordinates
        v_palm = pin.getFrameVelocity(self.model, self.data, self.palm_frame_id, pin.LOCAL)
        a_palm = pin.getFrameAcceleration(self.model, self.data, self.palm_frame_id, pin.LOCAL)

        if gravity:
            # add gravity to the palm frame acceleration (in local coordinates)
            R = self.data.oMf[self.palm_frame_id].rotation # rotation matrix from palm to world
            a_palm = pin.Motion(a_palm.linear - R.T @ self.model.gravity.linear, a_palm.angular)

        # compute the regressor A 6 x nv for the palm frame
        A = pin.bodyRegressor(v_palm, a_palm)

        # map the regressor A wrench to the joint torques using the palm frame Jacobian
        J = pin.computeFrameJacobian(self.model, self.data, q_pin, self.palm_frame_id, pin.LOCAL) # 6 x nv
        return J.T @ A # J.T @ A = Y (nv x 10)

    def object_regressor_arm(self, q_isaac, v_isaac, v_r_arm, a_r_arm):
        """Slotine-Li payload regressor for the 5 arm joints: (5 x 10).
        Y @ theta = M_obj a_r + C_obj(q, v) v_r + g_obj, arm rows only."""

        # convert to Pinocchio joint order
        q_pin = self.to_pin_q(q_isaac) # current joint pose
        v_pin = self.to_pin_q(v_isaac) # current joint velocity

        # full vector joints but replaced the arm entries with the reference values
        v_r = v_pin.copy()
        v_r[self.arm_v] = v_r_arm       # reference velocity
        a_r = np.zeros(self.model.nv)
        a_r[self.arm_v] = a_r_arm       # reference acceleration

        # Polarization Identity Trick: as object mass is unknown
        zeros = np.zeros(self.model.nv) # to zero the Colioris term
        M_g_term = self.palm_regressor(q_pin, zeros, a_r, gravity=True) # Mass inertia + Gravity term
        C1_term = self.palm_regressor(q_pin, v_pin + v_r, zeros, gravity=False) # Coriolis term 1
        C2_term = self.palm_regressor(q_pin, v_pin - v_r, zeros, gravity=False) # Coriolis term 2

        # The final regressor is the sum of the three terms (M term will cancel out since it is needed in C term)
        Y = M_g_term + 0.25 * (C1_term - C2_term)

        # return the arm rows only
        return Y[self.arm_v, :] # 5 x 10 shape


if __name__ == "__main__":
    # testing the PinModel class
    urdf_path = str(REPO_ROOT / "assets" / "g1_37dof.urdf")
    names = list(pin.buildModelFromUrdf(urdf_path).names)[1:]   # 37 names, skipping 'universe'
    rev = names[::-1]                                           # reversed, like a different simulator order
    pm = PinModel(urdf_path, rev)

    rng = np.random.default_rng(1)
    qp = rng.uniform(-0.5, 0.5, pm.model.nq)
    vp = rng.uniform(-1, 1, pm.model.nv)
    ap = rng.uniform(-1, 1, pm.model.nv)

    payload = pin.Inertia(0.7, np.array([0.05, -0.02, 0.03]), np.diag([0.002, 0.003, 0.0015]))
    theta = payload.toDynamicParameters()

    m2 = pm.model.copy()
    jid = pm.model.frames[pm.palm_frame_id].parentJoint
    m2.inertias[jid] = m2.inertias[jid] + pm.model.frames[pm.palm_frame_id].placement.act(payload)
    d2 = m2.createData()

    arm_v = np.array([pm.model.joints[pm.model.getJointId(n)].idx_v
                                   for n in LEFT_ARM])

    true_extra = pin.rnea(m2, d2, qp, vp, ap) - pin.rnea(pm.model, pm.data, qp, vp, ap)
    print("max diff:", np.abs(pm.palm_regressor(qp, vp, ap) @ theta - true_extra).max())   # expect ~1e-15

    # --- Part 2: Slotine-Li object regressor check ---
    def sl_torque(m, d, q_pin, v_pin, vr, ar):
        """M(q) a_r + C(q, v) v_r + g(q), computed directly by Pinocchio."""
        M = pin.crba(m, d, q_pin); M = np.triu(M) + np.triu(M, 1).T
        return M @ ar + pin.computeCoriolisMatrix(m, d, q_pin, v_pin) @ vr + pin.computeGeneralizedGravity(m, d, q_pin)

    # Inputs in ISAAC order (here: rev), which is what object_regressor_arm expects
    q_is   = rng.uniform(-0.5, 0.5, 37)
    v_is   = rng.uniform(-1, 1, 37)
    vr_arm = rng.uniform(-1, 1, 5)      # reference velocity, LEFT_ARM order
    ar_arm = rng.uniform(-1, 1, 5)      # reference acceleration, LEFT_ARM order

    # The same quantities in PINOCCHIO order, for the ground truth
    q_p, v_p = pm.to_pin_q(q_is), pm.to_pin_v(v_is)
    vr = v_p.copy();              vr[pm.arm_v] = vr_arm     # actual v elsewhere, reference on the arm
    ar = np.zeros(pm.model.nv);   ar[pm.arm_v] = ar_arm     # zero elsewhere, reference on the arm

    # Ground truth: Slotine-Li torque with the payload, minus without it, arm rows only
    true_extra_sl = (sl_torque(m2, d2, q_p, v_p, vr, ar)
                     - sl_torque(pm.model, pm.data, q_p, v_p, vr, ar))[pm.arm_v]

    Y_arm = pm.object_regressor_arm(q_is, v_is, vr_arm, ar_arm)
    print("Y_arm shape:", Y_arm.shape)                                          # expect (5, 10)
    print("Slotine-Li max diff:", np.abs(Y_arm @ theta - true_extra_sl).max())  # expect ~1e-15

#     print("nq:", pm.model.nq)
#     print("isaac_to_q:", pm.isaac_to_q)    # expect 36, 35, ..., 0
#     print("arm_q:", pm.arm_q, " arm_v:", pm.arm_v)   # expect 5 indices each

#     q = np.zeros(37)
#     for n, v in zip(LEFT_ARM, [0.35, 0.16, 0.0, 0.87, 0.0]):
#         q[rev.index(n)] = v
#     print("g arm:", pm.gravity_arm(q).round(4))

#     rng = np.random.default_rng(0)
#     q_rand  = rng.uniform(-0.5, 0.5, 37)      # Isaac order (rev)
#     v_rand  = rng.uniform(-1.0, 1.0, 37)
#     a_arm   = rng.uniform(-1.0, 1.0, 5)
#     arm_in_isaac = [rev.index(n) for n in LEFT_ARM]

#     # Using the ACTUAL arm velocity as the "reference" must reproduce full inverse dynamics
#     tau_mine = pm.arm_model_torque(q_rand, v_rand, v_rand[arm_in_isaac], a_arm)

#     a_full = np.zeros(pm.model.nv)
#     a_full[pm.arm_v] = a_arm
#     tau_rnea = pin.rnea(pm.model, pm.data,
#                         pm.to_pin_q(q_rand), pm.to_pin_v(v_rand), a_full)[pm.arm_v]

#     print("max diff vs rnea:", np.abs(tau_mine - tau_rnea).max())   # expect ~1e-12