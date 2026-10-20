"""Convert the Isaac Lab G1 USD (37-DOF, 5-DOF arms) into a dynamics-only URDF.

Usage:  python usd_to_urdf.py g1.usd g1_37dof.urdf

Each URDF link frame is placed at its parent joint frame (USD localPos1/localRot1
in the child body), so joint axes are expressed in the child link frame as URDF
expects. Inertials are re-expressed in that frame. No meshes are written.
"""
import sys
import math
import numpy as np
from pxr import Usd, UsdPhysics, Gf

AXIS = {"X": "1 0 0", "Y": "0 1 0", "Z": "0 0 1"}


def quat_to_R(q):
    w, (x, y, z) = q.GetReal(), q.GetImaginary()
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:  # unset quaternion -> identity
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def T(pos, quat):
    M = np.eye(4)
    M[:3, :3] = quat_to_R(quat)
    M[:3, 3] = list(pos)
    return M


def inv(M):
    Mi = np.eye(4)
    Mi[:3, :3] = M[:3, :3].T
    Mi[:3, 3] = -M[:3, :3].T @ M[:3, 3]
    return Mi


def rpy(R):
    # URDF convention: R = Rz(yaw) Ry(pitch) Rx(roll)
    pitch = math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))
    if abs(math.cos(pitch)) < 1e-9:
        roll, yaw = 0.0, math.atan2(-R[0, 1], R[1, 1])
    else:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def origin(M):
    x, y, z = M[:3, 3]
    r, p, yw = rpy(M[:3, :3])
    return f'<origin xyz="{x:.9g} {y:.9g} {z:.9g}" rpy="{r:.9g} {p:.9g} {yw:.9g}"/>'


def main(usd_path, urdf_path):
    stage = Usd.Stage.Open(usd_path)

    bodies, joints = {}, []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies[prim.GetName()] = prim
        elif prim.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(prim)
            b0 = j.GetBody0Rel().GetTargets()
            b1 = j.GetBody1Rel().GetTargets()
            if not b0 or not b1:
                continue
            joints.append(dict(
                name=prim.GetName(), prim=prim,
                parent=b0[0].name, child=b1[0].name,
                T0=T(j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get()),
                T1=T(j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get()),
            ))

    # Link frame F_L, expressed in USD body coordinates (root: identity)
    F = {name: np.eye(4) for name in bodies}
    for j in joints:
        F[j["child"]] = j["T1"]

    out = ['<?xml version="1.0"?>', '<robot name="g1_37dof">']

    for name, prim in bodies.items():
        m = prim.GetAttribute("physics:mass").Get()
        com = prim.GetAttribute("physics:centerOfMass").Get()
        diag = prim.GetAttribute("physics:diagonalInertia").Get()
        axes = prim.GetAttribute("physics:principalAxes").Get()
        if not all(math.isfinite(c) for c in com):  # "unset" CoM on dummy links
            com = (0.0, 0.0, 0.0)
        Tin = inv(F[name]) @ T(com, axes)  # inertia frame in link frame
        out += [
            f'  <link name="{name}">',
            '    <inertial>',
            f'      {origin(Tin)}',
            f'      <mass value="{m:.9g}"/>',
            f'      <inertia ixx="{diag[0]:.9g}" ixy="0" ixz="0" '
            f'iyy="{diag[1]:.9g}" iyz="0" izz="{diag[2]:.9g}"/>',
            '    </inertial>',
            '  </link>',
        ]

    for j in joints:
        prim = j["prim"]
        Tj = inv(F[j["parent"]]) @ j["T0"]  # joint frame in parent link frame
        if prim.IsA(UsdPhysics.RevoluteJoint):
            rj = UsdPhysics.RevoluteJoint(prim)
            lo = math.radians(rj.GetLowerLimitAttr().Get())
            hi = math.radians(rj.GetUpperLimitAttr().Get())
            eff = prim.GetAttribute("drive:angular:physics:maxForce").Get() or 0.0
            vel = prim.GetAttribute("physxJoint:maxJointVelocity").Get()
            vel = math.radians(vel) if vel else 0.0
            out += [
                f'  <joint name="{j["name"]}" type="revolute">',
                f'    {origin(Tj)}',
                f'    <parent link="{j["parent"]}"/>',
                f'    <child link="{j["child"]}"/>',
                f'    <axis xyz="{AXIS[rj.GetAxisAttr().Get()]}"/>',
                f'    <limit lower="{lo:.9g}" upper="{hi:.9g}" effort="{eff:.9g}" velocity="{vel:.9g}"/>',
                '  </joint>',
            ]
        else:
            out += [
                f'  <joint name="{j["name"]}" type="fixed">',
                f'    {origin(Tj)}',
                f'    <parent link="{j["parent"]}"/>',
                f'    <child link="{j["child"]}"/>',
                '  </joint>',
            ]

    out.append('</robot>')
    with open(urdf_path, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {urdf_path}: {len(bodies)} links, {len(joints)} joints")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])