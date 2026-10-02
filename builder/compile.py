"""Genome v1 -> MJCF (docs/GENOME.md). Usage: python -m builder.compile [--torques] genome/examples/*.json"""
import itertools
import sys

import mujoco
import numpy as np

from genome.genome import MAX_PARTS, load

SPAWN_CLEARANCE = 0.02  # m between lowest geom and floor at reset
ROOT_RGBA, PART_RGBA = "0.85 0.45 0.2 1", "0.35 0.5 0.7 1"
MUSCLE_STRESS = 3e5  # Pa, vertebrate muscle max isometric stress (~30 N/cm²)


def cross_section(node):
    """Area (m²) of a node's shape perpendicular to its long axis (local x)."""
    s = node["size"]
    if node["shape"] == "box":
        return 4.0 * s[1] * s[2]
    return np.pi * s[0] ** 2


def max_torque(node, strength):
    """Muscle model: force = stress * A (whole cross-section is muscle), moment arm = half the
    equivalent radius sqrt(A/pi). Torque ~ A^1.5 ~ L^3 (square-cube law vs weight*lever ~ L^4).
    `strength` is the genome's dimensionless multiplier."""
    a = cross_section(node)
    return strength * MUSCLE_STRESS * a * 0.5 * np.sqrt(a / np.pi)

HEADER = """<mujoco model="{id}">
  <compiler angle="degree"/>
  <option timestep="0.004" iterations="4" ls_iterations="8"><flag eulerdamp="disable"/></option>
  <default>
    <joint damping="0.5" armature="0.01"/>
    <geom contype="0" conaffinity="1" condim="3" friction="1 0.005 0.0001"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" contype="1" conaffinity="0" rgba="0.8 0.8 0.8 1"/>
"""


def shape_extent(node, is_root):
    """(center, half_extents) of a node's shape in its part frame."""
    s = node["size"]
    if node["shape"] == "box":
        h = np.array(s, float)
    elif node["shape"] == "capsule":
        h = np.array([s[1] + s[0], s[0], s[0]])
    else:
        h = np.full(3, s[0])
    reach = s[1] if node["shape"] == "capsule" else h[0]
    return (np.zeros(3) if is_root else np.array([reach, 0.0, 0.0])), h


def frame_from_dir(d):
    """Rotation (columns = child axes in parent frame): x along d, y = parent y made orthogonal."""
    x = np.asarray(d, float) / np.linalg.norm(d)
    for ref in (np.array([0.0, 1, 0]), np.array([0.0, 0, 1])):
        y = ref - ref.dot(x) * x
        if np.linalg.norm(y) > 1e-6:
            break
    y /= np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def unroll(g):
    """Graph -> list of parts {name, node, parent, pos, rot, joint} (parent index, -1 for root)."""
    parts = []

    def add(node, parent, pos, rot, joint, counts):
        if len(parts) >= MAX_PARTS:
            raise ValueError(f"genome {g['id']} unrolls to > {MAX_PARTS} parts")
        idx = len(parts)
        parts.append({"name": f"{node}{idx}", "node": node, "parent": parent,
                      "pos": pos, "rot": rot, "joint": joint})
        counts = {**counts, node: counts.get(node, 0) + 1}
        center, half = shape_extent(g["nodes"][node], idx == 0)
        for e in g["edges"]:
            child = e["child"]
            if e["parent"] != node or counts.get(child, 0) >= g["nodes"][child]["recursion"]:
                continue
            axes = ["xyz".index(a) for a in e["mirror"]]
            for flips in itertools.product((1.0, -1.0), repeat=len(axes)):
                s = np.ones(3)
                for a, f in zip(axes, flips):
                    s[a] = f
                p = center + np.asarray(e["pos"], float) * s * half
                add(child, idx, p, frame_from_dir(np.asarray(e["dir"], float) * s), e["joint"], counts)

    add(g["root"], -1, np.zeros(3), np.eye(3), None, {})
    return parts


def fmt(v):
    return " ".join(f"{x:.5g}" for x in np.ravel(v))


def geom_xml(node, is_root, rgba):
    s, kind = node["size"], node["shape"]
    center, _ = shape_extent(node, is_root)
    common = f'density="{node["density"]:g}" rgba="{rgba}"'
    if kind == "capsule":
        a, b = (-s[1], s[1]) if is_root else (0.0, 2 * s[1])
        return f'<geom type="capsule" size="{s[0]:g}" fromto="{a:g} 0 0 {b:g} 0 0" {common}/>'
    return f'<geom type="{kind}" size="{fmt(s)}" pos="{fmt(center)}" {common}/>'


def to_mjcf(g, root_z=0.0):
    parts = unroll(g)
    children = {i: [j for j, p in enumerate(parts) if p["parent"] == i] for i in range(len(parts))}
    joints = []

    def body(i, depth):
        p, ind = parts[i], "  " * (depth + 2)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, p["rot"].flatten())
        pos = np.array([0, 0, root_z]) if i == 0 else p["pos"]
        lines = [f'{ind}<body name="{p["name"]}" pos="{fmt(pos)}" quat="{fmt(quat)}">']
        if i == 0:
            lines.append(f'{ind}  <freejoint name="root"/>')
            lines.append(f'{ind}  <camera name="track" mode="trackcom" pos="0 -2.5 1.2" xyaxes="1 0 0 0 0.43 0.9"/>')
        else:
            for k, axis in enumerate(p["joint"]["axes"]):
                name = f'{p["name"]}_j{k}'
                lo, hi = p["joint"]["range"]
                lines.append(f'{ind}  <joint name="{name}" type="hinge" axis="{fmt(axis)}" range="{lo:g} {hi:g}"/>')
                joints.append((name, max_torque(g["nodes"][p["node"]], p["joint"]["strength"])))
        lines.append(f'{ind}  ' + geom_xml(g["nodes"][p["node"]], i == 0, ROOT_RGBA if i == 0 else PART_RGBA))
        for c in children[i]:
            lines += body(c, depth + 1)
        lines.append(f"{ind}</body>")
        return lines

    bodies = body(0, 0)
    acts = [f'    <motor name="{n}" joint="{n}" gear="{s:.4g}"/>' for n, s in joints]
    return (HEADER.format(id=g["id"]) + "\n".join(bodies) + "\n  </worldbody>\n  <actuator>\n"
            + "\n".join(acts) + "\n  </actuator>\n</mujoco>\n")


def lowest_point(m):
    """Min world z over all non-floor geoms at qpos0 (from each geom's local AABB)."""
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    low = np.inf
    for gi in range(m.ngeom):
        if m.geom_bodyid[gi] == 0:
            continue
        R = d.geom_xmat[gi].reshape(3, 3)
        c, h = m.geom_aabb[gi][:3], m.geom_aabb[gi][3:]
        low = min(low, d.geom_xpos[gi][2] + R[2] @ c - np.abs(R[2]) @ h)
    return low


def compile_genome(g):
    """Returns (MJCF string, MjModel) with the root spawned just above the floor."""
    m = mujoco.MjModel.from_xml_string(to_mjcf(g))
    xml = to_mjcf(g, root_z=SPAWN_CLEARANCE - lowest_point(m))
    return xml, mujoco.MjModel.from_xml_string(xml)


def main(args):
    flags = [a for a in args if a.startswith("--")]
    for path in (a for a in args if not a.startswith("--")):
        g = load(path)
        _, m = compile_genome(g)
        print(f"{g['id']} parts {m.nbody - 1} joints {m.njnt - 1} actuators {m.nu} mass {m.body_subtreemass[1]:.1f}kg")
        if "--torques" in flags:
            torques = {}
            for i in range(m.nu):
                part = m.body(m.jnt_bodyid[m.actuator_trnid[i, 0]]).name.rstrip("0123456789")
                torques.setdefault(part, round(float(m.actuator_gear[i, 0]), 1))
            print(f"  max torque N·m per part type: {torques}")


if __name__ == "__main__":
    main(sys.argv[1:])
