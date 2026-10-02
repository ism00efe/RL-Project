# Genome v1

Directed graph (Karl Sims 1994 style). Nodes = part types, edges = how a child part attaches to a parent part.
The compiler (`builder/`) unrolls the graph into a tree of parts and emits MJCF. JSON, one file per genome.

```json
{
  "id": "quadruped", "parent_id": null, "generation": 0,
  "root": "torso",
  "nodes": {
    "torso": {"shape": "box", "size": [0.2, 0.08, 0.04]},
    "thigh": {"shape": "capsule", "size": [0.03, 0.08]},
    "seg":   {"shape": "box", "size": [0.06, 0.04, 0.03], "recursion": 6}
  },
  "edges": [
    {"parent": "torso", "child": "thigh", "pos": [0.8, 1, 0], "dir": [0, 0.3, -1], "mirror": ["y"],
     "joint": {"axes": [[0, 0, 1], [0, 1, 0]], "range": [-40, 40], "strength": 1}}
  ]
}
```

## Nodes
- `shape`: `box` | `capsule` | `sphere`. `size` in meters, MuJoCo half-size convention:
  box `[hx, hy, hz]`, capsule `[radius, half_length]`, sphere `[radius]`.
- `recursion` (default 1): max times this node may appear on one root→leaf path. A self-edge (`seg → seg`) with recursion n gives a chain of n parts.
- `density` (default 1000 kg/m³).

## Part frame
Every part has a local frame whose origin is its joint (root: its center). The shape extends along local +x:
capsule from x=0 to x=2·half_length, box/sphere centered at x=hx / x=r. The root's shape is centered on its origin, local x = world forward.
A part's *extent box* = center ± (half extents) of its shape.

## Edges
- `parent`, `child`: node names.
- `pos`: attachment point on the parent, as a fraction of the parent's extent box (`[-1..1]³`, `[1,0,0]` = tip).
- `dir`: direction (parent frame) along which the child extends; it becomes the child's local +x.
  The child's local y is the parent's y made orthogonal to `dir` (parent z if degenerate).
- `mirror` (default `[]`): subset of `"x"`, `"y"`, `"z"`. Each axis adds reflected copies (pos and dir negated on that axis); 2 axes → 4 copies.
- `joint.axes`: 0–2 hinge axes in the child frame (0 = rigid weld). `range`: degrees, symmetric `[lo, hi]` for every axis.
  `strength` (default 1): dimensionless muscle multiplier. Max torque = strength · σ · A · ½√(A/π), σ = 3·10⁵ Pa, A = cross-section of the
  child part perpendicular to its long axis (capsule/sphere πr², box 4·hy·hz), so torque ∝ A^1.5 (Phase 2b.2; before that, strength was gear in N·m).
  The env adds a force–velocity limit: torque in the direction of motion falls linearly to 0 at `max_joint_speed`. Action ∈ [-1, 1].

## Unrolling limits
Parts total ≤ 32. Edge order = attachment order. `id`, `parent_id`, `generation` are carried into recordings (lineage).
