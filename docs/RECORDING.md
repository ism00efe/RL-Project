# Recording format v1

One JSON file per trajectory. Phase-independent: the viewer knows only this format, never MuJoCo models or genomes.
Coordinates: MuJoCo world frame, z up, meters. Quaternions: `[w, x, y, z]`. Floats rounded to 4 decimals.

```json
{
  "format": "evo-recording",
  "version": 1,
  "meta": {
    "genome_id": "go1",            // string id of the body
    "generation": 0,               // int, evolution generation (0 outside evolution)
    "parent_id": null,             // genome_id of parent, null for roots (lineage tree)
    "env": "Go1JoystickFlatTerrain",
    "fitness": 0.93,               // scalar fitness of this trajectory
    "fitness_name": "mean_vx",     // what fitness measures
    "seed": 1,
    "source": "runs/phase1_go1/snapshots/end"
  },
  "dt": 0.02,                      // seconds between frames
  "num_frames": 500,
  "parts": [                       // one per rigid body (world excluded), index = part id
    {
      "name": "trunk",
      "parent": null,              // parent part index, null if attached to world
      "geoms": [                   // primitives in the part's local frame
        {"type": "box", "size": [0.12, 0.04, 0.06], "pos": [0, 0, 0], "quat": [1, 0, 0, 0], "rgba": [0.5, 0.5, 0.5, 1]}
      ]
    }
  ],
  "frames": {                      // [num_frames][num_parts][3|4], world pose of each part
    "pos":  [[[0, 0, 0.27]]],
    "quat": [[[1, 0, 0, 0]]]
  }
}
```

Geom types and `size` follow MuJoCo conventions (half-sizes):
`sphere [r]`, `capsule [r, half_length]` (along local z), `cylinder [r, half_length]` (along local z),
`ellipsoid [rx, ry, rz]`, `box [hx, hy, hz]`.

Only primitive geoms are stored; mesh geoms are skipped (stock Go1 meshes are ~100k vertices; evolved bodies are primitives only).
Ground/terrain is not part of v1; `meta.env` names it.
