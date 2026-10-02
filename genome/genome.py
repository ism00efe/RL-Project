"""Genome v1 (docs/GENOME.md): load, validate, save. Genomes are plain dicts with defaults filled in."""
import json

SHAPES = {"box": 3, "capsule": 2, "sphere": 1}
NODE_DEFAULTS = {"recursion": 1, "density": 1000.0}
EDGE_DEFAULTS = {"mirror": []}
JOINT_DEFAULTS = {"axes": [], "range": [-45.0, 45.0], "strength": 1.0}
MAX_PARTS = 32


def normalize(g):
    """Fill defaults and validate; returns a new dict."""
    g = json.loads(json.dumps(g))
    g.setdefault("parent_id", None)
    g.setdefault("generation", 0)
    for name, n in g["nodes"].items():
        for k, v in NODE_DEFAULTS.items():
            n.setdefault(k, v)
        if n["shape"] not in SHAPES or len(n["size"]) != SHAPES[n["shape"]]:
            raise ValueError(f"node {name}: bad shape/size {n['shape']} {n['size']}")
        if min(n["size"]) <= 0 or n["recursion"] < 1:
            raise ValueError(f"node {name}: sizes must be > 0, recursion >= 1")
    if g["root"] not in g["nodes"]:
        raise ValueError(f"root {g['root']} not a node")
    for i, e in enumerate(g["edges"]):
        for k, v in EDGE_DEFAULTS.items():
            e.setdefault(k, v)
        e["joint"] = {**JOINT_DEFAULTS, **e.get("joint", {})}
        if e["parent"] not in g["nodes"] or e["child"] not in g["nodes"]:
            raise ValueError(f"edge {i}: unknown node")
        if len(e["pos"]) != 3 or len(e["dir"]) != 3 or not any(e["dir"]):
            raise ValueError(f"edge {i}: pos/dir must be 3-vectors, dir nonzero")
        if not set(e["mirror"]) <= {"x", "y", "z"} or len(e["joint"]["axes"]) > 2:
            raise ValueError(f"edge {i}: bad mirror or > 2 joint axes")
    return g


def load(path):
    with open(path) as f:
        return normalize(json.load(f))


def save(g, path):
    with open(path, "w") as f:
        json.dump(g, f, indent=1)
