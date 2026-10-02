"""Copy recordings into the viewer (viewer/public/recordings/<genome_id>.json + index.json).
Stdlib only. Usage: python record/collect.py runs/phase2/*/recordings/end.json [--out viewer/public/recordings]"""
import argparse
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recordings", nargs="+")
    ap.add_argument("--out", default="viewer/public/recordings")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    index = []
    for path in args.recordings:
        with open(path) as f:
            meta = json.load(f)["meta"]
        name = f"{meta['genome_id']}.json"
        shutil.copyfile(path, os.path.join(args.out, name))
        index.append({"file": name, "label": f"{meta['genome_id']} · {meta['env']}"})
    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    print(f"{args.out} recordings {len(index)}")


if __name__ == "__main__":
    main()
