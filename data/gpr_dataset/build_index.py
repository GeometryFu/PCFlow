import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--ins_dir", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--ext", default=".png")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    ins_dir = Path(args.ins_dir)
    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_dir.glob(f"*{args.ext}"))
    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir} with ext {args.ext}")

    with open(out, "w", encoding="utf-8") as f:
        for p in image_paths:
            stem = p.stem
            in_path = ins_dir / f"{stem}.in"
            if not in_path.exists():
                continue
            rec = {"id": stem, "image_path": str(p), "in_path": str(in_path)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()