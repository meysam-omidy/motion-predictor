"""
Pack the minimal data train_adaptive_kalman_real.py needs into one zip for Colab.

The real-detection trainer never reads images: per sequence it only needs
  <dataset>/<split>/<seq>/gt/gt.txt
  <dataset>/<split>/<seq>/seqinfo.ini      (image width/height for normalization)
plus one detection file per sequence
  <detector>/<dataset>/<seq>.txt           (frame, x1, y1, x2, y2, score)

Zip layout (what colab_train_real.ipynb expects):
  Datasets/MOT17/train/MOT17-02-FRCNN/gt/gt.txt
  Datasets/MOT17/train/MOT17-02-FRCNN/seqinfo.ini
  Detections/YOLOXx/MOT17/MOT17-02-FRCNN.txt
  ...

Example:
  python pack_for_colab.py --out C:/Projects/mot_colab_data.zip
"""
from __future__ import annotations
import argparse, os, zipfile


def main(args):
    n_seq = n_det = 0
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for ds in args.datasets:
            for split in args.splits:
                split_dir = os.path.join(args.datasets_root, ds, split)
                if not os.path.isdir(split_dir):
                    print(f"  skip {ds}/{split}: not found")
                    continue
                for seq in sorted(os.listdir(split_dir)):
                    seq_dir = os.path.join(split_dir, seq)
                    gt = os.path.join(seq_dir, "gt", "gt.txt")
                    ini = os.path.join(seq_dir, "seqinfo.ini")
                    det = os.path.join(args.det_root, args.detector, ds, seq + ".txt")
                    if not (os.path.isfile(gt) and os.path.isfile(ini)):
                        print(f"  skip {ds}/{split}/{seq}: missing gt.txt or seqinfo.ini")
                        continue
                    # if not os.path.isfile(det):
                    #     print(f"  warn {ds}/{split}/{seq}: no detections ({det}) -> sequence unused")
                    # else:
                    #     zf.write(det, f"Detections/{args.detector}/{ds}/{seq}.txt")
                    #     n_det += 1
                    zf.write(gt, f"Datasets/{ds}/{split}/{seq}/gt/gt.txt")
                    zf.write(ini, f"Datasets/{ds}/{split}/{seq}/seqinfo.ini")
                    n_seq += 1
    print(f"Packed {n_seq} sequences, {n_det} detection files -> {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets_root", type=str, default="C:/Projects/.Datasets")
    p.add_argument("--det_root", type=str, default="C:/Projects/.Detections")
    p.add_argument("--detector", type=str, default="YOLOXx")
    p.add_argument("--datasets", nargs="+", default=["MOT17", "MOT20", "DanceTrack", "SportsMOT"])
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--out", type=str, default="mot_colab_data.zip")
    main(p.parse_args())
