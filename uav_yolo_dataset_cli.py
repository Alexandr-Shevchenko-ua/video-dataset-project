#!/usr/bin/env python
import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import json
import random
import shutil
from ultralytics import YOLO
import yaml
from collections import defaultdict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ----------------- ARGPARSE -----------------

def parse_args():
    p = argparse.ArgumentParser(
        description="UAV video -> YOLOv11 dataset (images + YOLO TXT labels)."
    )
    p.add_argument("--videos-root", type=str, required=True,
                   help="Root directory with UAV videos.")
    p.add_argument("--output-root", type=str, required=True,
                   help="Output dataset root (will contain images/, labels/, data.yaml).")
    p.add_argument("--weights", type=str, required=False,
                   help="Path to YOLOv11 weights for pre-label.")
    p.add_argument("--no-prelabel", action="store_true",
                   help="If set, do NOT run YOLO prelabel (images only).")

    # frame sampling / quality
    p.add_argument("--fps-target", type=float, default=3.0,
                   help="Target frames per second to sample from videos.")
    p.add_argument("--min-ssim", type=float, default=0.97,
                   help="Threshold for SSIM-like near-duplicate filtering (higher => fewer frames).")
    p.add_argument("--min-blur-var", type=float, default=50.0,
                   help="Min Laplacian variance; below => too blurry -> drop.")
    p.add_argument("--min-brightness", type=float, default=20.0,
                   help="Min average brightness (0-255). Below => drop.")
    p.add_argument("--max-frames-per-video", type=int, default=400,
                   help="Safety cap on frames extracted per video.")

    # YOLO inference
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size for YOLO.")
    p.add_argument("--conf-thres", type=float, default=0.25, help="YOLO confidence threshold.")
    p.add_argument("--iou-thres", type=float, default=0.45, help="YOLO NMS IoU threshold.")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="YOLO device, e.g. 'cuda:0' or 'cpu'.")

    # split
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--split-by-folder", action="store_true",
                   help="Group videos by their parent folder when splitting train/val/test.")

    # class names (single-class by default)
    p.add_argument(
        "--names",
        type=str,
        default='{"0":"heavy_vehicle"}',
        help='JSON dict of class_id->name, e.g. \'{"0":"heavy_vehicle"}\'',
    )
    p.add_argument("--force-single-class", action="store_true",
                   help="Force all detections to class 0 (single-class dataset).")

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ----------------- QUALITY METRICS -----------------

def laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def avg_brightness(gray: np.ndarray) -> float:
    return float(gray.mean())


def ssim_simple(a: np.ndarray, b: np.ndarray) -> float:
    """Very simple SSIM-like metric for near-duplicate filtering."""
    a = a.astype(np.float32) / 255.0
    b = b.astype(np.float32) / 255.0
    mu_a = a.mean()
    mu_b = b.mean()
    sigma_a = a.var()
    sigma_b = b.var()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    C1, C2 = 0.01**2, 0.03**2
    num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    den = (mu_a**2 + mu_b**2 + C1) * (sigma_a + sigma_b + C2)
    return float(num / (den + 1e-8))


# ----------------- LIST VIDEOS -----------------

def list_videos(root: Path):
    VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
    for p in root.rglob("*"):
        if p.suffix.lower() in VIDEO_EXTS:
            yield p, str(p.relative_to(root))


# ----------------- FRAME EXTRACTION -----------------

def extract_frames_from_video(
    video_path: Path,
    video_id: str,
    out_dir: Path,
    fps_target: float,
    min_ssim: float,
    min_blur_var: float,
    min_brightness: float,
    max_frames_per_video: int,
):
    """Returns list of (frame_path, frame_idx)."""

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(int(round(fps / fps_target)), 1)

    saved = []
    frame_idx = 0
    last_frame_small = None
    basename = Path(video_id).stem  # використовуємо відносний id, але stem як ім'я

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # quality filters
        blur_v = laplacian_variance(gray)
        if blur_v < min_blur_var:
            frame_idx += 1
            continue

        bright = avg_brightness(gray)
        if bright < min_brightness:
            frame_idx += 1
            continue

        # downscale for SSIM
        h, w = gray.shape
        scale = 256.0 / max(h, w)
        small = cv2.resize(gray, (int(w * scale), int(h * scale)))

        if last_frame_small is not None:
            ssim_val = ssim_simple(small, last_frame_small)
            if ssim_val > min_ssim:
                frame_idx += 1
                continue

        frame_name = f"{basename}_f{frame_idx:06d}.jpg"
        out_path = out_dir / frame_name
        cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        saved.append((out_path, frame_idx))
        last_frame_small = small

        frame_idx += 1
        if len(saved) >= max_frames_per_video:
            break

    cap.release()
    return saved


# ----------------- YOLO PRELABEL + META -----------------

def run_yolo_prelabel(
    model: YOLO,
    samples,  # list of (frame_path, frame_idx)
    labels_dir: Path,
    meta_fp,
    video_id: str,
    imgsz: int,
    conf_thres: float,
    iou_thres: float,
    force_single_class: bool = True,
    device: str = "cuda:0",
):
    labels_dir.mkdir(parents=True, exist_ok=True)
    for frame_path, frame_idx in tqdm(samples, desc=f"YOLO prelabel [{video_id}]", leave=False):
        results = model.predict(
            source=str(frame_path),
            imgsz=imgsz,
            conf=conf_thres,
            iou=iou_thres,
            verbose=False,
            device=device,
        )
        result = results[0]
        boxes = result.boxes

        # META: якщо боксів нема -> записуємо нуль
        if boxes is None or len(boxes) == 0:
            rec = {
                "video_id": video_id,
                "frame_file": frame_path.name,
                "frame_idx": frame_idx,
                "num_boxes": 0,
            }
            meta_fp.write(json.dumps(rec) + "\n")
            continue

        label_lines = []
        areas = []
        for b in boxes:
            cls_id = 0 if force_single_class else int(b.cls.item())
            x1, y1, x2, y2 = b.xyxyn[0].tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            ww = (x2 - x1)
            hh = (y2 - y1)
            label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")
            areas.append(ww * hh)

        lbl_path = labels_dir / (frame_path.stem + ".txt")
        with open(lbl_path, "w") as f:
            f.write("\n".join(label_lines) + "\n")

        rec = {
            "video_id": video_id,
            "frame_file": frame_path.name,
            "frame_idx": frame_idx,
            "num_boxes": len(areas),
            "area_frac_min": float(min(areas)),
            "area_frac_max": float(max(areas)),
            "area_frac_mean": float(sum(areas) / len(areas)),
        }
        meta_fp.write(json.dumps(rec) + "\n")


# ----------------- SPLIT -----------------

def make_video_level_split_grouped(video_ids, train_ratio, val_ratio, test_ratio, seed=42, by_folder=False):
    rng = random.Random(seed)

    if not by_folder:
        vids = list(video_ids)
        rng.shuffle(vids)
        n = len(vids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_videos = set(vids[:n_train])
        val_videos = set(vids[n_train:n_train + n_val])
        test_videos = set(vids[n_train + n_val:])
        return train_videos, val_videos, test_videos

    # group by parent folder (relative parent of video_id)
    folder_to_videos = defaultdict(list)
    for vid in video_ids:
        parent = str(Path(vid).parent)
        folder_to_videos[parent].append(vid)

    folders = list(folder_to_videos.keys())
    rng.shuffle(folders)
    n_f = len(folders)
    n_train = int(n_f * train_ratio)
    n_val = int(n_f * val_ratio)
    train_folders = set(folders[:n_train])
    val_folders = set(folders[n_train:n_train + n_val])
    test_folders = set(folders[n_train + n_val:])

    train_videos, val_videos, test_videos = set(), set(), set()
    for folder, vids in folder_to_videos.items():
        if folder in train_folders:
            train_videos.update(vids)
        elif folder in val_folders:
            val_videos.update(vids)
        else:
            test_videos.update(vids)
    return train_videos, val_videos, test_videos


def move_to_final_layout(
    video_to_frames,
    frames_root: Path,
    labels_root: Path,
    out_root: Path,
    train_videos,
    val_videos,
    test_videos,
):
    # Гарантуємо наявність фінальних директорій
    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    skipped_missing = 0
    moved_images = 0
    moved_labels = 0

    for video_id, samples in video_to_frames.items():
        if video_id in train_videos:
            split = "train"
        elif video_id in val_videos:
            split = "val"
        else:
            split = "test"

        for frame_path, frame_idx in samples:
            src_img = frame_path
            if not src_img.exists():
                skipped_missing += 1
                continue

            src_lbl = labels_root / (frame_path.stem + ".txt")
            dst_img = out_root / "images" / split / src_img.name
            dst_lbl = out_root / "labels" / split / (frame_path.stem + ".txt")

            shutil.move(str(src_img), str(dst_img))
            moved_images += 1

            if src_lbl.exists():
                shutil.move(str(src_lbl), str(dst_lbl))
                moved_labels += 1

    print(f"[MOVE] Moved images: {moved_images}, labels: {moved_labels}, "
          f"skipped missing frames: {skipped_missing}")


# ----------------- DATA.YAML -----------------

def write_data_yaml(out_root: Path, names_dict: dict):
    yaml_path = out_root / "data.yaml"
    y = {
        "path": str(out_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {int(k): v for k, v in names_dict.items()},
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(y, f, sort_keys=False)
    print(f"[INFO] Wrote data.yaml -> {yaml_path}")


# ----------------- MAIN -----------------

def main():
    args = parse_args()
    random.seed(args.seed)

    videos_root = Path(args.videos_root)
    out_root = Path(args.output_root)
    tmp_frames_root = out_root / "_tmp_frames"
    tmp_labels_root = out_root / "_tmp_labels"

    out_root.mkdir(parents=True, exist_ok=True)
    tmp_frames_root.mkdir(parents=True, exist_ok=True)
    tmp_labels_root.mkdir(parents=True, exist_ok=True)

    names_dict = json.loads(args.names)

    # meta file
    meta_dir = out_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_fp = open(meta_dir / "frames.jsonl", "w")

    # YOLO model (optional)
    model = None
    if not args.no_prelabel:
        if not args.weights:
            raise ValueError("Prelabel is enabled but --weights is not provided.")
        print(f"[INFO] Loading YOLOv11 model from {args.weights} ...")
        model = YOLO(args.weights)
        model.to(args.device)
    else:
        print("[INFO] Running in --no-prelabel mode (images only).")

    video_to_frames = defaultdict(list)
    total_frames_extracted = 0

    videos = list(list_videos(videos_root))
    print(f"[INFO] Found {len(videos)} videos under {videos_root}")

    for vid_path, video_id in tqdm(videos, desc="Processing videos"):
        frames_dir = tmp_frames_root / Path(video_id).stem
        samples = extract_frames_from_video(
            video_path=vid_path,
            video_id=video_id,
            out_dir=frames_dir,
            fps_target=args.fps_target,
            min_ssim=args.min_ssim,
            min_blur_var=args.min_blur_var,
            min_brightness=args.min_brightness,
            max_frames_per_video=args.max_frames_per_video,
        )

        if not samples:
            continue

        total_frames_extracted += len(samples)

        if model is not None:
            run_yolo_prelabel(
                model=model,
                samples=samples,
                labels_dir=tmp_labels_root,
                meta_fp=meta_fp,
                video_id=video_id,
                imgsz=args.imgsz,
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
                force_single_class=args.force_single_class,
                device=args.device,
            )
        else:
            for frame_path, frame_idx in samples:
                rec = {
                    "video_id": video_id,
                    "frame_file": frame_path.name,
                    "frame_idx": frame_idx,
                    "num_boxes": None,
                }
                meta_fp.write(json.dumps(rec) + "\n")

        video_to_frames[video_id].extend(samples)

    meta_fp.close()

    # split
    train_videos, val_videos, test_videos = make_video_level_split_grouped(
        video_to_frames.keys(),
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        seed=args.seed,
        by_folder=args.split_by_folder,
    )

    move_to_final_layout(
        video_to_frames,
        frames_root=tmp_frames_root,
        labels_root=tmp_labels_root,
        out_root=out_root,
        train_videos=train_videos,
        val_videos=val_videos,
        test_videos=test_videos,
    )

    write_data_yaml(out_root, names_dict)

    shutil.rmtree(tmp_frames_root, ignore_errors=True)
    shutil.rmtree(tmp_labels_root, ignore_errors=True)

    print(f"[DONE] Dataset ready at: {out_root}")
    print(f"[STATS] Videos: {len(videos)}, extracted frames: {total_frames_extracted}")


if __name__ == "__main__":
    main()
