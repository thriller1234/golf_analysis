"""
クラブ用 YOLO Segment データセット CLI。

  引数なし … extract → annotate → verify を順に実行
  extract / run / prepare … 動画から画像切り出しのみ
  annotate … 画面上で帯アノテ（2点）のみ
  verify … 画像とラベルの対応確認のみ

学習は scripts/train_yolov8_club.py を直接実行する。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# --- 共有定数 ---
CLASS_NAME = "golf_club"
CLASS_ID = 0
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_DATASET = Path("data/datasets/golf_club_seg")
DEFAULT_TARGET_IMAGES = 500
DEFAULT_STRIP_HALF_WIDTH = 7.0
WIN = "golf_club_annotate"
MAX_DISPLAY_SIDE = 1440


def _root() -> Path:
    return _ROOT


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else _root() / p


# ----- extract (prepare) -----
def is_validation_video(video_stem: str, train_ratio: float) -> bool:
    d = hashlib.md5(video_stem.encode("utf-8")).digest()
    u = int.from_bytes(d[:4], "big") / (2**32)
    return u > train_ratio


def collect_video_files(input_path: Path, *, recursive: bool) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTS:
            raise SystemExit(f"対応していない拡張子です: {input_path}")
        return [input_path.resolve()]
    if not input_path.is_dir():
        raise SystemExit(f"入力がありません: {input_path}")
    found: List[Path] = []
    if recursive:
        for ext in VIDEO_EXTS:
            found.extend(input_path.rglob(f"*{ext}"))
    else:
        for p in sorted(input_path.iterdir()):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                found.append(p.resolve())
    return sorted(set(found))


def video_frame_counts(videos: List[Path]) -> Dict[Path, int]:
    out: Dict[Path, int] = {}
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            continue
        nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if nf > 0:
            out[v.resolve()] = nf
    return out


def strip_quad_from_centerline(
    ax: int,
    ay: int,
    bx: int,
    by: int,
    half_w: float,
    img_w: int,
    img_h: int,
) -> List[Tuple[int, int]]:
    """中心線 AB に直交する帯（細長い四角形）の 4 頂点。画像内にクリップ。"""
    dx, dy = bx - ax, by - ay
    length = float(np.hypot(dx, dy))
    if length < 1.0:
        return []
    nx, ny = -dy / length, dx / length
    ox, oy = nx * half_w, ny * half_w

    def clip(px: float, py: float) -> Tuple[int, int]:
        return (
            int(np.clip(round(px), 0, img_w - 1)),
            int(np.clip(round(py), 0, img_h - 1)),
        )

    return [
        clip(ax + ox, ay + oy),
        clip(bx + ox, by + oy),
        clip(bx - ox, by - oy),
        clip(ax - ox, ay - oy),
    ]


def sample_random_frames(counts: Dict[Path, int], target_n: int, seed: int) -> List[Tuple[Path, int]]:
    """ランダムに (動画, フレーム) を選ぶ。各動画から先に1枚ずつ（可能なら）、残りは動画間を均等に近づける。"""
    rng = random.Random(seed)
    paths = list(counts.keys())
    if not paths:
        raise SystemExit("有効な動画フレームがありません（長さ 0 など）。")
    max_possible = sum(counts.values())
    target_effective = min(target_n, max_possible)
    weights = [1.0] * len(paths)

    picked: List[Tuple[Path, int]] = []
    seen: Set[Tuple[str, int]] = set()

    # まず各ファイルから 1 フレームずつ（多様性・ファイル間バランス）
    if target_effective >= len(paths):
        rng.shuffle(paths)
        for v in paths:
            if len(picked) >= target_effective:
                break
            nf = counts[v]
            for _try in range(min(nf * 2, 200)):
                fi = rng.randrange(0, nf)
                key = (str(v), fi)
                if key not in seen:
                    seen.add(key)
                    picked.append((v, fi))
                    break

    attempts = 0
    max_attempts = max(target_effective * 300, 80000)
    while len(picked) < target_effective and attempts < max_attempts:
        attempts += 1
        if len(seen) >= max_possible:
            break
        v = rng.choices(paths, weights=weights, k=1)[0]
        nf = counts[v]
        fi = rng.randrange(0, nf)
        key = (str(v), fi)
        if key in seen:
            continue
        seen.add(key)
        picked.append((v, fi))

    if len(picked) < target_n:
        print(
            f"注意: 目標 {target_n} 枚に対し、重複なしで取得できたのは {len(picked)} 枚です（合計ユニークフレーム上限 {max_possible}）。"
        )
    return picked


def read_frame_at(video_path: Path, fi: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if nf <= 0:
        cap.release()
        return None
    fi = max(0, min(fi, nf - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    return frame


def write_data_yaml(out_dir: Path) -> None:
    import yaml

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": [CLASS_NAME],
    }
    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True)


def write_annotation_guide(out_dir: Path) -> None:
    p = out_dir / "ANNOTATION.md"
    body = f"""# 手動アノテーションのメモ（自動生成）

## クラス: ID `0` / 名前 `{CLASS_NAME}`

## 推奨（同梱 CLI）

```powershell
python scripts/club_dataset.py annotate --strip-half-width {DEFAULT_STRIP_HALF_WIDTH:.0f} --out <このフォルダ>
```

シャフトの両端付近を左クリック 2 回 → Enter。`--strip-half-width`（原画像ピクセル、片側幅、既定 {DEFAULT_STRIP_HALF_WIDTH:.0f}）で帯の太さを指定し、YOLO segment 形式で保存されます。画面上ではその幅で半透明表示されます。

## 細いシャフトと太いヘッド

帯は**一定の太さ**です。目標を「主にシャフトの太さに合わせる」なら、既定の片側幅のまま、2 点を**グリップ近く→ヘッド付け根／トゥ方向**の中心線上に取ると一貫しやすいです。ヘッドの外周までは厳密に取れません（片側幅を大きくしすぎるとシャフト部分の余白が増えすぎるため、**太さは画像解像度に合わせて** `--strip-half-width` だけ調整するのがおすすめです）。

## 作業の流れ

1. `python scripts/club_dataset.py annotate --out <このフォルダ>`
2. `python scripts/club_dataset.py verify --out <このフォルダ>`
3. `python scripts/train_yolov8_club.py --dataset <このフォルダ>`

## 一覧 CSV

`images_for_annotation.csv`（`extract` 時、既定）。不要なら `extract --no-manifest`。
"""
    p.write_text(body, encoding="utf-8")


def write_manifest(out_dir: Path, rows: list[tuple[str, str, str]]) -> None:
    with open(out_dir / "images_for_annotation.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "filename", "path_relative_to_dataset_root"])
        for split, name, rel in rows:
            w.writerow([split, name, rel.replace("\\", "/")])


def clean_image_dirs(out_dir: Path) -> None:
    for split in ("train", "val"):
        d = out_dir / split / "images"
        if not d.is_dir():
            continue
        for p in d.glob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                p.unlink()


def cmd_extract(args: argparse.Namespace) -> None:
    inp = args.input if args.input is not None else Path("data/videos")
    inp = _resolve(inp)
    od = _resolve(args.out)
    videos = collect_video_files(inp, recursive=args.recursive)
    if not videos:
        raise SystemExit(f"動画が見つかりません: {inp}")
    counts = video_frame_counts(videos)
    if not counts:
        raise SystemExit("読み込める動画がありませんでした。")
    picks = sample_random_frames(counts, min(args.target_images, sum(counts.values())), args.seed)
    od.mkdir(parents=True, exist_ok=True)
    for sub in ("train", "val"):
        (od / sub / "images").mkdir(parents=True, exist_ok=True)
        (od / sub / "labels").mkdir(parents=True, exist_ok=True)
    if args.clean:
        clean_image_dirs(od)
    write_data_yaml(od)
    write_annotation_guide(od)
    by_video: Dict[Path, List[int]] = defaultdict(list)
    for v, fi in picks:
        by_video[v].append(fi)
    manifest_rows: list[tuple[str, str, str]] = []
    n_written = 0
    train_ratio = float(min(0.999, max(0.01, args.train_ratio)))
    for video_path, frame_indices in tqdm(
        sorted(by_video.items(), key=lambda x: str(x[0])), desc="videos"
    ):
        stem = video_path.stem
        split = "val" if is_validation_video(stem, train_ratio) else "train"
        for fi in tqdm(sorted(set(frame_indices)), desc=stem[:24], leave=False):
            frame = read_frame_at(video_path, fi)
            if frame is None:
                print(f"読み取り失敗: {video_path.name} frame {fi}")
                continue
            img_name = f"{stem}_f{fi:06d}.jpg"
            img_path = od / split / "images" / img_name
            cv2.imwrite(str(img_path), frame)
            manifest_rows.append((split, img_name, f"{split}/images/{img_name}"))
            n_written += 1
    if not args.no_manifest:
        write_manifest(od, manifest_rows)
    try:
        rel = od.relative_to(_root())
    except ValueError:
        rel = od
    print(f"完了: 画像 {n_written} 枚 → {od}")


# ----- annotate -----
def collect_tasks(dataset: Path, splits: List[str]) -> List[Tuple[str, Path, Path]]:
    out: List[Tuple[str, Path, Path]] = []
    for sp in splits:
        img_dir = dataset / sp / "images"
        lab_dir = dataset / sp / "labels"
        if not img_dir.is_dir():
            continue
        lab_dir.mkdir(parents=True, exist_ok=True)
        for img_path in sorted(img_dir.iterdir(), key=lambda p: p.name.lower()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            out.append((sp, img_path, lab_dir / f"{img_path.stem}.txt"))
    return out


def label_is_complete(lab_path: Path) -> bool:
    if not lab_path.is_file():
        return False
    try:
        parts = lab_path.read_text(encoding="utf-8").strip().split()
    except OSError:
        return False
    return len(parts) >= 7 and parts[0].isdigit()


def load_polygon_pixels(lab_path: Path, w: int, h: int) -> List[Tuple[int, int]]:
    if not lab_path.is_file():
        return []
    parts = lab_path.read_text(encoding="utf-8").strip().split()
    if len(parts) < 7:
        return []
    pts: List[Tuple[int, int]] = []
    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            break
        pts.append((int(round(float(parts[i]) * w)), int(round(float(parts[i + 1]) * h))))
    return pts


def save_yolo_polygon(lab_path: Path, points: List[Tuple[int, int]], w: int, h: int) -> None:
    parts = [str(CLASS_ID)]
    for x, y in points:
        parts.append(f"{x / w:.6f}")
        parts.append(f"{y / h:.6f}")
    lab_path.parent.mkdir(parents=True, exist_ok=True)
    lab_path.write_text(" ".join(parts) + "\n", encoding="utf-8")


def display_scale(orig: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    h, w = orig.shape[:2]
    m = max(h, w)
    if m <= MAX_DISPLAY_SIDE:
        return orig.copy(), 1.0, (w, h)
    sc = MAX_DISPLAY_SIDE / float(m)
    nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
    disp = cv2.resize(orig, (nw, nh), interpolation=cv2.INTER_AREA)
    return disp, sc, (w, h)


def disp_to_orig(xd: int, yd: int, scale: float) -> Tuple[int, int]:
    if scale >= 1.0:
        return xd, yd
    return int(round(xd / scale)), int(round(yd / scale))


def orig_to_disp(xo: int, yo: int, scale: float) -> Tuple[int, int]:
    if scale >= 1.0:
        return xo, yo
    return int(round(xo * scale)), int(round(yo * scale))


def _blend_polygon_fill(
    dst: np.ndarray, pts_disp: List[Tuple[int, int]], bgr: Tuple[int, int, int], alpha: float
) -> None:
    """pts は表示座標。alpha で塗りつぶし（学習時マスクと同じ形状を画面上で再現）。"""
    if len(pts_disp) < 3:
        return
    a = float(np.clip(alpha, 0.0, 1.0))
    arr = np.array(pts_disp, dtype=np.int32).reshape((-1, 1, 2))
    overlay = dst.copy()
    cv2.fillPoly(overlay, [arr], bgr)
    cv2.addWeighted(overlay, a, dst, 1.0 - a, 0, dst)


def draw_ui(
    base_disp: np.ndarray,
    points_orig: List[Tuple[int, int]],
    scale: float,
    title_line: str,
    *,
    strip_half_w: float,
    ow: int,
    oh: int,
    ref_poly_orig: List[Tuple[int, int]] | None = None,
) -> np.ndarray:
    """帯アノテ用 UI。2 点確定時は原画像座標で算出した四角形を表示スケールに合わせて半透明塗り（線の太さではなく実セグメント幅）。"""
    vis = base_disp.copy()
    h, w = vis.shape[:2]
    pts_disp = [orig_to_disp(px, py, scale) for px, py in points_orig]

    if ref_poly_orig and len(ref_poly_orig) >= 3:
        ref_d = [orig_to_disp(px, py, scale) for px, py in ref_poly_orig]
        _blend_polygon_fill(vis, ref_d, (80, 160, 80), 0.22)

    if len(points_orig) == 2 and ow > 0 and oh > 0:
        ax, ay = points_orig[0]
        bx, by = points_orig[1]
        quad = strip_quad_from_centerline(ax, ay, bx, by, strip_half_w, ow, oh)
        if len(quad) == 4:
            qd = [orig_to_disp(qx, qy, scale) for qx, qy in quad]
            _blend_polygon_fill(vis, qd, (0, 180, 255), 0.42)
            arr = np.array(qd, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [arr], True, (255, 220, 100), 1, cv2.LINE_AA)

    for p in pts_disp:
        cv2.circle(vis, p, 3, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, p, 4, (255, 255, 255), 1, cv2.LINE_AA)

    bar = np.zeros((56, w, 3), dtype=np.uint8)
    bar[:] = (40, 40, 40)
    vis = np.vstack([vis, bar])
    cv2.putText(vis, title_line[:120], (8, h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    help_str = (
        f"左クリックで端点（2点まで）| Enter: 保存して次へ | R: クリア | U: 戻す | K: スキップ | Q/Esc: 終了 "
        f"| 帯の片側幅={strip_half_w:.0f}px（原画像）"
    )
    cv2.putText(vis, help_str, (8, h + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    return vis


def cmd_annotate(args: argparse.Namespace) -> None:
    ds = _resolve(args.out).resolve()
    sp_list = [s.strip() for s in args.splits.split(",") if s.strip()]
    all_tasks = collect_tasks(ds, sp_list)
    if not all_tasks:
        raise SystemExit(f"画像がありません: {ds}")
    if args.no_resume:
        tasks = all_tasks
    else:
        tasks = [t for t in all_tasks if not label_is_complete(t[2])]
        skipped = len(all_tasks) - len(tasks)
        if skipped:
            print(f"既にラベル済みの画像 {skipped} 枚をスキップします（全件やり直す: --no-resume）")
    if not tasks:
        print("アノテーション対象の画像がありません。")
        return
    strip_hw = float(max(0.5, args.strip_half_width))
    print(
        f"帯アノテ（2点のみ）: シャフトの両端付近を左クリック → Enter で保存（原画像の片側幅 {strip_hw:.1f} px）。\n"
        "※ 既存ラベルは参照として薄く表示します（--no-resume 時）。クリックは常に 2 点からやり直しです。\n"
    )
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    idx = 0
    try:
        while idx < len(tasks):
            split, img_path, lab_path = tasks[idx]
            orig = cv2.imread(str(img_path))
            if orig is None:
                print(f"読み込み失敗: {img_path}")
                idx += 1
                continue
            oh, ow = orig.shape[:2]
            base_disp, scale, _ = display_scale(orig)
            disp_h = base_disp.shape[0]
            points: List[Tuple[int, int]] = []
            ref_poly: List[Tuple[int, int]] | None = None
            if args.no_resume and lab_path.exists():
                loaded = load_polygon_pixels(lab_path, ow, oh)
                if len(loaded) >= 3:
                    ref_poly = loaded

            def on_mouse(event: int, x: int, y: int, _f: int, _p: None) -> None:
                if event != cv2.EVENT_LBUTTONDOWN or y >= disp_h:
                    return
                if len(points) >= 2:
                    return
                ox, oy = disp_to_orig(x, y, scale)
                points.append((int(np.clip(ox, 0, ow - 1)), int(np.clip(oy, 0, oh - 1))))

            cv2.setMouseCallback(WIN, on_mouse)
            inner = True
            while inner:
                title = f"{idx + 1}/{len(tasks)}  [{split}]  {img_path.name}  |  pts={len(points)}/2"
                cv2.setWindowTitle(WIN, title)
                cv2.imshow(
                    WIN,
                    draw_ui(
                        base_disp,
                        points,
                        scale,
                        title,
                        strip_half_w=strip_hw,
                        ow=ow,
                        oh=oh,
                        ref_poly_orig=ref_poly,
                    ),
                )
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q") or key == 27:
                    print("終了しました。")
                    return
                if key == ord("r"):
                    points.clear()
                elif key == ord("u") and points:
                    points.pop()
                elif key == ord("k"):
                    idx += 1
                    inner = False
                elif key in (13, 10):
                    if len(points) == 2:
                        ax, ay = points[0]
                        bx, by = points[1]
                        quad = strip_quad_from_centerline(ax, ay, bx, by, strip_hw, ow, oh)
                        if len(quad) == 4:
                            save_yolo_polygon(lab_path, quad, ow, oh)
                            print(f"保存: {lab_path.relative_to(ds)}")
                            idx += 1
                            inner = False
                        else:
                            print("2 点が近すぎます。離して選び直してください。")
                    else:
                        print("端点を 2 点クリックしてから Enter を押してください。")
    finally:
        cv2.setMouseCallback(WIN, lambda *a: None)
        cv2.destroyWindow(WIN)
    print(f"完了: {len(tasks)} 枚を処理しました。")


# ----- verify -----
def _img_stems(img_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in img_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out[p.stem] = p
    return out


def cmd_verify(args: argparse.Namespace) -> int:
    dataset = _resolve(args.out).resolve()
    if not dataset.is_dir():
        print(f"エラー: データセットがありません: {dataset}")
        return 1
    if not (dataset / "data.yaml").is_file():
        print(f"注意: data.yaml がありません: {dataset / 'data.yaml'}")
    total_missing = 0
    total_orphan = 0
    for split in ("train", "val"):
        img_dir = dataset / split / "images"
        lab_dir = dataset / split / "labels"
        if not img_dir.is_dir():
            print(f"  [{split}] images/ なし → スキップ")
            continue
        stems_img = _img_stems(img_dir)
        labels = {p.stem: p for p in lab_dir.glob("*.txt")} if lab_dir.is_dir() else {}
        missing = sorted(set(stems_img) - set(labels))
        orphan = sorted(set(labels) - set(stems_img))
        if missing:
            print(f"  [{split}] 画像はあるが .txt が無い: {len(missing)} 件（例: {missing[:3]}）")
            total_missing += len(missing)
        if orphan:
            print(f"  [{split}] .txt だけ残っている: {len(orphan)} 件（例: {orphan[:3]}）")
            total_orphan += len(orphan)
        n_img = len(stems_img)
        n_lab = len([s for s in stems_img if s in labels])
        print(f"  [{split}] 画像 {n_img} 枚 / 対応ラベル付き {n_lab} 枚")
    if total_missing or total_orphan:
        print("\n※ アノテ途中ではラベル無しが残ります。")
        return 0
    print("\n整合チェック: 各画像に対応する .txt が揃っています。")
    return 0


def _normalize_step(step: str | None) -> str | None:
    if step is None:
        return None
    if step in ("run", "prepare"):
        return "extract"
    return step


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="クラブ用 YOLO Segment データセット CLI（引数なしで extract→annotate→verify）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "step",
        nargs="?",
        choices=("extract", "annotate", "verify", "run", "prepare"),
        default=None,
        metavar="STEP",
        help="extract / annotate / verify（run・prepare は extract と同じ）。省略時は 3 工程を順に実行。",
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=None,
        help="動画ファイルまたはディレクトリ（既定: data/videos）",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_DATASET,
        help="データセットルート（extract の出力先、annotate・verify の対象でもある）",
    )
    ap.add_argument("--target-images", type=int, default=DEFAULT_TARGET_IMAGES)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--no-manifest", action="store_true")
    ap.add_argument("--no-resume", dest="no_resume", action="store_true", help="annotate: 既存ラベルも含め再アノテ")
    ap.add_argument("--splits", type=str, default="train,val", help="annotate: 対象 split（カンマ区切り）")
    ap.add_argument(
        "--strip-half-width",
        type=float,
        default=DEFAULT_STRIP_HALF_WIDTH,
        help="annotate: 原画像ピクセルでの帯の中心線の片側幅（全幅は約2倍）",
    )
    return ap


def main() -> None:
    args = build_parser().parse_args()
    step = _normalize_step(args.step)

    if step is None:
        print("=== extract（動画→画像）===\n")
        cmd_extract(args)
        print("\n=== annotate（帯・2点）===\n")
        cmd_annotate(args)
        print("\n=== verify（整合確認）===\n")
        raise SystemExit(cmd_verify(args))

    if step == "extract":
        cmd_extract(args)
    elif step == "annotate":
        cmd_annotate(args)
    elif step == "verify":
        raise SystemExit(cmd_verify(args))


if __name__ == "__main__":
    main()
