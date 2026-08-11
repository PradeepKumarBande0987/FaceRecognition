from __future__ import annotations

import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


C_GREEN = (50, 220, 50)
C_YELLOW = (30, 200, 230)
C_RED = (50, 50, 220)
C_WHITE = (255, 255, 255)
C_GOLD = (30, 215, 255)
C_DARK = (20, 20, 20)
C_GRAY = (150, 150, 150)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_B = cv2.FONT_HERSHEY_DUPLEX


def _det_get(det: Any, key: str, default: Any = None) -> Any:
    """Read detection fields from either dict-like or object-like outputs."""
    if isinstance(det, dict):
        return det.get(key, default)
    return getattr(det, key, default)


def _coerce_bbox(bbox: Any) -> Tuple[int, int, int, int] | None:
    if bbox is None:
        return None
    try:
        x1, y1, x2, y2 = bbox
        return int(x1), int(y1), int(x2), int(y2)
    except Exception:
        return None


def _build_face_crop(aligner: Any, frame: np.ndarray, bbox: Tuple[int, int, int, int] | None, landmarks: Any) -> np.ndarray | None:
    """Try common aligner signatures; fallback to bbox crop when needed."""
    if bbox is not None and landmarks is not None:
        try:
            return aligner.align(frame, bbox, landmarks)
        except TypeError:
            pass
        except Exception:
            pass

    if landmarks is not None:
        try:
            return aligner.align(frame, landmarks)
        except Exception:
            pass

    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


class OverlayRenderer:
    """Renders overlays for live webcam recognition."""

    def __init__(self, show_topk: bool = True, show_fps: bool = True, show_security: bool = True, top_k: int = 3):
        self.show_topk = show_topk
        self.show_fps = show_fps
        self.show_security = show_security
        self.top_k = top_k
        self._fps_buf = deque(maxlen=30)

    def match_color(self, score: float) -> Tuple[int, int, int]:
        if score >= 0.65:
            return C_GREEN
        if score >= 0.45:
            return C_YELLOW
        return C_RED

    def draw_bbox(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], color: Tuple[int, int, int], thick: int = 2):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

    def draw_label(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], text: str, color: Tuple[int, int, int], sub_text: str = ""):
        x1, y1, x2, y2 = bbox
        (tw, th), _ = cv2.getTextSize(text, FONT_B, 0.60, 2)
        pad = 5
        lx1 = x1
        ly1 = max(0, y1 - th - 2 * pad - 2)
        lx2 = x1 + tw + 2 * pad
        ly2 = y1

        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, -1)
        cv2.putText(frame, text, (lx1 + pad, ly2 - pad), FONT_B, 0.60, C_WHITE, 2, cv2.LINE_AA)

        if sub_text:
            cv2.putText(frame, sub_text, (x1, y2 + 18), FONT, 0.45, color, 1, cv2.LINE_AA)

    def draw_security_badge(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], result: Any):
        if not self.show_security:
            return

        x1, y1, x2, y2 = bbox
        badges = [
            ("L", result.liveness_ok, result.liveness_score),
            ("A", result.antispoof_ok, result.antispoof_score),
            ("D", result.adversarial_ok, result.adversarial_score),
        ]

        bx = x2 + 5
        for i, (letter, ok, score) in enumerate(badges):
            by = y1 + i * 22
            color = C_GREEN if ok else C_RED
            cv2.rectangle(frame, (bx, by), (bx + 20, by + 18), color, -1)
            cv2.putText(frame, letter, (bx + 4, by + 14), FONT, 0.45, C_WHITE, 1, cv2.LINE_AA)
            cv2.putText(frame, f"{score:.2f}", (bx + 23, by + 14), FONT, 0.38, color, 1, cv2.LINE_AA)

    def draw_status_bar(self, frame: np.ndarray, db_size: int, n_faces: int, fps: float):
        h, w = frame.shape[:2]
        bh = 34
        ov = frame.copy()
        cv2.rectangle(ov, (0, h - bh), (w, h), (10, 10, 10), -1)
        cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)

        parts = [
            f"DB: {db_size:,} ids",
            f"Faces: {n_faces}",
            f"FPS: {fps:.1f}" if self.show_fps else "",
            "Q:Quit  S:Snap  H:Help  SPACE:Pause",
        ]
        x = 10
        for p in parts:
            if not p:
                continue
            (tw, _), _ = cv2.getTextSize(p, FONT, 0.48, 1)
            cv2.putText(frame, p, (x, h - 10), FONT, 0.48, C_GRAY, 1, cv2.LINE_AA)
            x += tw + 28

    def draw_topk_panel(self, frame: np.ndarray, matches: List[Dict], face_idx: int = 0):
        if not self.show_topk or not matches:
            return

        h, w = frame.shape[:2]
        pw = 260
        lh = 46
        ph = len(matches) * lh + 48
        px1 = w - pw - 8
        py1 = 8 + face_idx * (ph + 8)
        px2 = w - 8
        py2 = py1 + ph

        if py2 >= h or px1 < 0:
            return

        ov = frame.copy()
        cv2.rectangle(ov, (px1, py1), (px2, py2), C_DARK, -1)
        cv2.addWeighted(ov, 0.82, frame, 0.18, 0, frame)

        top_col = self.match_color(matches[0]["similarity"])
        cv2.rectangle(frame, (px1, py1), (px2, py2), top_col, 1)

        cv2.putText(frame, f"Top-{len(matches)} Matches", (px1 + 8, py1 + 20), FONT_B, 0.50, C_GOLD, 1, cv2.LINE_AA)

        for i, m in enumerate(matches):
            ry = py1 + 38 + i * lh
            col = self.match_color(m["similarity"])
            s = m["similarity"]
            cv2.putText(frame, f"{i+1}. {m['display_name'][:18]}", (px1 + 10, ry + 8), FONT, 0.45, C_WHITE, 1, cv2.LINE_AA)
            cv2.putText(frame, f"{s:.3f}", (px2 - 52, ry + 8), FONT, 0.40, col, 1, cv2.LINE_AA)

    def draw_help(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        lines = [
            ("KEYBOARD SHORTCUTS", C_GOLD),
            ("Q / ESC  Quit", C_WHITE),
            ("S        Save snapshot", C_WHITE),
            ("H        Toggle help", C_WHITE),
            ("T        Toggle top-K", C_WHITE),
            ("L        Toggle liveness", C_WHITE),
            ("A        Toggle anti-spoof", C_WHITE),
            ("F        Toggle FPS", C_WHITE),
            ("SPACE    Pause/Resume", C_WHITE),
            ("R        Re-register DB", C_WHITE),
        ]

        bw, bh = 300, len(lines) * 22 + 20
        bx1 = w // 2 - bw // 2
        by1 = h // 2 - bh // 2
        ov = frame.copy()
        cv2.rectangle(ov, (bx1 - 10, by1 - 10), (bx1 + bw + 10, by1 + bh + 10), (15, 15, 15), -1)
        cv2.addWeighted(ov, 0.92, frame, 0.08, 0, frame)
        cv2.rectangle(frame, (bx1 - 10, by1 - 10), (bx1 + bw + 10, by1 + bh + 10), C_GOLD, 1)

        for i, (text, col) in enumerate(lines):
            cv2.putText(frame, text, (bx1, by1 + 18 + i * 22), FONT, 0.50, col, 1, cv2.LINE_AA)

    def update_fps(self) -> float:
        self._fps_buf.append(time.perf_counter())
        if len(self._fps_buf) < 2:
            return 0.0
        dt = self._fps_buf[-1] - self._fps_buf[0]
        return (len(self._fps_buf) - 1) / max(dt, 1e-6)


def run_webcam_recognition(
    extractor: Any,
    detector: Any,
    aligner: Any,
    database: Any,
    security: Any,
    cfg: Any,
    snapshot_dir: Path | None = None,
) -> dict:
    """Main real-time webcam loop extracted from main.py."""
    cap = cv2.VideoCapture(cfg.webcam_id)
    if not cap.isOpened():
        return {"error": "no_webcam"}

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
    cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    renderer = OverlayRenderer(
        show_topk=cfg.show_topk,
        show_fps=cfg.show_fps,
        show_security=cfg.show_security,
        top_k=cfg.top_k,
    )

    frame_count = 0
    snap_count = 0
    session_start = time.perf_counter()
    paused = False
    show_help = False
    liveness_overlay_on = True
    antispoof_overlay_on = True
    if snapshot_dir is None:
        snapshot_dir = Path("docs/results/webcam_snapshots")
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    win_name = "LFW Face Recognition"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            frame_count += 1
            fps = renderer.update_fps()

            if paused:
                overlay_txt = "  PAUSED - Press SPACE to resume  "
                (tw, th), _ = cv2.getTextSize(overlay_txt, FONT_B, 0.9, 2)
                x0 = (frame.shape[1] - tw) // 2
                y0 = frame.shape[0] // 2
                cv2.rectangle(frame, (x0 - 10, y0 - th - 10), (x0 + tw + 10, y0 + 10), C_DARK, -1)
                cv2.putText(frame, overlay_txt, (x0, y0), FONT_B, 0.9, C_GOLD, 2, cv2.LINE_AA)
                cv2.imshow(win_name, frame)

                key = cv2.waitKey(30) & 0xFF
                if key in [ord("q"), 27]:
                    break
                if key == ord(" "):
                    paused = False
                if key == ord("h"):
                    show_help = not show_help
                continue

            detections = detector.detect(frame)

            for i, det in enumerate(detections):
                bbox = _coerce_bbox(_det_get(det, "bbox"))
                landmarks = _det_get(det, "landmarks")
                face_crop = _det_get(det, "face_crop")
                if face_crop is None:
                    face_crop = _build_face_crop(aligner, frame, bbox, landmarks)
                if face_crop is None or face_crop.size == 0 or bbox is None:
                    continue
                sec_result = security.check(face_crop)

                if not liveness_overlay_on:
                    sec_result.liveness_ok = True
                    sec_result.liveness_score = 1.0
                    sec_result.liveness_reason = "off"
                if not antispoof_overlay_on:
                    sec_result.antispoof_ok = True
                    sec_result.antispoof_score = 1.0
                    sec_result.antispoof_reason = "off"

                if sec_result.all_passed or not cfg.enable_security:
                    emb = extractor.extract(face_crop)
                    matches = database.search(emb, top_k=cfg.top_k, threshold=cfg.threshold)
                else:
                    matches = []

                if matches:
                    color = renderer.match_color(matches[0]["similarity"])
                    name = matches[0]["display_name"]
                    score = matches[0]["similarity"]
                    label = f"{name[:22]}  {score:.2f}"
                    sub = sec_result.summary() if not sec_result.all_passed else ""
                elif not sec_result.all_passed:
                    color = C_RED
                    label = "UNKNOWN PERSON"
                    sub = sec_result.summary()
                else:
                    color = C_RED
                    label = "UNKNOWN"
                    sub = ""

                renderer.draw_bbox(frame, bbox, color)
                renderer.draw_label(frame, bbox, label, color, sub)
                renderer.draw_security_badge(frame, bbox, sec_result)

                if renderer.show_topk and matches:
                    renderer.draw_topk_panel(frame, matches, face_idx=i)

            renderer.draw_status_bar(frame, len(database), len(detections), fps)
            if show_help:
                renderer.draw_help(frame)
            cv2.imshow(win_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:
                break
            if key == ord("s"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = snapshot_dir / f"snap_{ts}_{snap_count:03d}.jpg"
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                snap_count += 1
            elif key == ord("h"):
                show_help = not show_help
            elif key == ord("t"):
                renderer.show_topk = not renderer.show_topk
            elif key == ord("l"):
                liveness_overlay_on = not liveness_overlay_on
            elif key == ord("a"):
                antispoof_overlay_on = not antispoof_overlay_on
            elif key == ord("f"):
                renderer.show_fps = not renderer.show_fps
            elif key == ord(" "):
                paused = True
            elif key == ord("r"):
                if hasattr(database, "_embeddings") and isinstance(getattr(database, "_embeddings"), dict):
                    database._embeddings.clear()
                if hasattr(database, "_metadata") and isinstance(getattr(database, "_metadata"), dict):
                    database._metadata.clear()
                if hasattr(database, "_ids") and isinstance(getattr(database, "_ids"), list):
                    database._ids.clear()
                if hasattr(database, "_embs"):
                    try:
                        database._embs = None
                    except Exception:
                        pass
                print("[INFO] Re-register requested: in-memory DB cleared. Restart to rebuild from dataset.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - session_start
    return {
        "frames_processed": frame_count,
        "snapshots_saved": snap_count,
        "session_seconds": round(elapsed, 1),
        "avg_fps": round(frame_count / max(elapsed, 0.01), 1),
        "db_size": len(database),
        "threshold_used": cfg.threshold,
    }
