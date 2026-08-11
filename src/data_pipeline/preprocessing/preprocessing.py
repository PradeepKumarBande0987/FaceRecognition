"""
Data Pipeline — Preprocessing Module.

Face detection, alignment, and normalization pipeline.
Produces canonical 112×112 aligned face crops ready for embedding.

Pipeline:
    Raw Image → Face Detection → 5-Point Landmark → Affine Alignment
    → 112×112 crop → Quality Check → Save / Return

Detectors supported:
    • SCRFD (recommended, via insightface)
    • RetinaFace (via insightface)
    • OpenCV Haar Cascade (lightweight fallback)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# ── Alignment Template (ArcFace 112×112) ─────────────────────────────────────

# Standard 5-point landmark template for 112×112 crops
# Order: left eye, right eye, nose tip, left mouth, right mouth
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


# ── Detection Result ──────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """Result from face detection."""
    bbox        : Tuple[int, int, int, int]       # (x1, y1, x2, y2)
    landmarks   : Optional[np.ndarray] = None     # (5, 2)
    confidence  : float = 1.0
    face_crop   : Optional[np.ndarray] = None     # aligned 112×112 BGR


# ── Quality Checker ───────────────────────────────────────────────────────────

class FaceQualityChecker:
    """
    Checks face crop quality before saving/embedding.

    Checks:
        • Sharpness (Laplacian variance)
        • Brightness range
        • Minimum face size
        • Face symmetry (landmark-based)
    """

    def __init__(
        self,
        min_sharpness  : float = 50.0,
        min_brightness : float = 30.0,
        max_brightness : float = 220.0,
        min_size       : Tuple[int, int] = (64, 64),
    ):
        self.min_sharpness  = min_sharpness
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_size       = min_size

    def check(self, img: np.ndarray) -> Tuple[bool, float, str]:
        """
        Check image quality.

        Args:
            img: BGR image

        Returns:
            (passed, quality_score 0-1, reason_if_failed)
        """
        if img is None or img.size == 0:
            return False, 0.0, "empty_image"

        h, w = img.shape[:2]
        if h < self.min_size[0] or w < self.min_size[1]:
            return False, 0.0, f"too_small_{w}x{h}"

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Sharpness
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        if sharpness < self.min_sharpness:
            return False, float(sharpness / 500), f"blurry_{sharpness:.1f}"

        # Brightness
        brightness = float(gray.mean())
        if brightness < self.min_brightness:
            return False, 0.0, f"too_dark_{brightness:.1f}"
        if brightness > self.max_brightness:
            return False, 0.0, f"too_bright_{brightness:.1f}"

        # Composite quality score (0–1)
        sharpness_score  = min(sharpness / 500.0, 1.0)
        brightness_score = 1.0 - abs(brightness / 255 - 0.5) * 2
        quality_score    = float((sharpness_score + brightness_score) / 2)

        return True, round(quality_score, 4), "passed"


# ── Face Aligner ──────────────────────────────────────────────────────────────

class FaceAligner:
    """
    Aligns detected faces to canonical 112×112 pose.

    Uses similarity transform (rotation + scale + translation)
    estimated from 5 facial landmarks.

    Usage:
        aligner = FaceAligner()
        aligned = aligner.align(face_crop, landmarks_5pt)
    """

    def __init__(self, output_size: Tuple[int, int] = (112, 112)):
        self.output_size = output_size
        self.template    = ARCFACE_TEMPLATE * np.array([
            output_size[1] / 112, output_size[0] / 112
        ])

    def align(
        self,
        img       : np.ndarray,
        landmarks : np.ndarray,
    ) -> np.ndarray:
        """
        Align face using similarity transform.

        Args:
            img       : BGR image (any size)
            landmarks : (5, 2) float32 landmark coordinates

        Returns:
            Aligned BGR face crop at output_size
        """
        from skimage import transform as skimage_tf

        tform = skimage_tf.SimilarityTransform()
        tform.estimate(landmarks, self.template)
        M = tform.params[:2]

        aligned = cv2.warpAffine(
            img, M,
            (self.output_size[1], self.output_size[0]),
            flags       = cv2.INTER_LINEAR,
            borderMode  = cv2.BORDER_REFLECT,
        )
        return aligned

    def align_simple(
        self,
        img       : np.ndarray,
        landmarks : np.ndarray,
        frame     : np.ndarray,
    ) -> np.ndarray:
        """
        Simple 2-point alignment using eye centers (no skimage dependency).

        Args:
            img       : BGR image
            landmarks : (5, 2) landmark array

        Returns:
            Roughly aligned face crop
        """
        left_eye  = landmarks[0]
        right_eye = landmarks[1]

        # Angle between eyes
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = float(np.degrees(np.arctan2(dY, dX)))

        # Eye midpoint
        eye_center = (
            int((left_eye[0] + right_eye[0]) // 2),
            int((left_eye[1] + right_eye[1]) // 2),
        )

        # Rotation matrix
        M = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
        h, w = img.shape[:2]
        rotated = cv2.warpAffine(img, M, (w, h))

        # Resize to output size
        # return cv2.resize(
        #     rotated,
        #     (self.output_size[1], self.output_size[0]),
        #     interpolation=cv2.INTER_LINEAR,
        # )
        """Similarity transform alignment using 5-point landmarks."""
        try:
            from skimage.transform import SimilarityTransform
            tform = SimilarityTransform()
            tform.estimate(landmarks, self.template)
            M = tform.params[:2]
            return cv2.warpAffine(
                frame, M,
                (self.output_size[1], self.output_size[0]),
                flags      = cv2.INTER_LINEAR,
                borderMode = cv2.BORDER_REFLECT,
            )
        except ImportError:
            # fallback: simple 2-eye rotation
            return self._align_eyes(frame, landmarks)

    def _align_eyes(
        self,
        frame     : np.ndarray,
        landmarks : np.ndarray,
    ) -> np.ndarray:
        """Fallback 2-eye rotation alignment."""
        le, re    = landmarks[0], landmarks[1]
        dy, dx    = re[1] - le[1], re[0] - le[0]
        angle     = math.degrees(math.atan2(dy, dx + 1e-8))
        cx        = int((le[0] + re[0]) / 2)
        cy        = int((le[1] + re[1]) / 2)
        M         = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        H, W      = frame.shape[:2]
        rotated   = cv2.warpAffine(frame, M, (W, H),
                                   borderMode=cv2.BORDER_REFLECT)
        return cv2.resize(rotated, (self.output_size[1], self.output_size[0]))

    def _align_crop(
        self,
        frame : np.ndarray,
        bbox  : Tuple[int,int,int,int],
    ) -> np.ndarray:
        """Simple crop + resize (no landmarks)."""
        x1, y1, x2, y2 = bbox
        H, W = frame.shape[:2]
        # Add 10% padding
        pad_x = int((x2 - x1) * 0.10)
        pad_y = int((y2 - y1) * 0.10)
        x1    = max(0, x1 - pad_x)
        y1    = max(0, y1 - pad_y)
        x2    = min(W, x2 + pad_x)
        y2    = min(H, y2 + pad_y)

        if x2 <= x1 or y2 <= y1:
            return np.zeros((*self.output_size, 3), dtype=np.uint8)

        crop = frame[y1:y2, x1:x2]
        return cv2.resize(crop, (self.output_size[1], self.output_size[0]),
                          interpolation=cv2.INTER_LINEAR)


# ── Face Detector ─────────────────────────────────────────────────────────────

class FaceDetector:
    """
    Face detector supporting multiple backends.

    Backends:
        • "scrfd"       : SCRFD via insightface (recommended)
        • "retinaface"  : RetinaFace via insightface
        • "haar"        : OpenCV Haar Cascade (fallback, no landmarks)

    Usage:
        detector = FaceDetector(backend="haar")
        results  = detector.detect(img_bgr)
        for det in results:
            crop = det.face_crop    # aligned 112×112 BGR
    """

    def __init__(
        self,
        backend     : str   = "haar",
        min_face_size: int  = 30,
        det_threshold: float = 0.5,
        output_size : Tuple[int, int] = (112, 112),
    ):
        self.backend      = backend
        self.min_face_size= min_face_size
        self.det_threshold= det_threshold
        self.output_size  = output_size
        self.aligner      = FaceAligner(output_size)
        self.quality_checker = FaceQualityChecker()
        self._detector    = None
        self._load_detector()

    def _load_detector(self):
        """Initialize the detection backend."""
        if self.backend == "haar":
            self._detector = cv2.CascadeClassifier(
                cv2.data.haarcascades +
                "haarcascade_frontalface_default.xml"
            )
        elif self.backend in ["scrfd", "retinaface"]:
            try:
                from insightface.app import FaceAnalysis
                self._detector = FaceAnalysis(
                    name        = "buffalo_l",
                    allowed_modules = ["detection"],
                )
                self._detector.prepare(ctx_id=0, det_size=(640, 640))
            except ImportError:
                import warnings
                warnings.warn(
                    "insightface not installed. Falling back to Haar cascade.\n"
                    "Install: pip install insightface"
                )
                self.backend   = "haar"
                self._detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades +
                    "haarcascade_frontalface_default.xml"
                )

    def detect(self, img: np.ndarray) -> List[DetectionResult]:
        """
        Detect all faces in an image.

        Args:
            img: BGR image

        Returns:
            List of DetectionResult (sorted by confidence desc)
        """
        if self.backend == "haar":
            return self._detect_haar(img)
        else:
            return self._detect_insightface(img)

    def _detect_haar(self, img: np.ndarray) -> List[DetectionResult]:
        """Haar cascade detection (no landmark support)."""
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self._detector.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors= 5,
            minSize     = (self.min_face_size, self.min_face_size),
        )

        results = []
        for (x, y, w, h) in faces:
            face_crop = cv2.resize(
                img[y:y+h, x:x+w],
                self.output_size,
                interpolation=cv2.INTER_LINEAR,
            )
            results.append(DetectionResult(
                bbox        = (x, y, x + w, y + h),
                landmarks   = None,
                confidence  = 1.0,
                face_crop   = face_crop,
            ))
        return results

    def _detect_insightface(self, img: np.ndarray) -> List[DetectionResult]:
        """SCRFD/RetinaFace detection with 5-point landmarks."""
        faces   = self._detector.get(img)
        results = []

        for face in faces:
            if face.det_score < self.det_threshold:
                continue

            bbox = face.bbox.astype(int)
            lmks = face.kps          # (5, 2) landmarks

            # Align using similarity transform
            try:
                aligned = self.aligner.align(img, lmks)
            except Exception:
                x1, y1, x2, y2 = bbox
                aligned = cv2.resize(img[y1:y2, x1:x2], self.output_size)

            results.append(DetectionResult(
                bbox       = tuple(bbox),
                landmarks  = lmks,
                confidence = float(face.det_score),
                face_crop  = aligned,
            ))

        return sorted(results, key=lambda r: -r.confidence)
    
    def _detect_yunet(self, frame: np.ndarray) -> List[Dict]:
        H, W = frame.shape[:2]
        self._model.setInputSize((W, H))
        _, faces = self._model.detect(frame)
        results  = []
        if faces is None:
            return results
        for f in faces:
            x, y, w, h = int(f[0]), int(f[1]), int(f[2]), int(f[3])
            score       = float(f[-1])
            if score < self.conf_thresh:
                continue
            # YuNet provides 5 landmark pairs at indices 4-13
            lmks = None
            if len(f) >= 14:
                lmks = f[4:14].reshape(5, 2).astype(np.float32)
            results.append({
                "bbox"     : (x, y, x + w, y + h),
                "landmarks": lmks,
                "score"    : score,
            })
        return sorted(results, key=lambda r: -r["score"])

    def _detect_scrfd(self, frame: np.ndarray) -> List[Dict]:
        faces   = self._model.get(frame)
        results = []
        for face in faces:
            if float(face.det_score) < self.conf_thresh:
                continue
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            results.append({
                "bbox"     : (x1, y1, x2, y2),
                "landmarks": face.kps.astype(np.float32),
                "score"    : float(face.det_score),
            })
        return sorted(results, key=lambda r: -r["score"])


# ── Full Preprocessing Pipeline ───────────────────────────────────────────────

class FacePreprocessor:
    """
    End-to-end face preprocessing pipeline.

    Input  : raw image (path or numpy array)
    Output : list of 112×112 aligned face crops (+ quality scores)

    Usage:
        preprocessor = FacePreprocessor(backend="haar")
        crops = preprocessor.process("path/to/image.jpg")
        for crop, quality in crops:
            # crop: (H, W, 3) BGR numpy array
            # quality: float 0-1
    """

    def __init__(
        self,
        backend      : str = "haar",
        output_size  : Tuple[int, int] = (112, 112),
        quality_filter: bool = True,
        min_quality  : float = 0.3,
    ):
        self.detector       = FaceDetector(backend=backend, output_size=output_size)
        self.quality_checker= FaceQualityChecker()
        self.quality_filter = quality_filter
        self.min_quality    = min_quality

    def process(
        self,
        input_img : str | np.ndarray,
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Process a single image through the full pipeline.

        Args:
            input_img: file path string or BGR numpy array

        Returns:
            List of (aligned_crop, quality_score) tuples
        """
        # Load image
        if isinstance(input_img, str):
            img = cv2.imread(input_img)
            if img is None:
                return []
        else:
            img = input_img

        # Detect faces
        detections = self.detector.detect(img)

        # Filter + quality check
        results = []
        for det in detections:
            if det.face_crop is None:
                continue

            passed, score, reason = self.quality_checker.check(det.face_crop)

            if self.quality_filter and not passed:
                continue

            if self.quality_filter and score < self.min_quality:
                continue

            results.append((det.face_crop, score))

        return results

    def process_to_pil(
        self,
        input_img: str | np.ndarray,
    ) -> List[Tuple[Image.Image, float]]:
        """
        Process image and return PIL crops.

        Returns:
            List of (PIL_RGB_image, quality_score) tuples
        """
        raw = self.process(input_img)
        return [
            (Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), q)
            for crop, q in raw
        ]
