from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

VEHICLE_CLASS_NAMES = {"car", "motorbike", "motorcycle", "bus", "truck", "bicycle"}
# When user shows a phone screen with car/plate image, run OCR on phone bbox
PHONE_CLASS_NAME = "cell phone"

# Common Uttar Pradesh (India) RTO region codes, e.g. UP16 (Noida), UP32 (Lucknow)
UP_RTO_CODES = {
    "UP11", "UP12", "UP13", "UP14", "UP15", "UP16", "UP17", "UP18", "UP19", "UP20",
    "UP21", "UP22", "UP23", "UP24", "UP25", "UP26", "UP27", "UP30", "UP31", "UP32",
    "UP33", "UP34", "UP35", "UP36", "UP37", "UP38", "UP40", "UP41", "UP42", "UP43",
    "UP44", "UP45", "UP46", "UP47", "UP50", "UP51", "UP52", "UP53", "UP54", "UP55",
    "UP56", "UP57", "UP58", "UP60", "UP61", "UP62", "UP63", "UP64", "UP65", "UP66",
    "UP67", "UP68", "UP70", "UP71", "UP72", "UP73", "UP74", "UP75", "UP76", "UP77",
    "UP78", "UP79", "UP80", "UP81", "UP82", "UP83", "UP84", "UP85"
}


def _clean_plate_text(text: str) -> str:
    """Keep only alphanumeric (and common plate chars), uppercase."""
    if not text or not text.strip():
        return ""
    s = re.sub(r"[^A-Za-z0-9]", "", text.strip().upper())
    return s if 2 <= len(s) <= 15 else ""


def _normalize_plate(text: str) -> str:
    """
    Normalize OCR output by removing spaces and dashes and uppercasing.
    """
    if not text:
        return ""
    return re.sub(r"[-\s]", "", text.strip().upper())


def _normalize_up_plate(text: str) -> str:
    """
    Try to normalize OCR output into a valid Uttar Pradesh plate string.
    Handles common confusions like I/L -> 1, O -> 0 in digit positions.
    Returns normalized plate (e.g. 'UP14EH0701') or '' if it cannot be fixed.
    """
    cleaned = _clean_plate_text(text)
    if not cleaned or not cleaned.startswith("UP"):
        return ""

    normalized = _normalize_plate(cleaned)
    # Basic length guard
    if len(normalized) < 8 or len(normalized) > 12:
        return ""

    # First 4 chars must be a known UP RTO code like UP16, UP32, etc.
    region = normalized[:4]
    if region not in UP_RTO_CODES:
        return ""

    pattern = r"UP[0-9]{2}[A-Z]{1,3}[0-9]{4}"

    # If it's already a clean match, keep as is.
    if re.fullmatch(pattern, normalized):
        return normalized

    # Try to auto-correct obvious character confusions in numeric positions.
    chars = list(normalized)
    # Numeric positions: RTO digits (2, 3) and the last 4 digits
    numeric_positions = {2, 3, len(chars) - 4, len(chars) - 3, len(chars) - 2, len(chars) - 1}
    # Map of common OCR confusions (in numeric positions)
    char_to_digit = {
        "O": "0",
        "D": "0",
        "Q": "0",
        "I": "1",
        # 'L' on UP plates is often mis-read where '4' should be
        "L": "4",
        "Z": "2",
        "S": "5",
        "B": "8",
        "G": "6",
    }

    for idx in numeric_positions:
        if 0 <= idx < len(chars):
            ch = chars[idx]
            if ch in char_to_digit:
                chars[idx] = char_to_digit[ch]

    candidate = "".join(chars)
    if re.fullmatch(pattern, candidate) and candidate[:4] in UP_RTO_CODES:
        return candidate

    return ""


def _is_valid_up_plate(text: str) -> bool:
    return bool(_normalize_up_plate(text))


def _enhance_plate_crop(crop: np.ndarray) -> np.ndarray:
    """
    Apply simple preprocessing to make plate characters sharper for OCR:
    denoise + contrast + binary threshold.
    """
    if crop.size == 0:
        return crop
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop

    # Mild denoising while preserving edges
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)

    # Binary threshold (Otsu) to highlight digits/letters
    _, th = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def _tesseract_read_plate(crop: np.ndarray) -> str:
    """
    Fallback OCR using Tesseract with a whitelist tuned for plates.
    Returns normalized text if possible, otherwise cleaned text.
    """
    if pytesseract is None:
        return ""
    try:
        enhanced = _enhance_plate_crop(crop)
        text = pytesseract.image_to_string(
            enhanced,
            config="--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        )
        cleaned = _clean_plate_text(text)
        if not cleaned:
            return ""
        normalized_up = _normalize_up_plate(cleaned)
        if normalized_up:
            return normalized_up
        return cleaned
    except Exception:
        return ""


def _read_text_from_region(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, reader: Any, min_conf: float = 0.25) -> str:
    """Run OCR on a full region (e.g. phone screen) and return best plate-like text."""
    h, w = frame.shape[:2]
    crop_x1 = max(0, x1)
    crop_x2 = min(w, x2)
    crop_y1 = max(0, y1)
    crop_y2 = min(h, y2)
    if crop_x2 - crop_x1 < 20 or crop_y2 - crop_y1 < 15:
        return ""
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return ""
    ch, cw = crop.shape[:2]
    if ch < 30 or cw < 60:
        scale = max(2, 60 // cw, 30 // ch)
        crop = cv2.resize(crop, (cw * scale, ch * scale), interpolation=cv2.INTER_CUBIC)
    try:
        results = reader.readtext(crop)
        best_fallback = ""
        for (_bbox, text, conf) in (results or []):
            if conf < min_conf:
                continue
            cleaned = _clean_plate_text(text)
            if not cleaned:
                continue
            # Prefer plates that match or can be auto-corrected to a Uttar Pradesh format
            normalized_up = _normalize_up_plate(cleaned)
            if normalized_up:
                return normalized_up
            # Remember some cleaned text as a fallback (for non-UP or close matches)
            if not best_fallback:
                best_fallback = cleaned
        # If EasyOCR didn't give a clear UP plate, try Tesseract on the same region
        tess_text = _tesseract_read_plate(crop)
        if tess_text:
            return tess_text
        if best_fallback:
            return best_fallback
    except Exception:
        # If EasyOCR fails, still try Tesseract as a backup
        tess_text = _tesseract_read_plate(crop)
        if tess_text:
            return tess_text
    return ""


def _read_plate_from_crop(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, reader: Any) -> str:
    """Crop lower part of vehicle bbox (plate region) and run OCR. Handles small bboxes (e.g. car on phone)."""
    h, w = frame.shape[:2]
    if x2 <= x1 or y2 <= y1:
        return ""
    box_h = y2 - y1
    box_w = x2 - x1
    # Plate is usually in lower 25–45% of vehicle; for small boxes use full lower half
    crop_y1 = max(0, y2 - int(box_h * 0.5))
    crop_y2 = y2
    crop_x1 = max(0, x1)
    crop_x2 = min(w, x2)
    crop_h = crop_y2 - crop_y1
    crop_w = crop_x2 - crop_x1
    if crop_h < 12 or crop_w < 24:
        return ""
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return ""
    # Upscale small crops so OCR can read better (e.g. car on phone screen)
    if crop_h < 40 or crop_w < 80:
        scale = max(2, 40 // crop_h, 80 // crop_w)
        crop = cv2.resize(crop, (crop_w * scale, crop_h * scale), interpolation=cv2.INTER_CUBIC)
    try:
        results = reader.readtext(crop)
        best_fallback = ""
        for (_bbox, text, conf) in (results or []):
            if conf < 0.25:
                continue
            cleaned = _clean_plate_text(text)
            if not cleaned:
                continue
            normalized_up = _normalize_up_plate(cleaned)
            if normalized_up:
                return normalized_up
            if not best_fallback:
                best_fallback = cleaned
        # Try a Tesseract pass if EasyOCR didn't give a strong UP-format plate
        tess_text = _tesseract_read_plate(crop)
        if tess_text:
            return tess_text
        if best_fallback:
            return best_fallback
    except Exception:
        # EasyOCR failure – fall back to Tesseract if available
        tess_text = _tesseract_read_plate(crop)
        if tess_text:
            return tess_text
    return ""


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class DetectionPipeline:
    def __init__(self) -> None:
        self._model = YOLO("yolov8n.pt")
        self._tracker = DeepSort(max_age=30, n_init=3)
        self._ocr_reader: Optional[Any] = None

    def _get_ocr_reader(self) -> Optional[Any]:
        if easyocr is None:
            return None
        if self._ocr_reader is None:
            try:
                self._ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            except Exception:
                self._ocr_reader = False  # type: ignore
        return self._ocr_reader if self._ocr_reader else None

    def process_frame(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Lower conf (0.2) so cars on phone screens / smaller in frame still get detected
        yolo_results = self._model(rgb, imgsz=640, conf=0.2, verbose=False)[0]

        detections: List[Detection] = []
        if yolo_results.boxes is not None and len(yolo_results.boxes) > 0:
            boxes_xyxy = yolo_results.boxes.xyxy.cpu().numpy()
            confs = yolo_results.boxes.conf.cpu().numpy()
            class_ids = yolo_results.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), conf, cid in zip(boxes_xyxy, confs, class_ids):
                name = self._model.names.get(int(cid), "object")
                if name not in VEHICLE_CLASS_NAMES:
                    continue
                detections.append(
                    Detection(bbox=(x1, y1, x2, y2), confidence=float(conf), class_id=int(cid), class_name=name)
                )

        # DeepSort expects each detection as ([left, top, width, height], confidence, class_name)
        tracker_inputs = []
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            w = max(0.0, float(x2 - x1))
            h = max(0.0, float(y2 - y1))
            tracker_inputs.append(([float(x1), float(y1), w, h], float(d.confidence), d.class_name))
        tracks = self._tracker.update_tracks(tracker_inputs, frame=frame)

        timestamp = datetime.now(timezone.utc).isoformat()
        events: List[Dict[str, Any]] = []
        vis_frame = frame.copy()
        ocr_reader = self._get_ocr_reader()

        # If user holds a phone showing car/plate, detect phone and run OCR on full screen
        if ocr_reader and yolo_results.boxes is not None and len(yolo_results.boxes) > 0:
            boxes_xyxy = yolo_results.boxes.xyxy.cpu().numpy()
            confs = yolo_results.boxes.conf.cpu().numpy()
            class_ids = yolo_results.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), conf, cid in zip(boxes_xyxy, confs, class_ids):
                name = self._model.names.get(int(cid), "object")
                if name != PHONE_CLASS_NAME:
                    continue
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                plate_text = _read_text_from_region(frame, x1, y1, x2, y2, ocr_reader)
                if plate_text:
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                    cv2.putText(vis_frame, f"phone: {plate_text}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 165, 0), 1, cv2.LINE_AA)
                    cv2.putText(vis_frame, plate_text, (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
                    events.append({
                        "track_id": 0,
                        "label": "plate",
                        "confidence": float(conf),
                        "bbox": [x1, y1, x2, y2],
                        "license_plate": plate_text,
                        "camera": "webcam-0",
                        "timestamp": timestamp,
                    })

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            label = track.det_class or "vehicle"

            license_plate = ""
            if ocr_reader:
                license_plate = _read_plate_from_crop(frame, x1, y1, x2, y2, ocr_reader)

            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis_frame,
                f"{label} #{track_id}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            if license_plate:
                cv2.putText(
                    vis_frame,
                    license_plate,
                    (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            events.append(
                {
                    "track_id": int(track_id),
                    "label": label,
                    "confidence": float(track.det_conf or 0.0),
                    "bbox": [x1, y1, x2, y2],
                    "license_plate": license_plate,
                    "camera": "webcam-0",
                    "timestamp": timestamp,
                }
            )

        # Fallback: if no confirmed tracks produced events but YOLO saw vehicles,
        # emit simple per-detection events so the frontend never stays empty.
        if not events and detections:
            for d in detections:
                x1, y1, x2, y2 = map(int, d.bbox)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    vis_frame,
                    f"{d.class_name}",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                events.append(
                    {
                        "track_id": 0,
                        "label": d.class_name,
                        "confidence": float(d.confidence),
                        "bbox": [x1, y1, x2, y2],
                        "license_plate": "",
                        "camera": "webcam-0",
                        "timestamp": timestamp,
                    }
                )

        return events, vis_frame

