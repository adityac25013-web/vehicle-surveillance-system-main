import base64
import time
from collections import Counter, defaultdict, deque
from typing import Dict, List

import cv2
import numpy as np
from flask import Flask, Response, current_app, jsonify, request, stream_with_context

from .db import crud
from .detection import DetectionPipeline, detect_plate_text_mobile_fallback
from .db.models import VehicleEvent

_mobile_plate_history = defaultdict(lambda: deque(maxlen=12))


def _normalize_plate_token(text: str) -> str:
    if not text:
        return ""
    token = "".join(ch for ch in text.upper() if ch.isalnum())
    if len(token) < 8 or len(token) > 12:
        return ""
    letters = sum(ch.isalpha() for ch in token)
    digits = sum(ch.isdigit() for ch in token)
    if letters < 2 or digits < 3:
        return ""
    return token


def _stabilize_mobile_plate(source: str, plate_text: str) -> str:
    token = _normalize_plate_token(plate_text)
    if not token:
        return ""
    history = _mobile_plate_history[source]
    history.append(token)
    counts = Counter(history)
    best, freq = counts.most_common(1)[0]
    # Use majority when available to reduce OCR jitter in mobile stream.
    return best if freq >= 2 else token


def _serialize_event(e: VehicleEvent) -> Dict:
    return {
        "id": e.id,
        "track_id": e.track_id,
        "label": e.label,
        "confidence": e.confidence,
        "license_plate": e.license_plate,
        "camera": e.camera,
        "timestamp": e.created_at.isoformat() if e.created_at else None,
    }


def register_routes(app: Flask) -> None:
    @app.get("/api/health")
    def health() -> Response:
        return jsonify({"status": "ok"})

    @app.get("/api/status")
    def status() -> Response:
        processor = current_app.config.get("VIDEO_PROCESSOR")
        if processor is None:
            return jsonify(
                {
                    "backend": "ok",
                    "camera": "unavailable",
                    "model": "ready",
                    "mode": "normal",
                    "stats": {"fps": 0.0, "detection_ms": 0.0},
                }
            )
        stats = processor.get_stats()
        camera_state = "ok" if stats.get("camera_failures", 0) < 10 else "degraded"
        return jsonify(
            {
                "backend": "ok",
                "camera": camera_state,
                "model": "ready",
                "mode": processor.get_runtime_mode(),
                "stats": stats,
            }
        )

    @app.get("/api/demo_mode")
    def get_demo_mode() -> Response:
        processor = current_app.config.get("VIDEO_PROCESSOR")
        mode = "normal"
        if processor is not None:
            mode = processor.get_runtime_mode()
        return jsonify({"mode": mode})

    @app.post("/api/demo_mode")
    def set_demo_mode() -> Response:
        payload = request.get_json(silent=True) or {}
        mode = str(payload.get("mode", "normal")).strip().lower()
        if mode not in {"normal", "smooth", "accurate"}:
            return jsonify({"error": "mode must be one of: normal, smooth, accurate"}), 400
        processor = current_app.config.get("VIDEO_PROCESSOR")
        pipeline = current_app.config.get("DETECTION_PIPELINE")
        if processor is not None:
            processor.set_runtime_mode(mode)
        elif pipeline is not None:
            pipeline.set_runtime_profile(mode)
        return jsonify({"mode": mode})

    @app.post("/api/reset_demo")
    def reset_demo() -> Response:
        processor = current_app.config.get("VIDEO_PROCESSOR")
        if processor is not None:
            processor.reset_demo_state()
        return jsonify({"ok": True})

    @app.get("/video_feed")
    def video_feed() -> Response:
        processor = current_app.config.get("VIDEO_PROCESSOR")
        if processor is None:
            return jsonify(
                {
                    "error": "Local video feed is unavailable in cloud mode. Use /api/process_frame from browser camera."
                }
            ), 400

        def generate():
            while True:
                frame = processor.get_jpeg_frame()
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                time.sleep(0.03)

        return Response(
            stream_with_context(generate()),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/events/live")
    def live_events() -> Response:
        processor = current_app.config.get("VIDEO_PROCESSOR")
        if processor is None:
            return jsonify({"events": [], "stats": {"fps": 0.0, "detection_ms": 0.0}})
        events = processor.get_latest_events()
        crud.log_vehicle_events(events)
        return jsonify({"events": events, "stats": processor.get_stats()})

    @app.post("/api/process_frame")
    def process_frame() -> Response:
        started_at = time.perf_counter()
        payload = request.get_json(silent=True) or {}
        image_data = payload.get("image", "")
        source = str(payload.get("source", "browser-camera")).strip() or "browser-camera"
        if not isinstance(image_data, str) or not image_data:
            return jsonify({"error": "Missing image payload."}), 400

        try:
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]
            binary = base64.b64decode(image_data)
            arr = np.frombuffer(binary, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return jsonify({"error": "Unable to decode image."}), 400
        except Exception:
            return jsonify({"error": "Invalid base64 image."}), 400

        # Lazy-load model pipeline on first frame request, so health endpoint stays fast.
        pipeline = current_app.config.get("DETECTION_PIPELINE")
        if pipeline is None:
            pipeline = DetectionPipeline()
            current_app.config["DETECTION_PIPELINE"] = pipeline
        events, vis_frame = pipeline.process_frame(frame)
        for e in events:
            e["camera"] = source

        if source.startswith("mobile"):
            plate_text = detect_plate_text_mobile_fallback(frame)
            stable_plate = _stabilize_mobile_plate(source, plate_text)
            # If detector missed entirely, create a fallback event.
            if stable_plate and not events:
                h, w = frame.shape[:2]
                events.append(
                    {
                        "track_id": 0,
                        "label": "plate",
                        "confidence": 0.55,
                        "bbox": [0, 0, w, h],
                        "license_plate": stable_plate,
                        "camera": source,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                )
            # If we have vehicle events but no plate text, enrich first vehicle-like event.
            elif stable_plate:
                for e in events:
                    if e.get("label") in {"car", "truck", "bus", "vehicle", "motorbike", "motorcycle"}:
                        if not (e.get("license_plate") or "").strip():
                            e["license_plate"] = stable_plate
                        break

        crud.log_vehicle_events(events)
        processing_ms = (time.perf_counter() - started_at) * 1000.0

        ok, jpeg = cv2.imencode(".jpg", vis_frame)
        annotated = ""
        if ok:
            annotated = "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode("ascii")

        return jsonify(
            {
                "events": events,
                "annotated_image": annotated,
                "stats": {"processing_ms": round(processing_ms, 1)},
            }
        )

    @app.get("/api/events/recent")
    def recent_events() -> Response:
        items: List[VehicleEvent] = crud.get_recent_events(limit=50)
        return jsonify({"events": [_serialize_event(e) for e in items]})
