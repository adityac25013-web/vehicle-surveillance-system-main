import base64
import time
from typing import Dict, List

import cv2
import numpy as np
from flask import Flask, Response, current_app, jsonify, request, stream_with_context

from .db import crud
from .db.models import VehicleEvent


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
                time.sleep(0.05)

        return Response(
            stream_with_context(generate()),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/events/live")
    def live_events() -> Response:
        processor = current_app.config.get("VIDEO_PROCESSOR")
        if processor is None:
            return jsonify({"events": []})
        events = processor.get_latest_events()
        crud.log_vehicle_events(events)
        return jsonify({"events": events})

    @app.post("/api/process_frame")
    def process_frame() -> Response:
        payload = request.get_json(silent=True) or {}
        image_data = payload.get("image", "")
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

        pipeline = current_app.config["DETECTION_PIPELINE"]
        events, vis_frame = pipeline.process_frame(frame)
        crud.log_vehicle_events(events)

        ok, jpeg = cv2.imencode(".jpg", vis_frame)
        annotated = ""
        if ok:
            annotated = "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode("ascii")

        return jsonify({"events": events, "annotated_image": annotated})

    @app.get("/api/events/recent")
    def recent_events() -> Response:
        items: List[VehicleEvent] = crud.get_recent_events(limit=50)
        return jsonify({"events": [_serialize_event(e) for e in items]})
