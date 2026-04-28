from app import create_app
from app.video_stream import VideoProcessor


def main() -> None:
    processor = VideoProcessor(camera_index=0)
    processor.start()
    processor.wait_for_first_frame(timeout=5.0)

    app = create_app()
    app.config["VIDEO_PROCESSOR"] = processor
    # Reuse same detector instance to avoid loading YOLO twice in local mode.
    app.config["DETECTION_PIPELINE"] = processor._pipeline

    try:
        app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
    finally:
        processor.stop()


if __name__ == "__main__":
    main()

