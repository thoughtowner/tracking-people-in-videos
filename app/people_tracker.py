import torch
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import time
import asyncio

import logging.config
from .logger import LOGGING_CONFIG, logger

BLINK_PHASES = 8
BLINK_FRAME_PER_COLOR = 1
MAIN_COLOR = (225, 41, 84)
PATH_COLOR = (255, 43, 36)


def process_video(input_path: str, output_path: str):
    logging.config.dictConfig(LOGGING_CONFIG)
    logger.info('Начало работы...')

    model = YOLO('yolov8s.pt')
    tracker = DeepSort(max_age=10)

    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )

    object_paths: dict[int, list[tuple[int, int]]] = {}
    object_detection_frames: dict[int, int] = {}
    tracking_results = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        person_detections = []
        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            if int(cls) == 0:
                x1, y1, x2, y2 = map(int, box)
                bbox = [x1, y1, x2 - x1, y2 - y1]
                person_detections.append(([bbox, float(conf), 'person']))

        tracks = tracker.update_tracks(person_detections, frame=frame)
        frame_data = {"frame": frame_count, "objects": []}

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            frame_data["objects"].append({
                "id": track_id,
                "bbox": [int(l), int(t), int(r), int(b)]
            })

            center_x = int((l + r) / 2)
            center_y = int((t + b) / 2)

            if track_id not in object_detection_frames:
                object_detection_frames[track_id] = frame_count

            frames_since_first = frame_count - object_detection_frames[track_id]
            phase = frames_since_first // BLINK_FRAME_PER_COLOR

            if phase < BLINK_PHASES * 2:
                color = (0, 0, 255) if phase % 2 == 0 else (0, 255, 0)
            else:
                color = MAIN_COLOR

            object_paths.setdefault(track_id, []).append((center_x, center_y))
            if len(object_paths[track_id]) > 100:
                object_paths[track_id].pop(0)

            for x, y in object_paths[track_id]:
                cv2.circle(frame, (x, y), 4, PATH_COLOR, -1)

            overlay = frame.copy()
            alpha = 0.4
            cv2.rectangle(overlay, (int(l), int(t)), (int(r), int(b)), color, -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), MAIN_COLOR, 2)
            cv2.putText(frame, f'ID: {track_id}', (int(l), int(t - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, MAIN_COLOR, 2)

        tracking_results.append(frame_data)
        out.write(frame)
        logger.info(f'Обработано {frame_count} (из {total_frames}) кадров')
        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logger.info('Обработка завершена!')

    return tracking_results


async def process_video_live(input_path: str, websocket, fps: int, client_id: str):
    from .main import clients

    model = YOLO('yolov8s.pt')
    tracker = DeepSort(max_age=10)

    cap = cv2.VideoCapture(input_path)
    frame_idx = 1
    client = clients[client_id]
    last_frame_time = time.time()
    frame_interval = 1 / fps

    object_paths: dict[int, list[tuple[int, int]]] = {}
    object_detection_frames: dict[int, int] = {}
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        if not client["playing_synchronized"]:
            logger.info(f"Остановка воспроизведения для клиента {client_id}")
            break

        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        person_detections = []
        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            if int(cls) == 0:
                x1, y1, x2, y2 = map(int, box)
                bbox = [x1, y1, x2 - x1, y2 - y1]
                person_detections.append(([bbox, float(conf), 'person']))

        tracks = tracker.update_tracks(person_detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()

            center_x = int((l + r) / 2)
            center_y = int((t + b) / 2)

            if track_id not in object_detection_frames:
                object_detection_frames[track_id] = frame_idx

            frames_since_first = frame_idx - object_detection_frames[track_id]
            phase = frames_since_first // BLINK_FRAME_PER_COLOR

            if phase < BLINK_PHASES * 2:
                color = (0, 0, 255) if phase % 2 == 0 else (0, 255, 0)
            else:
                color = MAIN_COLOR

            object_paths.setdefault(track_id, []).append((center_x, center_y))
            if len(object_paths[track_id]) > 100:
                object_paths[track_id].pop(0)

            for x, y in object_paths[track_id]:
                cv2.circle(frame, (x, y), 4, PATH_COLOR, -1)

            overlay = frame.copy()
            alpha = 0.4
            cv2.rectangle(overlay, (int(l), int(t)), (int(r), int(b)), color, -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), MAIN_COLOR, 2)
            cv2.putText(frame, f'ID: {track_id}', (int(l), int(t - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, MAIN_COLOR, 2)

        logger.info(f'Обработано {frame_idx} (из {total_frames}) кадров')

        current_time = time.time()
        elapsed_time = current_time - last_frame_time

        if elapsed_time < frame_interval:
            await asyncio.sleep(frame_interval - elapsed_time)
        else:
            success, jpeg = cv2.imencode('.jpg', frame)
            if not success:
                logger.info("Ошибка кодирования кадра!")
                continue

            try:
                await websocket.send_bytes(jpeg.tobytes())
            except Exception as e:
                logger.info(f"Ошибка отправки кадра: {e}")
                break

            last_frame_time = time.time()
            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Обработка видео завершена.")


if __name__ == '__main__':
    process_video('app/resources/videos/input_video_1.mp4', 'app/resources/videos/output_video.mp4')
