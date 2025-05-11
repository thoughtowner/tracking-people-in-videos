import asyncio
import shutil
import cv2
import random
import time


import logging.config
from .logger import LOGGING_CONFIG, logger


logging.config.dictConfig(LOGGING_CONFIG)

def process_video(input_path, output_path):
    shutil.copyfile(input_path, output_path)

    dummy_detections = [
        {"frame": 1, "objects": [{"id": 1, "bbox": [100, 150, 200, 300]}]},
        {"frame": 2, "objects": [{"id": 1, "bbox": [105, 152, 205, 302]}]},
    ]
    return dummy_detections


async def process_video_live(input_path, websocket, fps, client_id):
    from .main import clients
    cap = cv2.VideoCapture(input_path)
    frame_idx = 1
    client = clients[client_id]

    last_frame_time = time.time()
    frame_interval = 1 / fps

    while cap.isOpened():
        if not client["playing_synchronized"]:
            logger.info(f"Остановка воспроизведения для клиента {client_id}")
            break

        ret, frame = cap.read()
        if not ret:
            break

        delay = random.uniform(1, 2)
        await asyncio.sleep(delay)

        current_time = time.time()
        elapsed_time = current_time - last_frame_time

        if elapsed_time < frame_interval:
            await asyncio.sleep(frame_interval - elapsed_time)
        else:
            skipped_frames = int((elapsed_time - frame_interval) / frame_interval)
            for _ in range(skipped_frames):
                ret, _ = cap.read()
                if not ret:
                    break

            _, jpeg = cv2.imencode(".jpg", frame)
            if not _:
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
    logger.info("Обработка видео завершена.")


if __name__ == '__main__':
    process_video('app/resources/videos/input_video_1.mp4', 'app/resources/videos/output_video.mp4')
