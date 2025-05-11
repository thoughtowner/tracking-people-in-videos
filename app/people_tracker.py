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


# def process_video_live(input_path, buffer, stop_event, fps):
#     cap = cv2.VideoCapture(input_path)
#     frame_idx = 1
#
#     while cap.isOpened() and not stop_event.is_set():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # delay = random.uniform(0.1, 0.5)
#         # time.sleep(delay)
#
#         timestamp = frame_idx / fps
#         buffer.append((frame_idx, timestamp, frame.copy()))
#         logging.info(f"Кадр {frame_idx} добавлен в буфер")  # Логирование кадров в буфер
#         frame_idx += 1
#
#     cap.release()
#     logging.info("Обработка видео завершена")

async def process_video_live(input_path, websocket, fps):
    cap = cv2.VideoCapture(input_path)
    frame_idx = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        delay = random.uniform(0.1, 0.5)
        await asyncio.sleep(delay)

        _, jpeg = cv2.imencode(".jpg", frame)
        if not _:
            logger.info("Ошибка кодирования кадра!")
            continue

        try:
            await websocket.send_bytes(jpeg.tobytes())
        except Exception as e:
            logger.info(f"Ошибка отправки кадра: {e}")
            break

        frame_idx += 1

    cap.release()
    logger.info("Обработка видео завершена.")


if __name__ == '__main__':
    process_video('app/resources/videos/input_video_1.mp4', 'app/resources/videos/output_video.mp4')
