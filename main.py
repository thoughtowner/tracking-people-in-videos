import torch
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from yolov5 import utils
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadStreams, LoadImages
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device

# Загрузка модели YOLOv5
device = select_device('')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [0]  # только люди (class 0 в COCO)

# Инициализация DeepSORT
tracker = DeepSort(max_age=30)

# Входной видеофайл
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0]  # координаты [x1, y1, x2, y2, conf, class]

    # Подготовим bbox для DeepSORT
    person_detections = []
    for *box, conf, cls in detections:
        if int(cls) == 0:  # только люди
            x1, y1, x2, y2 = map(int, box)
            bbox = [x1, y1, x2 - x1, y2 - y1]  # [x, y, w, h]
            person_detections.append(([bbox, conf.item(), 'person']))

    # Обновление трекера
    tracks = tracker.update_tracks(person_detections, frame=frame)

    # Визуализация
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, w, h = track.to_ltrb()
        cv2.rectangle(frame, (int(l), int(t)), (int(l + w), int(t + h)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (int(l), int(t - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
