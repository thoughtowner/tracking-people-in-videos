from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import cv2
import time
import threading
import shutil
import uuid
import os
import asyncio
from .people_tracker import process_video, process_video_live

import logging.config
from .logger import LOGGING_CONFIG, logger


logging.config.dictConfig(LOGGING_CONFIG)
logger.info("Запуск приложения")

app = FastAPI()

clients = {}
tracking_data_store = {}

app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
def root():
    with open("app/static/index.html") as f:
        return f.read()

@app.get("/video")
def get_processed_video(video_id: str):
    output_filename = f"resources/videos/output_{video_id}.mp4"
    if not os.path.exists(output_filename):
        return JSONResponse(status_code=404, content={"detail": "Video not found"})
    return FileResponse(output_filename, media_type="video/mp4", filename="processed_video.mp4")

@app.get("/detections")
def get_detections(video_id: str):
    if video_id not in tracking_data_store:
        return JSONResponse(status_code=404, content={"detail": "Detections not found"})
    return tracking_data_store[video_id]

@app.post("/track")
async def track_people(video: UploadFile = File(...)):
    input_filename = f"resources/videos/input_{uuid.uuid4().hex}.mp4"
    output_filename = input_filename.replace("input_", "output_")

    os.makedirs("resources/videos", exist_ok=True)

    with open(input_filename, "wb") as f:
        shutil.copyfileobj(video.file, f)

    video_id = os.path.basename(output_filename).replace("output_", "").replace(".mp4", "")
    tracking_data_store[video_id] = []

    clients[video_id] = {
        "input_path": input_filename,
        "fps": 30,
        "playing_synchronized": True
    }

    return {
        "video_id": video_id,
        "video_url": f"/video?video_id={video_id}",
        "detections_url": f"/detections?video_id={video_id}"
    }

async def main_process_video(input_path: str, client_id: str, websocket: WebSocket, fps=30):
    buffer = []
    stop_event = threading.Event()

    clients[client_id] = {
        "buffer": buffer,
        "stop_event": stop_event,
        "start_time": time.time(),
        "fps": fps,
        "input_path": input_path,
        "websocket": websocket,
        "playing_synchronized": True
    }

    logger.info("Запуск обработки видео")
    await process_video_live(input_path, websocket, fps, client_id)
    logger.info("Процесс обработки видео завершен")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()

    client = clients.get(client_id)
    if not client:
        await websocket.close()
        return

    input_path = client["input_path"]
    fps = client["fps"]

    await main_process_video(input_path, client_id, websocket, fps)
    await websocket.close()
    logger.info("WebSocket закрыт.")

@app.post("/toggle_playing_mode")
async def toggle_playing_mode(data: dict):
    client_id = data.get("client_id")
    mode = data.get("mode")

    if not client_id or not mode:
        raise HTTPException(status_code=400, detail="Invalid request parameters")

    if client_id in clients:
        clients[client_id]["playing_synchronized"] = (mode == "sync")
        return {"status": "success"}
    else:
        raise HTTPException(status_code=404, detail="Client not found")
