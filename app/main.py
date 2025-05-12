from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import uuid
import os
import time
import threading
import asyncio

from .people_tracker import process_video, process_video_live
import logging.config
from .logger import LOGGING_CONFIG, logger

logging.config.dictConfig(LOGGING_CONFIG)
logger.info("Запуск приложения")

app = FastAPI()
clients: dict[str, dict] = {}
tracking_data_store: dict[str, list] = {}

app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
def root():
    with open("app/static/index.html", encoding="utf-8") as f:
        return f.read()

@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    input_dir = Path("resources/videos")
    input_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / f"input_{uuid.uuid4().hex}.mp4"

    with open(input_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    video_id = input_path.name.replace("input_", "").replace(".mp4", "")
    clients[video_id] = {"input_path": str(input_path), "fps": 30, "playing_synchronized": True}
    tracking_data_store[video_id] = []

    return {"video_id": video_id}

@app.post("/process/{video_id}")
def process_for_download(video_id: str):
    client = clients.get(video_id)
    if not client:
        raise HTTPException(status_code=404, detail="Video not found")

    input_path = client["input_path"]
    output_dir = Path("resources/videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"output_{video_id}.mp4"

    detections = process_video(input_path, str(output_path))
    tracking_data_store[video_id] = detections

    return {"video_url": f"/video/{video_id}", "detections_url": f"/detections/{video_id}"}

@app.get("/video/{video_id}")
def get_processed_video(video_id: str):
    output_path = Path(f"resources/videos/output_{video_id}.mp4")
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Processed video not found")
    return FileResponse(str(output_path), media_type="video/mp4", filename="processed_video.mp4")

@app.get("/detections/{video_id}")
def get_detections(video_id: str):
    data = tracking_data_store.get(video_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Detections not found")
    return data

@app.websocket("/ws/{video_id}")
async def websocket_endpoint(websocket: WebSocket, video_id: str):
    await websocket.accept()
    client = clients.get(video_id)
    if not client:
        await websocket.close()
        return

    await main_process_video(client["input_path"], video_id, websocket, client["fps"])
    await websocket.close()
    logger.info("WebSocket закрыт.")

async def main_process_video(input_path: str, client_id: str, websocket: WebSocket, fps=30):
    logger.info(f"Запуск live-обработки для {client_id}")
    await process_video_live(input_path, websocket, fps, client_id)
    logger.info("Live-обработка завершена")

@app.post("/toggle_playing_mode")
async def toggle_playing_mode(data: dict):
    client_id = data.get("client_id")
    mode = data.get("mode")
    client = clients.get(client_id)
    if not client or mode not in ("sync", "stop"):
        raise HTTPException(status_code=400, detail="Invalid request parameters")
    client["playing_synchronized"] = (mode == "sync")
    return {"status": "success"}
