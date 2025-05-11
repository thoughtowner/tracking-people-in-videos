from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import shutil
import uuid
import os
from .people_tracker import process_video

app = FastAPI()

# Смонтировать статику на /static, а не на /
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
def root():
    with open("app/static/index.html") as f:
        return f.read()

tracking_data_store = {}

@app.post("/track")
async def track_people(video: UploadFile = File(...)):
    input_filename = f"resources/videos/input_{uuid.uuid4().hex}.mp4"
    output_filename = input_filename.replace("input_", "output_")

    os.makedirs("resources/videos", exist_ok=True)

    with open(input_filename, "wb") as f:
        shutil.copyfileobj(video.file, f)

    detections = process_video(input_filename, output_filename)

    video_id = os.path.basename(output_filename).replace("output_", "").replace(".mp4", "")
    tracking_data_store[video_id] = detections

    return {
        "video_id": video_id,
        "video_url": f"/video?video_id={video_id}",
        "detections_url": f"/detections?video_id={video_id}"
    }

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
