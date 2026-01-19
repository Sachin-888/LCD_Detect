# from fastapi import FastAPI, Request
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.templating import Jinja2Templates
# from flask import Flask
# from ultralytics import YOLO
# import numpy as np
# import cv2
# import base64
# from flask_cors import CORS

# app = FastAPI()


# templates = Jinja2Templates(directory="lcd_detec2//templates")

# # ===== LOAD YOLOv8 =====
# model = YOLO("yolov8n.pt")  # nano = fastest

# LCD_CLASSES = {"tv", "monitor"}

# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})


# @app.post("/detect")
# async def detect(payload: dict):
#     image_data = payload["image"].split(",")[1]
#     img_bytes = base64.b64decode(image_data)

#     np_arr = np.frombuffer(img_bytes, np.uint8)
#     frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#     results = model(frame, conf=0.5, verbose=False)

#     lcd_detected = False

#     for r in results:
#         for box in r.boxes:
#             cls_id = int(box.cls[0])
#             label = model.names[cls_id]

#             if label in LCD_CLASSES:
#                 lcd_detected = True

#     return JSONResponse({"lcd_detected": lcd_detected})




# # if __name__ == "__main__":
# #     import uvicorn
# #     uvicorn.run(app, host=" 0.0.0.0", port=5000)
# import nest_asyncio
# import uvicorn

# nest_asyncio.apply()

# uvicorn.run(
#     "main:app",          # <-- filename:variable
#     host="0.0.0.0",     # <-- NO http://
#     port=5000,
#     reload=False
# )















from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import numpy as np
import cv2
import base64
import time
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ================== GLOBALS ==================
latest_frame = None
lcd_detected = False

os.makedirs("captures", exist_ok=True)

# ================== YOLO LOAD ==================
model = YOLO("yolo26n.pt")
LCD_CLASSES = {"monitor", "tv", "laptop"}

# ================== ROI CONFIG ==================
ROI_MARGIN = 40
CONF_THRESHOLD = 0.5
MIN_LCD_AREA_RATIO = 0.12
MIN_ASPECT = 1
MAX_ASPECT = 3.5


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/detect")
async def detect(payload: dict):
    global latest_frame, lcd_detected

    image_data = payload["image"].split(",")[1]
    frame = cv2.imdecode(
        np.frombuffer(base64.b64decode(image_data), np.uint8),
        cv2.IMREAD_COLOR
    )

    frame = cv2.resize(frame, (640, 480))
    h, w = frame.shape[:2]

    # ROI
    roi_x1, roi_y1 = ROI_MARGIN, ROI_MARGIN
    roi_x2, roi_y2 = w - ROI_MARGIN, h - ROI_MARGIN

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
    is_lcd = 0

    for box in results.boxes:
        label = model.names[int(box.cls[0])]
        conf = float(box.conf[0])

        if label not in LCD_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Validation checks
        if x1 < roi_x1 or y1 < roi_y1 or x2 > roi_x2 or y2 > roi_y2:
            continue

        bw, bh = x2 - x1, y2 - y1
        area_ratio = (bw * bh) / (w * h)
        aspect = bw / bh

        if area_ratio < MIN_LCD_AREA_RATIO:
            continue
        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            continue

        # ✅ VALID LCD
        is_lcd = 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"FULL {label.upper()} {int(conf*100)}%",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    lcd_detected = bool(is_lcd)
    latest_frame = frame.copy()

    # 👇 THIS IS IMPORTANT FOR FRONTEND
    return JSONResponse({"lcd_detected": lcd_detected})


@app.post("/capture")
async def capture():
    if not lcd_detected or latest_frame is None:
        return JSONResponse({"success": False}, status_code=400)

    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"captures/lcd_{ts}.jpg"
    cv2.imwrite(filename, latest_frame)

    return JSONResponse({"success": True, "file": filename})


# import nest_asyncio
# import uvicorn

# nest_asyncio.apply()

# uvicorn.run(
#     "main:app",          # <-- filename:variable
#     host="0.0.0.0",     # <-- NO http://
#     port=5000,
#     reload=False
# )
