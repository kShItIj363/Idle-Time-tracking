import numpy as np
from ultralytics import YOLO
import cv2
import time
import threading
import queue
import json
import pymongo
from pymongo import MongoClient
from datetime import datetime
from sort import Sort
import subprocess

# MongoDB Connection
client = MongoClient("mongodb+srv://thakretushar89:oL2JKFkYkIHvEvuk@replica.rvlq6.mongodb.net/arcis")
db = client["arcis"]
collection = db["boxes"]
collection_present = db["boxes1"]

# Load ROI Configuration from JSON
roi_file = "rois.json"
with open(roi_file, "r") as f:
    rois_data = json.load(f)

PADDING = 10  # Adjust ROI size for better accuracy
rois = rois_data.get("rois", [])
boxes = [(
    (min(roi["x1"], roi["x2"]) - PADDING, min(roi["y1"], roi["y2"]) - PADDING), 
    (max(roi["x1"], roi["x2"]) + PADDING, max(roi["y1"], roi["y2"]) + PADDING)
) for roi in rois]

print("Loaded ROIs:", boxes)

# Video Source
video_path = "rtmp://ptz.vmukti.com:80/live-record/1401"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video stream {video_path}")
    exit()

# Model and Tracker Initialization
model = YOLO("yolov8n.pt")
tracker = Sort(max_age=500, min_hits=10)

# Idle Time Thresholds
idle_threshold = 10
moving_threshold = 15
jitter_threshold = 3
history_frames = 5
idle_thresholdA = 50

# Tracking Data Structures
tracked_data = {idx: {
    "box_idle_time": None,
    "idle_periods": [],
    "total_idle_time": 0,
    "last_positions": [],
    "present_time": 0
} for idx in range(len(boxes))}

logged_idle_events = set()
start_time = time.time()
last_update_time = start_time

# Frame Queue
frame_queue = queue.Queue(maxsize=100)
stop_thread = False

# RTMP Output URL
rtmp_url = "rtmp://ptz.vmukti.com:80/live-record/podar"

# FFmpeg command for streaming
ffmpeg_cmd = [
    "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
    "-s", f"{int(cap.get(3))}x{int(cap.get(4))}",
    "-r", "10", "-i", "-",
    "-c:v", "libx264", "-preset", "ultrafast", "-b:v", "1500k",
    "-maxrate", "1500k", "-bufsize", "5000k",
    "-pix_fmt", "yuv420p", "-g", "30", "-tune", "zerolatency",
    "-f", "flv", rtmp_url
]

# Start FFmpeg Process
ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

def read_frames():
    global stop_thread
    while not stop_thread:
        success, frame = cap.read()
        if not success:
            stop_thread = True
            break
        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            time.sleep(0.01)

frame_thread = threading.Thread(target=read_frames)
frame_thread.start()

while True:
    current_time = time.time()
    if not frame_queue.empty():
        img = frame_queue.get()
        results = model(img, stream=True)
        detections = np.empty((0, 5))

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if cls == 0 and conf > 0.20:  # Only detect persons
                    detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

        resultsTracker = tracker.update(detections)

        for idx, box in enumerate(boxes):
            x1_b, y1_b = box[0]
            x2_b, y2_b = box[1]
            cv2.rectangle(img, (x1_b, y1_b), (x2_b, y2_b), (0, 255, 0), 2)
            cv2.putText(img, f'Box {idx}', (x1_b, y1_b - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            people_inside = False
            current_positions = []

            for result in resultsTracker:
                center_x = (int(result[0]) + int(result[2])) // 2
                center_y = (int(result[1]) + int(result[3])) // 2

                if x1_b <= center_x <= x2_b and y1_b <= center_y <= y2_b:
                    prev_positions = tracked_data[idx]["last_positions"]
                    if prev_positions:
                        distances = [
                            np.sqrt((center_x - px) ** 2 + (center_y - py) ** 2) 
                            for px, py in prev_positions[-history_frames:]
                        ]
                        max_movement = max(distances) if distances else 0
                        avg_movement = sum(distances) / len(distances) if distances else 0

                        if max_movement < moving_threshold and avg_movement < jitter_threshold:
                            current_positions.append((center_x, center_y))
                            people_inside = True
                    else:
                        current_positions.append((center_x, center_y))
                        people_inside = True

            if len(current_positions) > history_frames:
                current_positions = current_positions[-history_frames:]

            tracked_data[idx]["last_positions"] = current_positions

            if not people_inside:  # Box is idle
                if tracked_data[idx]["box_idle_time"] is None:
                    tracked_data[idx]["box_idle_time"] = time.time()
                idle_duration = time.time() - tracked_data[idx]["box_idle_time"]
                if idle_duration >= idle_threshold:
                    cv2.putText(img, f'Idle {idle_duration:.1f}s', (x1_b, y1_b - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            else:
                if tracked_data[idx]["box_idle_time"] is not None:
                    idle_time = time.time() - tracked_data[idx]["box_idle_time"]
                    if idle_time >= idle_thresholdA:
                        idle_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(tracked_data[idx]["box_idle_time"]))
                        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        event_key = f"{idx}_{idle_start_time}_{end_time}"
                        if event_key not in logged_idle_events:
                            logged_idle_events.add(event_key)
                            sendtime_date = datetime.strptime(datetime.now().strftime("%d%m%Y"), "%d%m%Y")
                            collection.insert_one({
                                "box_id": idx, "idle_start_time": idle_start_time, "end_time": end_time,
                                "idle_duration": round(idle_time, 2),
                                "total_idle_time": round(tracked_data[idx]["total_idle_time"], 2),
                                "sendtime": sendtime_date
                            })
                            tracked_data[idx]["idle_periods"].append([idle_start_time, end_time, round(idle_time, 2)])
                            tracked_data[idx]["total_idle_time"] += idle_time
                    tracked_data[idx]["box_idle_time"] = None
            elapsed_time = current_time - start_time
            tracked_data[idx]["present_time"] = elapsed_time - tracked_data[idx]["total_idle_time"]

        ffmpeg_proc.stdin.write(img.tobytes())

        if current_time - last_update_time >= 180:  # 30 minutes update
            sendtime_date = datetime.now()
            for idx in range(len(boxes)):
                collection_present.update_one(
                    {"box_id": idx},
                    {"$set": {
                        "present_time": round(tracked_data[idx]["present_time"], 2),
                        "total_idle_time": round(tracked_data[idx]["total_idle_time"], 2),
                        "sendtime": sendtime_date
                    }},
                    upsert=True
                )
            last_update_time = current_time

        if cv2.waitKey(1) == ord("q"):
            stop_thread = True
            break

frame_thread.join()
cap.release()
ffmpeg_proc.stdin.close()
ffmpeg_proc.wait()
cv2.destroyAllWindows()
