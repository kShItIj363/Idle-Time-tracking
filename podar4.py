import numpy as np
from ultralytics import YOLO
import cv2
import time
import threading
import queue
from sort import Sort
from helper import create_video_writer

# Video Source
video_path = "rtsp://admin:@192.168.6.55:554/ch0_0.264"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Video Writer
writer = create_video_writer(cap, "out2.mp4")

# Model and Tracker Initialization
model = YOLO("yolov8n.pt")
tracker = Sort(max_age=20)

# Tracking Data Structures
boxes = []
drawing_box = False
current_box = []
tracked_data = {}

# Resize Window
cv2.namedWindow("Video", cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow("Video", 1280, 720)

# Mouse Callback to Draw Boxes
def mouse_callback(event, x, y, flags, param):
    global drawing_box, current_box
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_box = True
        current_box = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE and drawing_box:
        if len(current_box) == 1:
            current_box.append((x, y))
        else:
            current_box[1] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing_box = False
        if len(current_box) == 2:
            x1, y1 = current_box[0]
            x2, y2 = current_box[1]
            boxes.append([(min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2))])
            idx = len(boxes) - 1
            tracked_data[idx] = {
                "entry_count": set(),
                "exit_count": set(),
                "time_inside": {},
                "time_idle": {},
                "box_idle_time": None,  # Track idle time for the box
            }

cv2.setMouseCallback("Video", mouse_callback)

# Frame Queue
frame_queue = queue.Queue(maxsize=10)
stop_thread = False

# Read Frames in a Separate Thread
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

# Main Processing Loop
while True:
    if not frame_queue.empty():
        img = frame_queue.get()
        results = model(img, stream=True)
        detections = np.empty((0, 5))

        # Object Detection
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if cls == 0 and conf > 0.20:  # Detect only people
                    detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

        resultsTracker = tracker.update(detections)

        # Process Each Box
        for idx, box in enumerate(boxes):
            x1_b, y1_b = box[0]
            x2_b, y2_b = box[1]
            
            # Draw Box
            cv2.rectangle(img, (x1_b, y1_b), (x2_b, y2_b), (0, 255, 0), 2)
            cv2.putText(img, f'Box {idx}', (x1_b, y1_b - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # Ensure box is initialized in tracking data
            if idx not in tracked_data:
                tracked_data[idx] = {
                    "entry_count": set(),
                    "exit_count": set(),
                    "time_inside": {},
                    "time_idle": {},
                    "box_idle_time": None,
                }

            people_inside = []
            for result in resultsTracker:
                x1, y1, x2, y2, id = map(int, result)
                cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
                inside = x1_b <= cx <= x2_b and y1_b <= cy <= y2_b

                if inside:
                    people_inside.append(id)

                    # Start tracking time inside the box
                    if id not in tracked_data[idx]["time_inside"]:
                        tracked_data[idx]["time_inside"][id] = time.time()

                    # Stop idle time when inside
                    if id in tracked_data[idx]["time_idle"]:
                        del tracked_data[idx]["time_idle"][id]

                    # Register Entry
                    if id not in tracked_data[idx]["entry_count"]:
                        tracked_data[idx]["entry_count"].add(id)
                        print(f"ID {id} - Entered Box {idx}")

                else:
                    # Stop tracking inside time
                    if id in tracked_data[idx]["time_inside"]:
                        duration = time.time() - tracked_data[idx]["time_inside"].pop(id)
                        print(f"ID {id} stayed in Box {idx} for {duration:.2f} seconds")

                        # Start idle time when leaving
                        if id not in tracked_data[idx]["time_idle"]:
                            tracked_data[idx]["time_idle"][id] = time.time()

                    # Register Exit
                    if id in tracked_data[idx]["entry_count"] and id not in tracked_data[idx]["exit_count"]:
                        tracked_data[idx]["exit_count"].add(id)
                        print(f"ID {id} - Exited Box {idx}")

                # Draw Person
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), -1)
                cv2.putText(img, f'ID {id}', (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # Display Inside Time
                if id in tracked_data[idx]["time_inside"]:
                    stay_duration = time.time() - tracked_data[idx]["time_inside"][id]
                    cv2.putText(img, f'{stay_duration:.1f}s', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Track Box Idle Time
            if not people_inside:
                if tracked_data[idx]["box_idle_time"] is None:
                    tracked_data[idx]["box_idle_time"] = time.time()
                idle_duration = time.time() - tracked_data[idx]["box_idle_time"]
                cv2.putText(img, f'Idle {idle_duration:.1f}s', (x1_b, y1_b - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            else:
                tracked_data[idx]["box_idle_time"] = None  # Reset idle timer when someone enters

            # Display Entry/Exit Counts
            cv2.putText(img, f'Entries: {len(tracked_data[idx]["entry_count"])}', (10, 30 + idx * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f'Exits: {len(tracked_data[idx]["exit_count"])}', (10, 50 + idx * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show Video
        cv2.imshow("Video", img)
        writer.write(img)
        time.sleep(0.03)

    if cv2.waitKey(1) == ord("q"):
        stop_thread = True
        break

frame_thread.join()
cap.release()
writer.release()
cv2.destroyAllWindows()

