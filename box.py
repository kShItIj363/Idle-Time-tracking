import cv2
import json

video_path = "rtmp://ptz.vmukti.com:80/live-record/1401"
cap = cv2.VideoCapture(video_path)

rois = []
drawing = False
start_point = None

def draw_rectangle(event, x, y, flags, param):
    global drawing, start_point, rois

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        rois.append({"id": len(rois), "x1": start_point[0], "y1": start_point[1], "x2": end_point[0], "y2": end_point[1]})

cap.set(cv2.CAP_PROP_POS_FRAMES, 100)  # Jump to a frame with clear view
ret, frame = cap.read()
cv2.imshow("Select ROIs", frame)
cv2.setMouseCallback("Select ROIs", draw_rectangle)

while True:
    temp_frame = frame.copy()
    for roi in rois:
        cv2.rectangle(temp_frame, (roi["x1"], roi["y1"]), (roi["x2"], roi["y2"]), (0, 255, 0), 2)
        cv2.putText(temp_frame, f"ROI {roi['id']}", (roi["x1"], roi["y1"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.imshow("Select ROIs", temp_frame)

    key = cv2.waitKey(1)
    if key == ord('s'):  # Save ROIs
        with open("rois.json", "w") as f:
            json.dump({"rois": rois}, f, indent=4)
        print("ROIs saved.")
        break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
