import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker

model = YOLO('yolov8s.pt')
Door = [(500, 40), (780, 20), (775, 465), (500, 440)]

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)
cv2.namedWindow('College_Main_Door')
cv2.setMouseCallback('College_Main_Door', RGB)
cap = cv2.VideoCapture('entry01.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
count = 0
tracker = Tracker()

def is_inside_polygon(x, y, polygon):
    poly_path = np.array(polygon, np.int32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(poly_path, (x, y), False) >= 0

student_count = 0
object_state = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    objects_rect = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'person' in c:
            objects_rect.append((x1, y1, x2 - x1, y2 - y1))
    objects_bbs_ids = tracker.update(objects_rect)
    for obj_bb_id in objects_bbs_ids:
        x, y, w, h, id = obj_bb_id
        cx = (x + x + w) // 2
        cy = (y + y + h) // 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        if id not in object_state:
            object_state[id] = {'inside': False}
        if is_inside_polygon(cx, cy, Door):
            if not object_state[id]['inside']:
                object_state[id]['inside'] = True
                student_count += 1
        else:
            object_state[id]['inside'] = False

    cv2.polylines(frame, [np.array(Door, np.int32)], True, (0, 0, 255), 2)
    cv2.putText(frame, 'Door', (Door[0][0], Door[0][1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    text = f'Student Count: {student_count}'
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    font_scale = 1
    font_thickness = 2
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    x, y = 5, 30
    background_top_left = (x, y - text_height - baseline)
    background_bottom_right = (x + text_width, y + baseline)

    cv2.rectangle(frame, background_top_left, background_bottom_right, bg_color, cv2.FILLED)
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, font_thickness)

    cv2.imshow("College_Main_Door", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
