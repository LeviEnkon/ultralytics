import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
model = YOLO('yolov8n.pt')
source = '0'
racker='botsort.yaml'
while True:
    ret, frame= cap.read()
    if not ret:
        break
    results= model.track(frame, conf=0.3, iou=0.5, persist=True, show=False)
    if results[0].boxes.id is None:
        pass
    
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    ids = results[0].boxes.id.cpu().numpy().astype(int)
    for box, id in zip(boxes, ids):
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Id {id}",
            (box[0], box[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



# for results in model.predict(source, show = True, stream = True):
#     for box in results.boxes:
#         boxclass, boxcoord = int(box.cls[0]), box.xywh[0]
#         h = float(boxcoord[3])
#         if boxclass == 0:
#             print("person")
#         else:
#             print("object no.", boxclass)
#         print("height=", h)