from ultralytics import YOLO
import cv2

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # You can use 'yolov8n.pt', 'yolov8s.pt', etc.

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    # Loop over each result in the list and plot annotations
    for result in results:
        annotated_frame = result.plot()  # Plot the detections on the frame

    # Display the annotated frame
    cv2.imshow('YOLOv8 Webcam', annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
