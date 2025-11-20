

from ultralytics import YOLO
from pathlib import Path
import cv2
import os


def predict_generate_list(model , image_path):
    # Making prediction from image
    results = model.predict(image_path , conf = 0.5)
    Shelf = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
            class_name = model.names[class_id]
            print(f"Detected : {class_name} , with Confidence {confidence}")
            Shelf.append(class_name)
    return Shelf

def run_webcam(model):
    """Runs real-time webcam detection."""
    print("\nStarting webcam... Press Q to quit.\n")

    cap = cv2.VideoCapture(0)  # 0 = default camera

    if not cap.isOpened():
        print("Error: Webcam not found.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        results = model(frame, conf=0.5)

        # Draw predictions on frame
        annotated_frame = results[0].plot()

        cv2.imshow("Ingredient Detection - Press Q to Quit", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()





if __name__ == '__main__':
    this_dir = Path(__file__).parent
    os.chdir(this_dir)

    detect_path = this_dir/"runs"
    train_folders = [f for f in os.listdir(detect_path) if os.path.isdir(detect_path/f) and f.startswith("train")]
    model_path = detect_path/ train_folders[-1]/"weights"/"best.pt"
    model = YOLO(model_path)
    if not train_folders:
        raise FileNotFoundError("No 'train' folder found in runs/. Train your model first.")
    
    print(f"Loading model from: {model_path}")

    USE_CAMERA = True   # Change to False if you want to test an image

    if USE_CAMERA:
        run_webcam(model)
    else:
        image_path = r"C:\Users\vanya\OneDrive\Desktop\VANYA\Recipe generation from food project\Dataset\valid\images\630a5320cee3bb9c112c76ba3d67dccb_jpg.rf.a2c40936878596bf00571d7001bfe366.jpg"
        detected_list = predict_generate_list(model, image_path)

        print("\nDetected Ingredients:")
        print(detected_list)
    
