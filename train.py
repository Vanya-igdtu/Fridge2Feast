from ultralytics import YOLO

def main():

    model = YOLO(
        r"C:\Users\vanya\OneDrive\Desktop\VANYA\Recipe generation from food project\runs\detect\train8\weights\last.pt"
    )

    results = model.train(
        data=r"C:\Users\vanya\OneDrive\Desktop\VANYA\Recipe generation from food project\Dataset\data.yaml",
        imgsz=384,
        epochs=15,
        batch=8,
        optimizer="Adam",
        lr0=0.0008,
        cos_lr=True,
        device='cpu',
        resume=False,
        project=r"C:\Users\vanya\OneDrive\Desktop\VANYA\Recipe generation from food project\runs",
        name="train8"
    )

if __name__ == "__main__":
    main()
