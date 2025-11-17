from ultralytics import YOLO
import cv2

def main():
    # 1. carregar modelo
    model = YOLO("content/runs/detect/train2/weights/best.pt")  # caminho para os pesos

    # 2. abrir vídeo
    cap = cv2.VideoCapture("videos/video.mp4")  


    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3. inferência
        results = model(frame)[0]

        # 4. desenhar bounding boxes
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{results.names[cls]} {conf:.2f}"

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, label, (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # 5. mostrar vídeo
        cv2.imshow("YOLO Output", frame)

        # tecla Q para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
