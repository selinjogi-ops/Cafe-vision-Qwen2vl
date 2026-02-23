"""
Cafe Monitoring System with Qwen2-VL
YOLOv8 + DeepSort + Qwen2-VL-2B-Instruct (Async)
"""
import cv2
import csv
import threading
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

# -------------------- LOAD QWEN2-VL --------------------
print("Loading Qwen2-VL...")
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_ID)
vlm = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)
vlm.eval()
print("Qwen2-VL loaded successfully")

# -------------------- YOLO & DEEPSORT --------------------
yolo = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

# -------------------- VIDEO SOURCE --------------------
cap = cv2.VideoCapture(0)

# -------------------- CSV LOG --------------------
CSV_FILE = "cafe_qwen2vl_log.csv"
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Time", "Qwen2-VL_Output"])

# -------------------- GLOBAL STATE --------------------
vlm_running = False
last_output = ""
frame_count = 0

# -------------------- VLM THREAD --------------------
def run_vlm_async(frame):
    global vlm_running, last_output
    vlm_running = True
    print("\n🔵 Qwen2-VL running...")

    try:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Proper multimodal message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe what is happening in the cafe."}
                ]
            }
        ]

        # Step 1: Format conversation
        text_prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        # Step 2: Convert to tensors properly
        inputs = processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt"
        )

        # Move tensors to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = vlm.generate(
                **inputs,
                max_new_tokens=60
            )

        text = processor.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0]

        last_output = text
        print("🟢 Qwen2-VL Output:", text)

        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(), text])

    except Exception as e:
        print("🔴 VLM Error:", e)

    vlm_running = False

# -------------------- MAIN LOOP --------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # -------- YOLO PERSON DETECTION --------
    results = yolo(frame, classes=[0], verbose=False)[0]  # person
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

    tracks = tracker.update_tracks(detections, frame=frame)

    for t in tracks:
        if not t.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, t.to_ltrb())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID {t.track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # -------- RUN VLM EVERY ~5 SECONDS --------
    frame_count += 1
    if frame_count % 150 == 0 and not vlm_running:
        threading.Thread(
            target=run_vlm_async,
            args=(frame.copy(),),
            daemon=True
        ).start()

    # -------- DISPLAY VLM OUTPUT --------
    if last_output:
        # Extract only assistant response
        if "assistant" in last_output.lower():
            clean_text = last_output.split("assistant")[-1]
        else:
            clean_text = last_output

        clean_text = clean_text.replace("\n", " ").strip()
        display_text = clean_text[:100]

        cv2.putText(
            frame,
            display_text,
            (20, frame.shape[0] - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
    )
    cv2.imshow("Cafe Monitoring (Qwen2-VL)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()