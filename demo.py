import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ==== Camera ====
try:
    from picamera2 import Picamera2, Preview
    USE_PICAM = True
except Exception:
    USE_PICAM = False
try:
    import cv2
except Exception:
    cv2 = None

# ==== CONFIG ====
CKPT_PATH = "/home/comp8296/comp8296-assign2/outputs_a2/best_mobilenetv3_quantized.pth"
IMG_SIZE = 224
SHOW_WINDOW = True
FALLBACK_CLASSES = ["Healthy", "Target_Spot", "Early_Blight", "Tomato_YellowLeaf_Curl_Virus"]

# HSV green
LOW  = np.array([35, 40, 40], np.uint8)
HIGH = np.array([85,255,255], np.uint8)

# ==== Load Model ====
def get_model(num_classes=4):
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m

def load_model():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = get_model(len(FALLBACK_CLASSES))
    if ckpt.get("quantized", False) or "quantized" in CKPT_PATH.lower():
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt.get("classes", FALLBACK_CLASSES)

# ==== Preprocess ====
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def preprocess_bgr(bgr):
    if cv2 is not None and (bgr.shape[1] != IMG_SIZE or bgr.shape[0] != IMG_SIZE):
        bgr = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA).copy()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2,0,1).float() / 255.0
    mean = torch.tensor(MEAN).view(3,1,1)
    std  = torch.tensor(STD).view(3,1,1)
    x = (x - mean) / std
    return x.unsqueeze(0)

# ==== Green detection ====
def green_bbox(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOW, HIGH)
    k = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    x,y,w,h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    if w*h < 0.02*bgr.shape[0]*bgr.shape[1]: return None
    return (x,y,w,h)

# ==== Main ====
def main():
    model, classes = load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"[INFO] Model loaded with {len(classes)} classes on {device.upper()}")

    # camera
    if USE_PICAM:
        cam = Picamera2()
        cam.configure(cam.create_video_configuration(main={"size": (640,480)}))
        cam.start_preview(Preview.NULL)
        cam.start()
        def read_frame():
            rgb = cam.capture_array()
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif cv2 is not None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: No camera found.")
            return
        def read_frame():
            ok, frame = cap.read()
            return frame if ok else None
    else:
        print("No camera available.")
        return

    print("[Running] Press Ctrl+C or ESC to quit.")

    try:
        with torch.no_grad():
            while True:
                bgr = read_frame()
                if bgr is None: 
                    continue

                # green box
                bb = green_bbox(bgr)
                if bb:
                    x0,y0,w,h = bb
                    crop = bgr[y0:y0+h, x0:x0+w].copy()
                else:
                    crop = bgr

                # classfication
                x = preprocess_bgr(crop).to(device)
                t0 = time.time()
                logits = model(x)
                infer_ms = (time.time() - t0)*1000
                prob = F.softmax(logits, dim=1)[0]
                idx = int(torch.argmax(prob))
                label = classes[idx] if 0 <= idx < len(classes) else f"cls{idx}"
                conf = float(prob[idx])*100

                print(f"Class: {label:20s} | Conf: {conf:5.1f}% | Infer: {infer_ms:6.2f} ms", end="\r", flush=True)

                if SHOW_WINDOW and cv2 is not None:
                    vis = bgr.copy()
                    if bb:
                        cv2.rectangle(vis, (x0,y0), (x0+w,y0+h), (0,255,0), 2)
                        cv2.putText(vis, f"{label} {conf:.1f}%", (x0, max(0,y0-8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    else:
                        cv2.putText(vis, f"{label} {conf:.1f}%", (10,30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    cv2.imshow("Tomato Leaf Demo (ESC to quit)", vis)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
    except KeyboardInterrupt:
        print("\n[STOP] User interrupted.")
    finally:
        if USE_PICAM:
            cam.stop()
        elif cv2 is not None:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
