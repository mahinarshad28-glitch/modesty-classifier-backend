from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2, numpy as np
from PIL import Image
import io, base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

CLASSES = [
    "Modest_Male",
    "Immodest_Male",
    "Modest_Female_Hijab",
    "Modest_Female_Niqab",
    "Immodest_Female"
]

# ── Load all models with correct file names ──
models = {
    "person1": {
        "detection":      YOLO("models/person1/DETECTION_best.pt"),
        "classification": None,  # not proper YOLO format,
        "segmentation":   YOLO("models/person1/SEGMENTATION_best.pt"),
        "regions": ["region1", "region2"]
    },
    "person2": {
        "detection":      YOLO("models/person2/modesty_detector_best.pt"),
        "classification": None,  # improperly saved — skipped,
        "segmentation":   YOLO("models/person2/modesty_segmentor_best.pt"),
        "regions": ["region3", "region4"]
    },
    "person3": {
        "detection":      YOLO("models/person3/detection_best.pt"),
        "classification": None,   # .h5 file — skipped for now
        "segmentation":   YOLO("models/person3/segmentation_best.pt"),
        "regions": ["region5", "region6"]
    },
}

def run_person_models(person_key, img_array):
    m = models[person_key]

    # Detection
    det_result = m["detection"].predict(img_array, conf=0.25)[0]
    detections = []
    if det_result.boxes is not None:
        for box in det_result.boxes:
            detections.append({
                "person":     person_key,
                "bbox":       box.xyxy[0].tolist(),
                "confidence": float(box.conf[0]),
                "class_id":   int(box.cls[0]),
                "class_name": CLASSES[int(box.cls[0])] if int(box.cls[0]) < len(CLASSES) else "unknown"
            })

    # Classification (skip if None)
    classification = None
    if m["classification"] is not None:
        cls_result = m["classification"].predict(img_array, conf=0.25)[0]
        if cls_result.probs is not None:
            top_cls_id = int(cls_result.probs.top1)
            classification = {
                "person":     person_key,
                "class_id":   top_cls_id,
                "class_name": CLASSES[top_cls_id] if top_cls_id < len(CLASSES) else "unknown",
                "confidence": float(cls_result.probs.top1conf)
            }

    # Segmentation
    seg_result = m["segmentation"].predict(img_array, conf=0.25)[0]
    seg_count = 0
    if seg_result.masks is not None:
        seg_count = len(seg_result.masks)

    return {
        "person":         person_key,
        "regions":        m["regions"],
        "detections":     detections,
        "classification": classification,
        "seg_count":      seg_count,
        "seg_result_obj": seg_result
    }

def merge_all_results(all_results, img_array):
    all_detections = []
    for r in all_results:
        all_detections.extend(r["detections"])
    all_detections.sort(key=lambda x: x["confidence"], reverse=True)

    best_classification = None
    for r in all_results:
        if r["classification"] is not None:
            if (best_classification is None or
                r["classification"]["confidence"] > best_classification["confidence"]):
                best_classification = r["classification"]

    best_seg_result = max(all_results, key=lambda x: x["seg_count"])
    annotated_img = best_seg_result["seg_result_obj"].plot()

    colors = {
        "person1": (0, 255, 0),
        "person2": (255, 165, 0),
        "person3": (0, 0, 255),
    }
    for det in all_detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color = colors.get(det["person"], (255, 255, 255))
        label = f'{det["class_name"]} {det["confidence"]:.2f}'
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_img, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    _, buffer = cv2.imencode('.jpg', annotated_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "annotated_image":        img_base64,
        "all_detections":         all_detections,
        "best_classification":    best_classification,
        "total_persons_detected": len(all_detections),
        "models_used":            [r["person"] for r in all_results],
        "per_model_summary": [
            {
                "person":           r["person"],
                "regions":          r["regions"],
                "detections_found": len(r["detections"]),
                "classification":   r["classification"],
                "seg_masks_found":  r["seg_count"]
            }
            for r in all_results
        ]
    }

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(img)

    all_results = []
    for person_key in ["person1", "person2", "person3"]:
        result = run_person_models(person_key, img_array)
        all_results.append(result)

    final = merge_all_results(all_results, img_array)
    return final

@app.get("/")
def root():
    return {"status": "Modesty Classifier API running", "models_loaded": 8}
