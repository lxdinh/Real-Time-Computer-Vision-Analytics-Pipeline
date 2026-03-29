## Real-Time Computer Vision Analytics Pipeline

A production-grade, **real-time staff vs. customer analytics pipeline** built in Python, designed for **edge deployment on NVIDIA Jetson**.  
The system combines **custom-trained YOLOv11 (82% precision)** with **OpenCV YuNet/SFace face recognition** and **TensorRT INT8 optimization** for sub‑20 ms inference.

---

### Project Overview

This project implements an end-to-end **real-time video analytics pipeline** that:

- Detects and classifies **Staff vs. Customers** using a **YOLOv11** model trained on a custom dataset.
- Performs **face detection and recognition** on detected staff regions using **OpenCV’s YuNet + SFace**.
- Targets **edge devices** (e.g., NVIDIA Jetson) with an **INT8 TensorRT export pipeline** to maximize FPS and minimize latency.

The architecture is modular and geared toward **production-readiness**, with clear separation between detection, recognition, database management, configuration, and export utilities.

---

### Features

- **Real-Time Staff vs. Customer Detection**
  - Custom-trained **YOLOv11** model (`yolo11n.pt`) trained on a domain-specific dataset.
  - Achieves **~82% precision** on the Staff vs. Customer classification task.
  - Clean `StaffDetector` wrapper around Ultralytics’ YOLOv11 API.

- **Face Recognition for Staff Identification**
  - Uses OpenCV’s **YuNet** (`FaceDetectorYN`) to localize faces within each detected person.
  - Uses OpenCV’s **SFace** (`FaceRecognizerSF`) to compute robust **128D face embeddings**.
  - Embeddings are compared against an in-memory **`FaceDatabase`** using cosine similarity / L2 distance.
  - Supports multi-image enrollment per identity (embeddings are averaged for robustness).

- **Edge Optimization (TensorRT)**
  - `export_trt.py` exports YOLOv11 weights to a **TensorRT `.engine`** with:
    - **FP16 (`half=True`)** and **INT8 (`int8=True`)** enabled by default.
  - Designed to be executed **on the target Jetson device** to generate **hardware-specific engines**.
  - Optimized configuration targeting **< 20 ms latency** per frame.

- **Modular, Extensible Architecture**
  - `config.py` centralizes paths, thresholds, and runtime hyperparameters.
  - `detector.py`, `recognizer.py`, and `database.py` provide clean, testable components.
  - `main.py` orchestrates a lean real-time loop for webcam/stream ingestion.

---

### Tech Stack

- **Language**
  - Python 3.x

- **Core Libraries**
  - **Ultralytics YOLOv11** (person / staff vs. customer detection)
  - **OpenCV (opencv-contrib-python)**  
    - `cv2.dnn` + `cv2.FaceDetectorYN` (YuNet) for face detection  
    - `cv2.FaceRecognizerSF` (SFace) for face embeddings & matching
  - **NumPy**

- **Edge / Inference Optimization**
  - **TensorRT** (via Ultralytics export) for:
    - FP16 & INT8 inference
    - Optimized `.engine` artefacts compiled for specific Jetson hardware

---

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd "Real-Time Computer Vision Pipeline"
```

#### 2. Create & Activate a Virtual Environment (Recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

#### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:

- `numpy`
- `opencv-contrib-python`
- `ultralytics`

#### 4. Place / Download Model Weights

- **YOLOv11 Staff vs. Customer model**

  Place your trained **YOLOv11** checkpoint at the project root:

  ```text
  Real-Time Computer Vision Pipeline/
    yolo11n.pt
  ```

  `config.YOLO_WEIGHTS_PATH` points to this location by default.

- **OpenCV YuNet + SFace**

  On first run, `recognizer.ensure_face_models()` will:

  - Create `models/weights/`
  - Download:
    - `face_detection_yunet_2023mar.onnx`
    - `face_recognition_sface_2021dec.onnx`

  from the official OpenCV mirrors (Hugging Face).  
  You may also manually place these files in `models/weights/` if preferred.

#### 5. Enroll Known Staff Faces

Add enrollment images to:

```text
data/known_faces/<staff_name>/*.jpg|*.png|...
```

For each `<staff_name>`, embeddings are extracted using SFace and averaged to build a robust template.

---

### Usage

#### 1. Training the Custom YOLOv11 Model (Optional)

If you want to retrain or fine-tune:

```bash
yolo train \
  model=yolo11n.pt \
  data=data/yolo_dataset/D-Vision.v2-yolov11x.yolov11/data.yaml \
  epochs=50 \
  imgsz=640
```

Ensure your `data.yaml` paths are correct and your dataset is laid out as:

```text
data/yolo_dataset/D-Vision.v2-yolov11x.yolov11/
  data.yaml
  train/images/
  valid/images/
  test/images/
```

#### 2. Running the Real-Time Pipeline

From the project root:

```bash
python main.py
```

The pipeline will:

1. Load / download YuNet + SFace weights.
2. Build the face database from `data/known_faces/`.
3. Load `yolo11n.pt` via `StaffDetector`.
4. Open `cv2.VideoCapture(0)`, process frames, and visualize:

   - **Green boxes** around detected persons (Staff / Customers).
   - **Blue boxes** around detected faces.
   - **Overlayed labels** for recognized staff names or `"Unknown"`.
   - **FPS overlay** (smoothed) in the top-left corner.

Press **`q`** to exit cleanly.

#### 3. Exporting YOLOv11 to TensorRT (Jetson / Edge Only)

> **IMPORTANT:** TensorRT engines are *hardware-specific*.  
> Only run this on the **target Jetson** (or deployment device), **not** your dev machine.

On the Jetson, with `yolo11n.pt` present:

```bash
python export_trt.py \
  --weights yolo11n.pt \
  --out-dir models/engines \
  --imgsz 640
```

By default this will:

- Enable **FP16 (`--half`)** and **INT8 (`--int8`)** where supported.
- Use `coco8.yaml` for INT8 calibration (you can pass your own `--data` YAML for domain calibration).
- Save the resulting `.engine` under `models/engines/`.

You can disable INT8 or FP16 if needed:

```bash
python export_trt.py --weights yolo11n.pt --out-dir models/engines --no-int8
python export_trt.py --weights yolo11n.pt --out-dir models/engines --no-half
```

---

### Folder Structure

High-level structure (non-exhaustive):

```text
Real-Time Computer Vision Pipeline/
├── config.py                # Central paths, thresholds, and hyperparameters
├── database.py              # Face gallery loading + cosine/L2 matching
├── detector.py              # YOLOv11 staff/customer detector wrapper
├── recognizer.py            # YuNet + SFace embedding pipeline + staff recognition
├── main.py                  # Real-time webcam loop (detection + recognition + visualization)
├── export_trt.py            # YOLOv11 → TensorRT INT8 export (Jetson-only)
├── requirements.txt         # Python dependencies (Ultralytics, OpenCV contrib, NumPy)
├── models/
│   ├── weights/             # YuNet + SFace ONNX models (auto-downloaded)
│   └── engines/             # TensorRT .engine files (exported on Jetson)
├── data/
│   ├── known_faces/         # Enrollment images per staff identity
│   └── yolo_dataset/        # YOLO training dataset (train/valid/test)
│       └── D-Vision.v2-yolov11x.yolov11/
│           ├── data.yaml
│           ├── train/
│           ├── valid/
│           └── test/
├── runs/                    # Ultralytics training outputs (ignored by Git)
└── .gitignore               # Excludes heavy artefacts (models, data, runs, venv, etc.)
```

---

### Notes & Future Work

- **Multi-stream support:** Extend `main.py` to handle multiple camera inputs or RTSP streams with thread-safe queues.
- **Person re-ID / trajectory analytics:** Build on top of the existing detections to add dwell time, path analytics, and area-based alerts.
- **Full TensorRT integration:** Wire the generated `.engine` files directly into the detection/recognition pipeline for maximum Jetson performance.

This project demonstrates a complete, **recruiter-ready** example of:

- Custom YOLOv11 training and deployment,
- Production-quality **face recognition** with OpenCV,
- Real-world **edge optimization** using TensorRT INT8 on NVIDIA Jetson hardware.