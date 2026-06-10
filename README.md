 Offline Facial Authentication System

An end-to-end offline facial authentication system that combines **Face Recognition**, **Liveness Detection**, and **Grad-CAM Explainability** — all wrapped in a clean Streamlit web interface.

---

## Features

- **Face Recognition** — Identifies individuals from a database of 5,749 people using FaceNet embeddings
- **Liveness Detection** — Classifies faces as real or spoofed with ~98% accuracy using MobileNetV2
- **Grad-CAM Explainability** — Visualizes which facial regions influenced the model's prediction
- **Streamlit UI** — User-friendly web interface for uploading and analyzing face images
- **Fully Offline** — No internet connection required after setup

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow, PyTorch |
| Face Detection | MTCNN |
| Face Recognition | FaceNet (InceptionResnetV1) |
| Liveness Detection | MobileNetV2 |
| Explainability | Grad-CAM |
| Computer Vision | OpenCV |
| UI | Streamlit |
| Training Environment | Google Colab |

---

## Project Structure

```
Faceproject/
│
├── faceRecognitions.ipynb      # Face recognition pipeline
├── LivenessDetection.ipynb     # Liveness detection training
├── GradCAM.ipynb               # Grad-CAM visualization
├── Deployment.ipynb            # Deployment preparation
│
├── app.py                      # Streamlit application
├── requirements.txt
│
├── models/
│   ├── embeddings.pkl          # FaceNet face embeddings
│   ├── liveness_model.h5       # Trained MobileNetV2 model
│   └── training_history.pkl
│
├── outputs/
│   ├── accuracy_plot.png
│   ├── loss_plot.png
│   ├── confusion_matrix.png
│   └── gradcam_output.png
│
└── Dataset/
    ├── lfw-deepfunneled/       # LFW face recognition dataset
    └── Colab_Data/             # Anti-spoof liveness dataset
```

---

## Datasets

### Face Recognition
- **Dataset:** [LFW — Labeled Faces in the Wild](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)
- **People:** 5,749 identities

### Liveness Detection
- **Dataset:** Face Anti-Spoof Small Dataset (Kaggle)
- **Split (from Training folder only):**

| Split | Real | Fake |
|---|---|---|
| Train (70%) | 2,711 | 5,456 |
| Validation (15%) | 580 | 1,168 |
| Test (15%) | 580 | 1,168 |

> Only RGB (color) images were used. Depth and IR channels were excluded.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/facial-authentication-system.git
cd facial-authentication-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Model Files

Place the following files in the `models/` directory:

- `embeddings.pkl` — generated from `faceRecognitions.ipynb`
- `liveness_model.h5` — generated from `LivenessDetection.ipynb`

### 4. Run the App

```bash
streamlit run app.py
```

---

## Model Architecture

### Face Recognition
```
Input Image -> MTCNN (Face Detection) -> FaceNet (Embedding) -> Cosine Similarity -> Identity
```

### Liveness Detection
```
Input Image -> MobileNetV2 (Pretrained) -> GlobalAveragePooling -> Dropout -> Dense(sigmoid) -> Real / Fake
```

- **Optimizer:** Adam
- **Loss:** Binary Crossentropy
- **Epochs:** 5
- **Test Accuracy:** ~98%

---

## Results

| Metric | Value |
|---|---|
| Liveness Test Accuracy | ~98% |
| Face Recognition Confidence (example) | 100% |

**Confusion Matrix (Liveness Detection):**

|  | Predicted Fake | Predicted Real |
|---|---|---|
| **Actual Fake** | 1,168 | 0 |
| **Actual Real** | 29 | 551 |

---

## Grad-CAM Explainability

Grad-CAM highlights which regions of the face the liveness model focuses on when making a prediction. The heatmap is overlaid on the original image for easy interpretation.

Example output saved to `outputs/gradcam_output.png`.

---

## Notebooks

| Notebook | Description |
|---|---|
| `faceRecognitions.ipynb` | Generates FaceNet embeddings for all LFW identities |
| `LivenessDetection.ipynb` | Trains MobileNetV2 for real/fake classification |
| `GradCAM.ipynb` | Generates Grad-CAM heatmap visualizations |
| `Deployment.ipynb` | Creates `app.py` and `requirements.txt` for Streamlit |

---

## Requirements

```
streamlit
tensorflow
torch
torchvision
facenet-pytorch
opencv-python
Pillow
numpy
matplotlib
scikit-learn
```

---

## Notes

- All notebooks were developed and trained on Google Colab with GPU runtime.
- Models are saved to and loaded from Google Drive during training.
- The Streamlit app runs fully offline once models are downloaded locally.

---

