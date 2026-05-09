# Signature Authenticity Recognition System

A machine learning-powered web application that classifies handwritten signatures as **Genuine** or **Forged** using a combination of classical computer vision feature extraction and a pre-trained classifier.

---

## Features

- Upload a signature image (PNG, JPG, JPEG)
- Automatic feature extraction using multiple computer vision techniques
- Binary classification: **Genuine** or **Forged**
- Confidence score displayed with each prediction
- Clean, dark-themed web UI built with Flask

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML / CV | scikit-learn, OpenCV, scikit-image |
| Model Storage | joblib (`.pkl` files) |
| Frontend | HTML, CSS, Jinja2 |

---

## Project Structure

```
SignatureAutheticityRecoginationSystem/
│
├── app.py                    # Flask application and feature extraction logic
├── signature_system.pkl      # Pre-trained classifier model
├── scaler_signature.pkl      # Pre-fitted feature scaler
├── signature.ipynb           # Jupyter notebook (model training & experimentation)
│
└── templates/
    └── index.html            # Web UI template
```

---

## How It Works

### Feature Extraction

Each uploaded signature image is processed through the following pipeline:

1. **Preprocessing** — Grayscale conversion, resize to 256×256, Gaussian blur, Otsu thresholding
2. **Hu Moments** — Shape descriptor capturing global geometry (7 features)
3. **HOG (Histogram of Oriented Gradients)** — Captures local edge and texture patterns
4. **LBP (Local Binary Pattern)** — Encodes micro-texture information (10 features)
5. **Skeletonization** — Counts skeleton pixels to measure stroke complexity
6. **Contour Features** — Area, perimeter, aspect ratio, extent, and solidity of the largest contour

All features are concatenated into a single vector, scaled with the pre-fitted scaler, and passed to the classifier.

### Prediction

The classifier outputs a probability for each class. The class with the higher probability is selected:
- `pred == 1` → **Genuine**
- `pred == 0` → **Forged**

The confidence percentage is displayed alongside the result.

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install flask werkzeug joblib opencv-python scikit-image scikit-learn numpy
```

> **Note:** For skeletonization via `cv2.ximgproc.thinning`, install the contrib version:
> ```bash
> pip install opencv-contrib-python
> ```

### Run the Application

```bash
python app.py
```

The app will start in debug mode at `http://127.0.0.1:5000`.

---

## Usage

1. Open the app in your browser at `http://127.0.0.1:5000`
2. Click **Choose Signature Image** and select a `.png`, `.jpg`, or `.jpeg` file
3. Click **Submit**
4. The result (**Genuine** or **Forged**) and confidence score will appear in the output box
5. Click **Clear** to reset and try another image

---

## Configuration

| Setting | Location | Default |
|---|---|---|
| Upload folder | `app.py` → `UPLOAD_FOLDER` | `static/uploads/` |
| Allowed extensions | `app.py` → `ALLOWED_EXTENSIONS` | `png, jpg, jpeg` |
| Secret key | `app.py` → `SECRET_KEY` | `replace-with-secure-key` |
| Classifier path | `app.py` → `MODEL_PATH` | `signature_system.pkl` |
| Scaler path | `app.py` → `SCALER_PATH` | `scaler_signature.pkl` |

> ⚠️ **Security Note:** Change the `SECRET_KEY` to a strong, random value before deploying to production.

---

## Model Training

The Jupyter notebook `signature.ipynb` contains the full model training pipeline, including:
- Dataset loading and preprocessing
- Feature extraction
- Model selection and training
- Scaler fitting and model serialization

To retrain the model, run all cells in `signature.ipynb` and ensure the output files (`signature_system.pkl` and `scaler_signature.pkl`) are placed in the project root.

---

## Known Limitations

- The model's accuracy depends entirely on the training dataset it was built on.
- Skeletonization falls back to `0.0` if `cv2.ximgproc` is unavailable (standard `opencv-python` does not include this module).
- Uploaded images are stored persistently in `static/uploads/`. Consider adding a cleanup mechanism for production use.

---

## License

This project is for educational and research purposes.
