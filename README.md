# 🛡️ VOXGUARD — Advanced Synthetic Voice Detection

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Framework-FastAPI-009688.svg?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/ML-PyTorch-EE4C2C.svg?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Frontend-TailwindCSS-38B2AC.svg?logo=tailwind-css" alt="TailwindCSS">
  <img src="https://img.shields.io/badge/Submission-KAYA_IIT_BHU-8A2BE2.svg" alt="Hackathon">
</div>

<br>

**VOXGUARD** is a high-performance, one-shot synthetic speech analysis engine built for the **KAYA IIT BHU Hackathon**. It provides an intuitive, glassmorphic web interface that allows users to seamlessly detect AI-generated audio and deepfakes using state-of-the-art machine learning models.

---

## 📖 Table of Contents
- [✨ Key Features](#-key-features)
- [🏗️ System Architecture](#️-system-architecture)
- [🛠️ Tech Stack](#️-tech-stack)
- [🚀 Local Installation & Setup](#-local-installation--setup)
- [📡 API Reference](#-api-reference)
- [🧠 How the ML Pipeline Works](#-how-the-ml-pipeline-works)
- [🔮 Future Roadmap](#-future-roadmap)

---

## ✨ Key Features

*   **🎙️ Live One-Shot Capture:** Utilizes the browser's Web Audio API to capture exactly 6 seconds of microphone audio locally, transmitting a single payload to prevent continuous, resource-heavy inference loops.
*   **📁 Full-Duration File Analysis:** Upload entire audio files. The engine automatically downmixes, resamples to 16kHz, and evaluates the complete recording using overlapping 4-second windows.
*   **📊 Transparent Confidence Scoring:** Doesn't just guess. It aggregates evidence across multiple audio windows, generating a final trust score based on model stability, variance, and synthetic probability.
*   **🛡️ Borderline Protection:** Built with conservative decision boundaries (Fake ≥ 0.70, Real ≤ 0.30) to output "Inconclusive" for ambiguous audio, minimizing false accusations.
*   **🎨 Glassmorphic Interface:** A deeply interactive UI featuring real-time waveform rendering, SVG ring progress bars, and reactive system health monitoring.

---

## 🏗️ System Architecture

1.  **Client-Side (Browser):** Handles audio ingestion (MediaStream API / File API), raw PCM extraction, and frontend visualizations. No inference happens in the browser.
2.  **Transport Layer:** Standard HTTP POST requests carrying raw binary `application/octet-stream` payloads with custom headers (`X-Audio-Sample-Rate`).
3.  **FastAPI Backend:** Asynchronous request handling, concurrency locking (ensuring GPU/CPU isn't overwhelmed by simultaneous requests).
4.  **Audio Preprocessing:** Normalization, signal-to-noise floor gating (RMS > 0.0035), and Librosa resampling.
5.  **Inference Engine:** PyTorch + Hugging Face Transformers running `Wav2Vec2` binary classification in batched forward passes.

---

## 🛠️ Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Backend Framework** | Python 3, FastAPI, Uvicorn |
| **Machine Learning** | PyTorch, Hugging Face `transformers` |
| **Audio Processing** | `librosa`, `numpy`, `soundfile` |
| **Frontend UI** | HTML5, JavaScript (ES6+), TailwindCSS (CDN) |

---

## 🚀 Local Installation & Setup

Follow these steps to run VOXGUARD locally on your machine.

### 1. Clone the Repository
```bash
git clone https://github.com/SuryanRaj/voxguard-synthetic-voice-detector.git
cd voxguard-synthetic-voice-detector
```

### 2. Create a Virtual Environment (Highly Recommended)
This keeps the project dependencies isolated from your main system.
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Server
Start the Uvicorn ASGI server to run the FastAPI application.
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 5. Access the Application
Open your web browser and navigate to:
👉 **`http://localhost:8000`**

---

## 📡 API Reference

VOXGUARD exposes RESTful endpoints for integration into larger systems:

*   `GET /health`
    *   Returns the status of the PyTorch model, device assignment (CPU/CUDA), and label mappings.
*   `POST /analyze-live`
    *   **Body:** Raw PCM Float32 binary data (Exactly 6 seconds).
    *   **Headers:** `X-Audio-Sample-Rate`
    *   **Returns:** JSON payload with `trust_score`, `state`, and `confidence`.
*   `POST /analyze-file`
    *   **Body:** Raw PCM Float32 binary data (Full length).
    *   **Returns:** Aggregated JSON results across the entire file duration.

---

## 🧠 How the ML Pipeline Works

The core of VOXGUARD relies on a robust data preparation and inference pipeline:
1. **Resampling:** All incoming audio is strictly resampled to `16,000 Hz` using a Kaiser-fast algorithm to match the model's training parameters.
2. **Windowing:** The audio is sliced into **4-second windows** with a **2-second stride** (50% overlap). This ensures context at the edge of cuts is not lost.
3. **Silence Rejection:** Any window with an RMS amplitude below `0.0035` is dropped to prevent the model from hallucinating on background static.
4. **Batch Inference:** The `mo-thecreator/Deepfake-audio-detection` classifier analyzes the windows in batches, outputting logits mapped to `fake` and `real` probabilities.

---

## 🔮 Future Roadmap

- [ ] **Multi-Model Ensemble:** Integrate secondary acoustic models to cross-verify borderline results.
- [ ] **Explainable AI (XAI):** Generate heatmaps over the waveform UI to show *where* exactly the synthetic anomalies were detected.
- [ ] **Speaker Diarization:** Separate multiple speakers in a single file and analyze them individually.

<br>

<div align="center">
  <i>Built with ❤️ for the KAYA IIT BHU Hackathon</i>
</div>
