# 🛡️ VOXGUARD — Synthetic Voice Detector

**Built for the KAYA IIT BHU Hackathon**

VOXGUARD is a fast, one-shot synthetic speech analyzer designed to detect AI-generated audio and deepfakes. It features a dark-themed, glassmorphic UI and processes audio locally using advanced machine learning. 

## ✨ Key Features

*   **Live Microphone Capture:** Captures exactly 6 seconds of live audio and evaluates it in a single HTTP request to prevent continuous heavy inference loops.
*   **Full File Analysis:** Supports uploading complete audio recordings, analyzing the full duration using overlapping 4-second model windows to ensure nothing is missed[cite: 1, 2].
*   **Transparent Scoring:** Aggregates evidence across the recording to generate a final trust score, synthetic probability, and model stability metric[cite: 1, 2].
*   **Interactive UI:** Features real-time waveform visualization, dynamic score rings, and backend health monitoring using TailwindCSS[cite: 2].
*   **Hardware Adaptive:** Automatically utilizes CUDA GPUs if available, and optimizes CPU thread counts for standard laptops.

## 🛠️ Tech Stack

*   **Backend:** Python, FastAPI, and Uvicorn[cite: 1, 3].
*   **Machine Learning:** PyTorch and Hugging Face `transformers`[cite: 3].
*   **Model:** `mo-thecreator/Deepfake-audio-detection` (Binary classifier: Fake vs. Real).
*   **Audio Processing:** `librosa`, `numpy`, and `soundfile` for resampling all audio to a target 16,000 Hz[cite: 1, 3].
*   **Frontend:** Vanilla HTML5, JavaScript, and TailwindCSS[cite: 2].

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/voxguard.git](https://github.com/yourusername/voxguard.git)
   cd voxguard
