from __future__ import annotations

import asyncio
import os
import time
import traceback
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


# ============================================================
# VOXGUARD — ONE-SHOT SYNTHETIC VOICE ANALYZER
# ============================================================

APP_TITLE = "VOXGUARD — Synthetic Voice Detector"
BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"

# This is the same detector that was successfully loading in your
# previous working build. Its current Hugging Face config exposes
# class 0 = fake and class 1 = real.
MODEL_ID = os.getenv(
    "VOICE_MODEL",
    "mo-thecreator/Deepfake-audio-detection",
)

TARGET_SAMPLE_RATE = 16_000

# LIVE MODE: users can submit after the four-second model minimum, and
# capture stops automatically at ten seconds. There is no continuous
# inference loop.
LIVE_CAPTURE_SECONDS = 10.0
LIVE_CAPTURE_SAMPLES = int(
    TARGET_SAMPLE_RATE * LIVE_CAPTURE_SECONDS
)

# The model works with 4-second waveform windows. A ten-second live
# recording produces four overlapping windows: 0–4, 2–6, 4–8, and 6–10 s.
MODEL_WINDOW_SECONDS = 4.0
MODEL_WINDOW_SAMPLES = int(
    TARGET_SAMPLE_RATE * MODEL_WINDOW_SECONDS
)
WINDOW_STEP_SECONDS = 2.0
WINDOW_STEP_SAMPLES = int(
    TARGET_SAMPLE_RATE * WINDOW_STEP_SECONDS
)

# File uploads are NOT duration-truncated. Every full 4-second window that
# covers the complete recording is evaluated. Batches only control memory.
INFERENCE_BATCH_SIZE = 8

# Very quiet/empty windows do not contain useful speech evidence and are
# skipped. This threshold is only a signal-quality gate, not a fake/real
# decision.
RMS_FLOOR = 0.0035

# Conservative decision bands. Borderline material is reported as
# inconclusive instead of forcing a confident label.
FAKE_THRESHOLD = 0.70
REAL_THRESHOLD = 0.30

app = FastAPI(title=APP_TITLE)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Keep CPU inference reasonable on typical laptops instead of allowing a
# large number of competing threads.
if DEVICE.type == "cpu":
    try:
        torch.set_num_threads(
            max(1, min(4, os.cpu_count() or 4))
        )
    except RuntimeError:
        pass

feature_extractor = None
model = None
MODEL_ERROR: Optional[str] = None
MODEL_LABELS: dict[str, str] = {}
FAKE_INDEX: Optional[int] = None
REAL_INDEX: Optional[int] = None

# Only one request runs model inference at a time. This keeps two browser
# tabs from competing for the same CPU/GPU model.
INFERENCE_LOCK = asyncio.Lock()


# ============================================================
# MODEL LOADING
# ============================================================

def _find_label_index(
    labels: dict[int, str],
    keywords: tuple[str, ...],
) -> Optional[int]:
    for index, label in labels.items():
        normalized = label.strip().lower()
        if any(keyword in normalized for keyword in keywords):
            return index
    return None


def load_detector() -> None:
    global feature_extractor
    global model
    global MODEL_ERROR
    global MODEL_LABELS
    global FAKE_INDEX
    global REAL_INDEX

    feature_extractor = None
    model = None
    MODEL_ERROR = None
    MODEL_LABELS = {}
    FAKE_INDEX = None
    REAL_INDEX = None

    print()
    print("=" * 72)
    print(f"Loading synthetic-speech detector: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    print("Mode: one-shot analysis — no live inference loop")
    print("=" * 72)

    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            MODEL_ID
        )

        model = AutoModelForAudioClassification.from_pretrained(
            MODEL_ID,
            use_safetensors=True,
        )

        if model.config.num_labels != 2:
            raise RuntimeError(
                "VOXGUARD expects a binary classifier, but the model reports "
                f"{model.config.num_labels} classes."
            )

        # Resolve the model's own semantic labels. We never silently guess
        # the direction of the classifier.
        raw_labels = getattr(model.config, "id2label", {}) or {}
        numeric_labels: dict[int, str] = {}

        for key, value in raw_labels.items():
            try:
                numeric_labels[int(key)] = str(value)
            except (TypeError, ValueError):
                continue

        MODEL_LABELS = {
            str(index): label
            for index, label in sorted(numeric_labels.items())
        }

        FAKE_INDEX = _find_label_index(
            numeric_labels,
            ("fake", "spoof", "synthetic", "deepfake"),
        )
        REAL_INDEX = _find_label_index(
            numeric_labels,
            ("real", "bonafide", "bona fide", "genuine", "human"),
        )

        if FAKE_INDEX is None or REAL_INDEX is None:
            raise RuntimeError(
                "The detector model did not expose explicit fake/real labels. "
                f"Loaded labels: {MODEL_LABELS or '(none)'}"
            )

        if FAKE_INDEX == REAL_INDEX:
            raise RuntimeError(
                "Fake and real labels resolved to the same classifier index."
            )

        model.to(DEVICE)
        model.eval()

        print(f"Model labels: {MODEL_LABELS}")
        print(f"Fake class index: {FAKE_INDEX}")
        print(f"Real class index: {REAL_INDEX}")
        print("Model loaded successfully!")
        print()

    except Exception as exc:
        MODEL_ERROR = str(exc)
        feature_extractor = None
        model = None
        FAKE_INDEX = None
        REAL_INDEX = None

        print()
        print("=" * 25 + " MODEL LOAD ERROR " + "=" * 25)
        print(MODEL_ERROR)
        traceback.print_exc()
        print("=" * 72)
        print()


load_detector()


def detector_ready() -> bool:
    return (
        feature_extractor is not None
        and model is not None
        and FAKE_INDEX is not None
        and REAL_INDEX is not None
    )


# ============================================================
# AUDIO UTILITIES
# ============================================================

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.nan_to_num(
        np.asarray(audio, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return np.clip(audio, -1.0, 1.0)


def audio_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(
        np.sqrt(
            np.mean(np.square(audio)) + 1e-12
        )
    )


def has_voice_activity(audio: np.ndarray) -> bool:
    """Reject near-silent windows without deciding fake vs real."""
    return audio_rms(audio) >= RMS_FLOOR


def resample_audio(
    audio: np.ndarray,
    input_sample_rate: int,
) -> np.ndarray:
    audio = normalize_audio(audio)

    if input_sample_rate == TARGET_SAMPLE_RATE:
        return audio

    if input_sample_rate < 8_000 or input_sample_rate > 96_000:
        raise ValueError(
            f"Unsupported audio sample rate: {input_sample_rate} Hz"
        )

    output = librosa.resample(
        audio,
        orig_sr=input_sample_rate,
        target_sr=TARGET_SAMPLE_RATE,
        res_type="kaiser_fast",
    )

    return np.asarray(output, dtype=np.float32)


def bytes_to_float32(raw: bytes) -> np.ndarray:
    if not raw:
        return np.empty(0, dtype=np.float32)

    if len(raw) % 4 != 0:
        raise ValueError(
            "The received PCM payload has an invalid byte length."
        )

    return np.frombuffer(
        raw,
        dtype="<f4",
    ).astype(np.float32, copy=True)


def make_full_windows(audio: np.ndarray) -> list[np.ndarray]:
    """Cover the complete recording using overlapping 4-second windows."""
    audio = normalize_audio(audio)

    if len(audio) < MODEL_WINDOW_SAMPLES:
        return []

    max_start = len(audio) - MODEL_WINDOW_SAMPLES

    starts = list(
        range(
            0,
            max_start + 1,
            WINDOW_STEP_SAMPLES,
        )
    )

    # Always include the exact ending window so the tail of an upload is
    # not omitted when its length is not an exact multiple of the stride.
    if not starts or starts[-1] != max_start:
        starts.append(max_start)

    unique_starts = []
    seen = set()

    for start in starts:
        start = int(start)
        if start not in seen:
            unique_starts.append(start)
            seen.add(start)

    return [
        audio[
            start:start + MODEL_WINDOW_SAMPLES
        ]
        for start in unique_starts
    ]


# ============================================================
# MODEL INFERENCE
# ============================================================

def classify_batch(
    windows: list[np.ndarray],
) -> list[dict[str, float]]:
    if not detector_ready():
        raise RuntimeError(
            MODEL_ERROR or "Detector model is not ready."
        )

    if not windows:
        return []

    # The windows are already fixed to 4 seconds, so the extractor can
    # process the list in one batched forward pass.
    inputs = feature_extractor(
        windows,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

    results: list[dict[str, float]] = []

    for row in probabilities:
        fake_probability = float(
            row[FAKE_INDEX].item()
        )
        real_probability = float(
            row[REAL_INDEX].item()
        )
        certainty = float(
            torch.max(row).item()
        )

        results.append({
            "fake_probability": fake_probability,
            "real_probability": real_probability,
            "certainty": certainty,
        })

    return results


def aggregate_results(
    results: list[dict[str, float]],
    source: str,
    duration_seconds: float,
    total_windows: int,
) -> dict:
    if not results:
        return {
            "ok": False,
            "state": "insufficient",
            "status": "Not enough usable speech",
            "detail": (
                "The recording was received, but its analysis windows "
                "were too quiet to provide useful model evidence."
            ),
            "trust_score": None,
            "fake_probability": None,
            "real_probability": None,
            "confidence": None,
            "windows": 0,
            "total_windows": total_windows,
            "duration_seconds": round(duration_seconds, 2),
            "source": source,
        }

    fake_scores = np.asarray(
        [item["fake_probability"] for item in results],
        dtype=np.float32,
    )
    real_scores = np.asarray(
        [item["real_probability"] for item in results],
        dtype=np.float32,
    )
    certainties = np.asarray(
        [item["certainty"] for item in results],
        dtype=np.float32,
    )

    # Average evidence across the whole recording. This avoids one unusual
    # window dominating the final result.
    fake_probability = float(np.mean(fake_scores))
    real_probability = float(np.mean(real_scores))

    model_certainty = float(np.mean(certainties))

    # Agreement tells the UI how stable the windows were. It is not a
    # calibrated probability of correctness.
    spread = float(np.std(fake_scores)) if len(fake_scores) > 1 else 0.0
    agreement = 1.0 - min(1.0, spread * 2.5)

    # Window vote is another stability signal. A window counts as a fake
    # vote only when it crosses the conservative fake threshold.
    fake_vote_ratio = float(
        np.mean(fake_scores >= FAKE_THRESHOLD)
    )
    real_vote_ratio = float(
        np.mean(fake_scores <= REAL_THRESHOLD)
    )

    confidence = float(
        np.clip(
            100.0 * (
                0.65 * model_certainty
                + 0.20 * agreement
                + 0.15 * max(
                    fake_vote_ratio,
                    real_vote_ratio,
                )
            ),
            0.0,
            100.0,
        )
    )

    trust_score = float(
        np.clip(
            real_probability * 100.0,
            0.0,
            100.0,
        )
    )

    # For a synthetic decision, require a strong average score AND majority
    # of usable windows supporting that direction. Same idea for real audio.
    if (
        fake_probability >= FAKE_THRESHOLD
        and fake_vote_ratio >= 0.60
    ):
        state = "synthetic"
        status = "Likely synthetic voice"
        detail = (
            "The analyzed windows consistently produced a high synthetic/fake score."
        )
    elif (
        fake_probability <= REAL_THRESHOLD
        and real_vote_ratio >= 0.60
    ):
        state = "real"
        status = "Likely human voice"
        detail = (
            "The analyzed windows consistently produced a lower synthetic/fake score."
        )
    else:
        state = "uncertain"
        status = "Inconclusive"
        detail = (
            "The model evidence is mixed or near the decision boundary, so VOXGUARD is not forcing a binary result."
        )

    return {
        "ok": True,
        "state": state,
        "status": status,
        "detail": detail,
        "trust_score": round(trust_score, 1),
        "fake_probability": round(fake_probability * 100.0, 1),
        "real_probability": round(real_probability * 100.0, 1),
        "confidence": round(confidence, 1),
        "windows": len(results),
        "total_windows": total_windows,
        "duration_seconds": round(duration_seconds, 2),
        "source": source,
    }


def analyze_recording(
    audio: np.ndarray,
    source: str,
) -> dict:
    if not detector_ready():
        return {
            "ok": False,
            "state": "error",
            "status": "Detector model unavailable",
            "detail": MODEL_ERROR or "Model failed to initialize.",
            "trust_score": None,
            "fake_probability": None,
            "real_probability": None,
            "confidence": None,
            "windows": 0,
            "total_windows": 0,
            "duration_seconds": round(
                len(audio) / TARGET_SAMPLE_RATE,
                2,
            ),
            "source": source,
        }

    audio = normalize_audio(audio)
    duration = len(audio) / TARGET_SAMPLE_RATE

    windows = make_full_windows(audio)

    if not windows:
        return {
            "ok": False,
            "state": "insufficient",
            "status": "Audio is shorter than the model window",
            "detail": (
                f"At least {MODEL_WINDOW_SECONDS:.0f} seconds of audio is required."
            ),
            "trust_score": None,
            "fake_probability": None,
            "real_probability": None,
            "confidence": None,
            "windows": 0,
            "total_windows": 0,
            "duration_seconds": round(duration, 2),
            "source": source,
        }

    usable_windows = [
        window
        for window in windows
        if has_voice_activity(window)
    ]

    if not usable_windows:
        return aggregate_results(
            [],
            source,
            duration,
            len(windows),
        )

    started = time.perf_counter()
    all_results: list[dict[str, float]] = []

    # Batch file windows to keep memory bounded. For the 6-second live
    # recording there are normally only two windows, so this is one pass.
    for start in range(
        0,
        len(usable_windows),
        INFERENCE_BATCH_SIZE,
    ):
        batch = usable_windows[
            start:start + INFERENCE_BATCH_SIZE
        ]
        all_results.extend(
            classify_batch(batch)
        )

    result = aggregate_results(
        all_results,
        source,
        duration,
        len(windows),
    )

    result["inference_seconds"] = round(
        time.perf_counter() - started,
        2,
    )
    result["usable_windows"] = len(usable_windows)

    return result


# ============================================================
# HTTP ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def get_frontend() -> HTMLResponse:
    if not INDEX_FILE.exists():
        return HTMLResponse(
            "index.html not found beside main.py",
            status_code=500,
        )

    return HTMLResponse(
        INDEX_FILE.read_text(encoding="utf-8")
    )


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({
        "ok": detector_ready(),
        "model": MODEL_ID,
        "device": str(DEVICE),
        "labels": MODEL_LABELS,
        "fake_index": FAKE_INDEX,
        "real_index": REAL_INDEX,
        "model_error": MODEL_ERROR,
        "live_capture_seconds": LIVE_CAPTURE_SECONDS,
        "model_window_seconds": MODEL_WINDOW_SECONDS,
        "live_mode": "capture once, analyze once",
        "file_mode": "full recording, overlapping windows",
        "target_sample_rate": TARGET_SAMPLE_RATE,
    })


@app.post("/analyze-live")
async def analyze_live(request: Request) -> JSONResponse:
    """Receive one 4-to-10-second PCM recording and return one result."""
    if not detector_ready():
        return JSONResponse(
            {
                "ok": False,
                "status": "Detector model unavailable",
                "error": MODEL_ERROR or "Model failed to initialize.",
            },
            status_code=503,
        )

    try:
        sample_rate = int(
            request.headers.get(
                "X-Audio-Sample-Rate",
                str(TARGET_SAMPLE_RATE),
            )
        )

        raw = await request.body()
        audio = bytes_to_float32(raw)

        if audio.size == 0:
            raise ValueError(
                "No microphone audio was received."
            )

        audio = await asyncio.to_thread(
            resample_audio,
            audio,
            sample_rate,
        )

        # The classifier needs at least one complete four-second window.
        # A tiny callback overshoot beyond the browser limit is ignored.
        if len(audio) < MODEL_WINDOW_SAMPLES:
            raise ValueError(
                "At least four seconds of microphone audio is required."
            )

        audio = audio[:LIVE_CAPTURE_SAMPLES]

        started = time.perf_counter()

        async with INFERENCE_LOCK:
            result = await asyncio.to_thread(
                analyze_recording,
                audio,
                "live_microphone",
            )

        result["server_roundtrip_seconds"] = round(
            time.perf_counter() - started,
            2,
        )

        return JSONResponse(result)

    except ValueError as exc:
        return JSONResponse(
            {
                "ok": False,
                "status": "Live analysis failed",
                "error": str(exc),
            },
            status_code=400,
        )
    except Exception as exc:
        print(f"Live analysis error: {exc}")
        traceback.print_exc()
        return JSONResponse(
            {
                "ok": False,
                "status": "Live analysis failed",
                "error": str(exc),
            },
            status_code=500,
        )


@app.post("/analyze-file")
async def analyze_file(request: Request) -> JSONResponse:
    """Receive a complete decoded audio recording and analyze its full duration."""
    if not detector_ready():
        return JSONResponse(
            {
                "ok": False,
                "status": "Detector model unavailable",
                "error": MODEL_ERROR or "Model failed to initialize.",
            },
            status_code=503,
        )

    try:
        sample_rate = int(
            request.headers.get(
                "X-Audio-Sample-Rate",
                str(TARGET_SAMPLE_RATE),
            )
        )

        raw = await request.body()
        audio = bytes_to_float32(raw)

        if audio.size == 0:
            raise ValueError(
                "The uploaded file contained no decoded audio samples."
            )

        audio = await asyncio.to_thread(
            resample_audio,
            audio,
            sample_rate,
        )

        duration = len(audio) / TARGET_SAMPLE_RATE

        if duration < MODEL_WINDOW_SECONDS:
            raise ValueError(
                f"The uploaded audio is {duration:.1f} seconds long. "
                f"At least {MODEL_WINDOW_SECONDS:.0f} seconds is required."
            )

        started = time.perf_counter()

        # ONE request, ONE final result. All windows belonging to the file
        # are processed before the response is returned.
        async with INFERENCE_LOCK:
            result = await asyncio.to_thread(
                analyze_recording,
                audio,
                "audio_file",
            )

        result["server_roundtrip_seconds"] = round(
            time.perf_counter() - started,
            2,
        )

        return JSONResponse(result)

    except ValueError as exc:
        return JSONResponse(
            {
                "ok": False,
                "status": "File analysis failed",
                "error": str(exc),
            },
            status_code=400,
        )
    except Exception as exc:
        print(f"Uploaded audio analysis error: {exc}")
        traceback.print_exc()
        return JSONResponse(
            {
                "ok": False,
                "status": "File analysis failed",
                "error": str(exc),
            },
            status_code=500,
        )


@app.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204)
