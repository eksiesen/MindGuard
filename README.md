# 🧠 MindGuard – AI-Based Attention Tracking System

MindGuard is an intelligent attention monitoring system that analyzes user focus in real-time using computer vision, audio processing, and AI-driven interventions.

## 🚀 Features

- 🎥 Real-time face tracking with MediaPipe
- 👁️ Eye Aspect Ratio (EAR) for attention detection
- 😴 Drowsiness detection (blink rate, eye closure, yawning)
- 🧭 Head pose estimation (yaw & pitch)
- 🎯 Hybrid attention scoring system
- 📊 Real-time attention smoothing & state machine logic
- 🎤 Speech-to-text using Whisper (ASR)
- 🧠 Keyword extraction (NLP)
- ❓ Dynamic quiz generation:
  - 📚 Predefined quiz bank
  - 🤖 LLM-based question generation (Ollama)
- 🔁 Smart intervention system (quiz triggered when attention drops)
- 📝 Logging system (CSV & JSONL)

## 🧠 How It Works

1. Webcam captures facial landmarks
2. Attention score is calculated using:
   - Eye movement (EAR)
   - Head position
   - Drowsiness indicators
3. Audio is transcribed and analyzed for keywords
4. If attention drops below a threshold:
   - A quiz is triggered
   - Questions are selected based on detected topic
5. System logs all interactions for analysis

## 🛠️ Technologies

- Python
- Streamlit
- OpenCV
- MediaPipe
- Whisper (ASR)
- Ollama (LLM)
- NumPy

## 📌 Use Case

Designed as a prototype for:
- Smart classroom systems
- Student attention monitoring
- Adaptive learning environments

## ⚠️ Note

This project is a prototype and does not include production-level optimization or privacy guarantees.